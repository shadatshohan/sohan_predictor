import pandas as pd
import torch as torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

import math
from torch.utils.data import Dataset
import itertools
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functools
import streamlit as st
import datetime

class FMModel(nn.Module):
    def __init__(self, n, k):
        super().__init__()

        self.w0 = nn.Parameter(torch.zeros(1))
        self.bias = nn.Embedding(n, 1)
        self.embeddings = nn.Embedding(n, k)
        
        
        with torch.no_grad(): trunc_normal_(self.embeddings.weight, std=0.01)
        with torch.no_grad(): trunc_normal_(self.bias.weight, std=0.01)

    def forward(self, X):

        # applying word dropout
        #X = torch.where(self.probs > 0.02, X, self.z)
        
        #X = torch.empty_like(X).bernoulli_(torch.empty(X.size(0),X.size(1)).uniform_(0, 1)) * X
        #X = torch.empty_like(X).bernoulli_() * X
        # if self.training:
        #   #dropout
        #   probs = torch.empty((X.size(0),X.size(1)),device=device).uniform_(0, 1)
        #   X = torch.where(probs > 0.05, X, torch.zeros_like(X, dtype=torch.int64, device = device))

        emb = self.embeddings(X)
        # calculate the interactions in complexity of O(nk) see lemma 3.1 from paper
        pow_of_sum = emb.sum(dim=1).pow(2)
        sum_of_pow = emb.pow(2).sum(dim=1)
        pairwise = (pow_of_sum-sum_of_pow).sum(1)*0.5
        bias = self.bias(X).squeeze().sum(1)
        
        return torch.sigmoid(self.w0 + bias + pairwise)*5.5
        
# copied from fastai: 
def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization."
 
    return x.normal_().fmod_(2).mul_(std).add_(mean)   




#For inference
##1.create tensor
def construct_dataloader(feature_values,batchsize):
  data_input = torch.tensor(feature_values)
  data_predict = torch.tensor([0.5]*len(feature_values)).float()
  dataset_predict = data.TensorDataset(data_input, data_predict)
  predict_dataloader = data.DataLoader(dataset_predict,batch_size=batchsize,shuffle=True)
  return predict_dataloader
    
#2.predict    
def predict(device, iterator, model):
    model.eval()
    for x,y in iterator:  
      with torch.no_grad():
          y_hat = model(x.to(device))
    concat_tensor = torch.cat((x.to(device), y_hat.unsqueeze(dim=1)), 1)
    return concat_tensor
    
#get the most similar child id if it exists, else return 0
def get_first_child(age,race,disability,placement,gender,templatechilddf):
    
    if ((templatechilddf.AGE_INDEX == age) &
                            (templatechilddf.RACE_INDEX == race) &
                            (templatechilddf.DISABILITY_INDEX == disability) & 
                            (templatechilddf.PLACEMENT_INDEX == placement) &
                            (templatechilddf.GENDER_INDEX == gender)).any():
        child_index = templatechilddf[(templatechilddf.AGE_INDEX == age) &
                            (templatechilddf.RACE_INDEX == race) &
                            (templatechilddf.DISABILITY_INDEX == disability) & 
                            (templatechilddf.PLACEMENT_INDEX == placement) &
                            (templatechilddf.GENDER_INDEX == gender)].iloc[0].CHILD_INDEX
        return child_index
    else:
        return 0  
          
#to get recommendations for the user input 
#1.get look up values from UI
def get_lookups(templatechilddf, agelookupdf, racelookupdf, disabilitylookupdf, placementlookupdf, genderlookupdf, age = '13 and above',race = 'African American',disability = 'No Disability',placement = '5 and above', gender = 'Male'):
    
    ageid = int(agelookupdf[agelookupdf.AGE_DEF == age].AGE_PLACED_INDEX)
    raceid = int(racelookupdf[racelookupdf.RACE_DEF == race].RACE_GROUP_INDEX)
    disability = int(disabilitylookupdf[disabilitylookupdf.DISABILITY_DEF == disability].DISABILITY_GROUP_INDEX)
    placement = int(placementlookupdf[placementlookupdf.PLACEMENT_DEF == placement].PLACEMENT_NUMBER_INDEX)
    gender = int(genderlookupdf[genderlookupdf.GENDER_DEF == gender].GENDER_INDEX)
    childid = get_first_child(ageid,raceid,disability,placement,gender,templatechilddf)
    return childid,ageid,raceid,disability,placement,gender

#2. get recommendations
@st.cache
def get_recommendations(modelinfer, device, providers, provider_biases, provider_embeddings, short_term = 0, childid = 0,raceid = 167737, ageid = 167734, disability = 167743, placement = 167746, gender = 167748, topN = 20):
    child_embedding = modelinfer.embeddings(torch.tensor(childid,device=device))
    race_embedding = modelinfer.embeddings(torch.tensor(raceid,device=device)) 
    age_embedding = modelinfer.embeddings(torch.tensor(ageid,device=device))
    disability_embedding = modelinfer.embeddings(torch.tensor(disability,device=device))
    placement_embedding = modelinfer.embeddings(torch.tensor(placement,device=device))
    gender_embedding = modelinfer.embeddings(torch.tensor(gender,device=device))
    metadata_embedding = child_embedding+ race_embedding+age_embedding+disability_embedding+placement_embedding+gender_embedding
    rankings = provider_biases.squeeze()+(metadata_embedding*provider_embeddings).sum(1)
#    pdetailed = pd.DataFrame([i for i in providers.iloc[rankings.argsort(descending=True).cpu()][['PROVIDER_INDEX','PROVIDER_ID','provider_name_confidential','MAX_SETTING','FLAGS']].values][:topN], columns = ['PROVIDER_INDEX','PROVIDER_ID','provider_name_confidential','MAX_SETTING','FLAGS'])

#    plist = [i for i in providers.iloc[rankings.argsort(descending=True).cpu()]['PROVIDER_INDEX'].values][:topN]
    if (short_term == 1):
        plist = [i for i in providers.iloc[rankings.argsort(descending=True).cpu()][providers.FLAGS.str.contains('Night')]['PROVIDER_INDEX'].values][:topN]
    else:
        plist = [i for i in providers.iloc[rankings.argsort(descending=True).cpu()]['PROVIDER_INDEX'].values][:topN]

    feature_values = []
    for i in range(topN):
        feature_values.append([childid,plist[i],ageid,raceid,disability,placement,gender])

    pdata = construct_dataloader(feature_values,topN)
    y_pred = predict(device,pdata,modelinfer)
    px = pd.DataFrame(y_pred.cpu().numpy(), columns = ['CHILD_INDEX','PROVIDER_INDEX','AGE_PLACED_INDEX','RACE_GROUP_INDEX','DISABILITY_GROUP_INDEX','PLACEMENT_NUMBER_INDEX', 'GENDER_INDEX','RATING'])
    return(pd.merge(providers, px, on = ['PROVIDER_INDEX'])[['PROVIDER_ID','provider_name_confidential','MAX_SETTING','FLAGS','RATING']].sort_values(by=['RATING'],ascending=False))


#load datasets
@st.cache
def load_and_prep_datasets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #set path 
    PATH = "."

    #load data 
    ratingsdf = pd.read_csv(PATH + '/data/ratingsprocessedpipeline.csv', low_memory=False)

    #load lookups
    agelookupdf = pd.read_csv(PATH + '/data/agelookup.csv', low_memory=False)
    racelookupdf = pd.read_csv(PATH + '/data/racelookup.csv', low_memory=False)
    disabilitylookupdf = pd.read_csv(PATH + '/data/disabilitylookup.csv', low_memory=False)
    placementlookupdf = pd.read_csv(PATH + '/data/placementlookup.csv', low_memory=False)
    genderlookupdf = pd.read_csv(PATH + '/data/genderlookup.csv', low_memory=False)
    templatechilddf = pd.read_csv(PATH + '/data/templatechilddf.csv', low_memory=False)
	
    #set feature columns and lengths
    feature_columns = ['CHILD_INDEX','PROVIDER_INDEX','AGE_PLACED_INDEX','RACE_GROUP_INDEX','DISABILITY_GROUP_INDEX','PLACEMENT_NUMBER_INDEX', 'GENDER_INDEX']
    lenmodel = ratingsdf[feature_columns].values.max() + 1
    lenfeatures = 120
    return device, templatechilddf, ratingsdf, agelookupdf, racelookupdf, disabilitylookupdf, placementlookupdf, genderlookupdf, lenmodel, lenfeatures



#load model 
@st.cache
def load_model(lenmodel, lenfeatures, device):
    #set path 
    PATH = "."
    modelinfer = FMModel(lenmodel, lenfeatures)
    modelinfer.load_state_dict(torch.load(PATH +'/model/fmstatedictRV2',map_location=torch.device('cpu')))
    modelinfer.to(device).eval()
    return modelinfer


#load providers
@st.cache(allow_output_mutation=True)
def load_providers(ratingsdf, modelinfer, device):
    providers = ratingsdf.drop_duplicates('PROVIDER_INDEX').copy()
    provider_embeddings = modelinfer.embeddings(torch.tensor(providers['PROVIDER_INDEX'].values,device=device).long())
    providers['embedding'] = provider_embeddings.tolist()
    provider_biases = modelinfer.bias(torch.tensor(providers['PROVIDER_INDEX'].values,device=device).long())
    providers['bias'] = provider_biases.cpu().detach().numpy()
    return providers, provider_biases, provider_embeddings

@st.cache
def regroup_age(birthday):
    diff = datetime.datetime.now().date() - birthday
    age = math.floor(diff.days / 365)
    if age <= 5:
        age_group = '0-5'
    elif age <= 12:
        age_group = '6-13'
    else:
        age_group = '13 and above'
    return age_group

@st.cache
def regroup_race(race, hispanic):
    if hispanic == 'Yes':
        race_group = 'Hispanic'
    elif race == 'White':
        race_group = 'White'
    elif race == 'Black':
        race_group = 'African American'
    elif race == 'Asian':
        race_group = 'Asian'
    elif race == 'Pacific Islander':
        race_group = 'Hawaiian'
    elif race == 'Native American':
        race_group = 'American Indian'
    elif race == 'Multi-Racial':
        race_group = 'Multi Racial'
    else:
        race_group = 'Unable to determine'
    return race_group

@st.cache
def regroup_placement(prev_placement):
    if prev_placement <= 4:
        placement_group = '1-4'
    else:
        placement_group = '5 and above'
    return placement_group

@st.cache
def regroup_disability(child_clindis, child_mr_flag, child_vishear_flag, child_phydis_flag, child_emotdist_flag, child_othermed_flag):
    if child_clindis != 'Yes':
        disability_group = 'No Disability'
    elif child_vishear_flag == True or child_phydis_flag == True or child_othermed_flag == True:
        disability_group = 'Clinical Disability'
    elif child_mr_flag == True or child_emotdist_flag == True:
        disability_group = 'Behavioral Disability'
    else:
        disability_group = 'Clinical Disability'
    return disability_group


@st.cache
def regroup_gender(child_gender):
    if child_gender == 'Male':
        gender_group = 'Male'
    elif child_gender == 'Female':
        gender_group = 'Female'
    return gender_group


















