

from pathlib import Path
import base64

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
from xgboost import XGBClassifier, XGBRegressor
import torch as torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import FMModel as FMModel
import DurationModel as DurationModel

from altair.vegalite.v4.schema.channels import X
import altair as alt
import geopandas as gpd
from PIL import Image
import streamlit.components.v1 as components
# Initial page config

st.set_page_config(
     page_title='Foster Care Abuse Risk',
     layout="wide",
     initial_sidebar_state="expanded",
)

##########################
# htmls
##########################

HTML_WRAPPER1 = """<div style="background-color: #EAFAF1; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
HTML_WRAPPER2 = """<div style="background-color: #ededed; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
#HTML_WRAPPER3 = """<div style="background-color: #FEF9E7; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
HTML_WRAPPER3 = """<div style="background-color: #EAFAF1; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
WHITECOLOR= '<p style="font-family:Courier; color:White; font-size: 20px;">....</p>'
WHITECOLORsmall= '<p style="font-family:Courier; color:White; font-size: 11px;">....</p>'
BANNER= '<p style="font-family:Helvetica Neue; color:Teal; font-size: 55px; line-height:25px;text-align: center;"><b>Foster Care Abuse Risk</b></p>'
BANNERsmall= '<p style="font-family:Arial; color:Teal; font-size: 20px;text-align: center;">Risk Assesments</p>'
BANNERleft= '<p style="font-family:Helvetica Neue; color:Teal; font-size: 55px; line-height:25px;text-align: left;"><b>Foster Care Abuse Risk</b></p>'
BANNERleftsmall= '<p style="font-family:Arial; color:Teal; font-size: 20px;text-align: left;">Powered by XGBoost</p>'
SIDEBARHEADING= '<p style="font-family:Arial; color:Teal; font-size: 20px;text-align: left;"><b>Foster Care Abuse Risk</b></p>'

vis_link_one = '''
<div class='tableauPlaceholder' id='viz1669351327460' style='position: relative'><noscript><a href='#'><img alt='Total Foster Care Cases Map ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;7Z&#47;7ZMZXM9DR&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;7ZMZXM9DR' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;7Z&#47;7ZMZXM9DR&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                
<script type='text/javascript'>                    
var divElement = document.getElementById('viz1669351327460');                    
var vizElement = divElement.getElementsByTagName('object')[0];                    
if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';
vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';
vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='2927px';}                     
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
vizElement.parentNode.insertBefore(scriptElement, vizElement);
 </script>
'''
vis_link_two = ''' <div class='tableauPlaceholder' id='viz1669352278204' style='position: relative'><noscript><a href='#'><img alt=' Neglect Ratio Cases ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fo&#47;FosterCareNeglectRatioCross-tab-UCBerkeleyMIDSCapstone&#47;CrosstabNumberNeglectcasesMap&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FosterCareNeglectRatioCross-tab-UCBerkeleyMIDSCapstone&#47;CrosstabNumberNeglectcasesMap' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fo&#47;FosterCareNeglectRatioCross-tab-UCBerkeleyMIDSCapstone&#47;CrosstabNumberNeglectcasesMap&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes%5C' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1669352278204');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>'''
v_3 = '''<div class='tableauPlaceholder' id='viz1669354708856' style='position: relative'><noscript><a href='#'><img alt=' Neglect Ratio Cases Heatmap ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fo&#47;FosterCareNeglectHeatmap-UCBerkeleyMIDSCapstone&#47;HeatmapNumberNeglectcasesMap2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FosterCareNeglectHeatmap-UCBerkeleyMIDSCapstone&#47;HeatmapNumberNeglectcasesMap2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fo&#47;FosterCareNeglectHeatmap-UCBerkeleyMIDSCapstone&#47;HeatmapNumberNeglectcasesMap2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes%5C' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1669354708856');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script> '''
v4 = '''<div class='tableauPlaceholder' id='viz1669354994885' style='position: relative'><noscript><a href='#'><img alt='Ratio of Foster Cases Compared To Population ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ra&#47;RatioofFosterCasesComparedtoPopulationHeatmap-UCBerkeleyMIDSCapstone&#47;RatioofFosterCasesComparedToPopulation&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='RatioofFosterCasesComparedtoPopulationHeatmap-UCBerkeleyMIDSCapstone&#47;RatioofFosterCasesComparedToPopulation' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ra&#47;RatioofFosterCasesComparedtoPopulationHeatmap-UCBerkeleyMIDSCapstone&#47;RatioofFosterCasesComparedToPopulation&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes%5C' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1669354994885');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>'''

st.title('USA Foster Care Risk Analysis Dashboard')
components.html(vis_link_one, width=1400, height=600, scrolling=True)
st.write('Visual one')
st.write(''' **Situation:** Surveys of child welfare practitioners in the foster care space identified the prevention of abuse and neglect of foster children as a key area for research. In the state of Virginia alone, child welfare practitioners receive nearly 38,000 reports of child abuse a year. Of those 38,000, roughly 10,000 investigations are conducted. Of those 10,000, roughly 3,000 are founded investigations. This means that out of 38,000 reports, there is only a 7.8% ratio of founded investigations.

Task: The availability of data related to foster cases enables our team to develop a trend analysis dashboard and triaging model to aid child welfare practitioners in analysing trends and risk factors associated with abuse.

Action: Our team will use public AFCARS data to visualize trends associated with abuse and neglect in foster cases at the national and state level, in addition to the development of a tree-based classification model for approximated triaging of foster cases in a sandbox environment.

Result: The goal of this project is to develop an easy-to-use educational tool that increases the ratio of founded investigations into physical and sexual abuse in the foster care system. 
''')
components.html(vis_link_two,width=1400, height=600, scrolling=True)
st.write('Visual two')
components.html(v_3,width=1400, height=600, scrolling=True)
st.write('Visual Three')
components.html(v4,width=1400, height=600, scrolling=True)
st.write('Visual Four')
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def set_session_resetter():
	st.session_state['resetter'] = False
    

	

def main():
    my_page = cs_sidebar()
    if my_page == 'Home': 
    	cs_home()
    elif my_page == 'Risk Assesment':
    	cs_body()
    elif my_page == 'Journey':
    	cs_journey()
    elif my_page == 'Architecture':
    	cs_architecture()
    elif my_page == 'Performance':
    	cs_performance()
    return None

# Thanks to streamlitopedia for the following code snippet

# sidebar

def cs_sidebar():
	#st.markdown( """ <style> .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; } </style> """, unsafe_allow_html=True, )
	st.sidebar.write(SIDEBARHEADING,unsafe_allow_html=True)
	#set_png_as_page_bg('father-and-daughters-hand.jpeg')
#	header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#	img_to_bytes("father-and-daughters-hand.jpeg")
#	)
#	st.sidebar.markdown(
#	header_html, unsafe_allow_html=True,
#	)
#	image = Image.open('TopBanner6.png')
#	st.image(image,  width=600 ) #
#	st.write(BANNER,unsafe_allow_html=True) 
#	st.write(BANNERsmall,unsafe_allow_html=True) 


	mypage = st.sidebar.radio(' ', ['Home', 'Risk Assesment', 'Performance'])

	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')



	return mypage


def cs_body():
    st.write(BANNERleft,unsafe_allow_html=True) 
    st.write(BANNERleftsmall,unsafe_allow_html=True) 

    col1, col2, col3 = st.columns(3)

    placed_before = 'Select one'
    num_prev_placements = 0
    child_num_prev_placements_good = 0
    child_num_prev_placements_bad = 0
    child_date_of_first_placement = datetime.date(2015,1,1)
    child_recent_placement_outcome = 'Select one'
    child_ctkfamst = 'Select one'
    child_caretakerage = float("Nan")
    child_hispanic = 'Select one'
    child_mr_flag = False
    child_vishear_flag = False
    child_phydis_flag = False
    child_emotdist_flag = False
    child_othermed_flag = False
    child_clindis = 'Select one'
    child_everadpt = 'Select one'
    child_everadpt_age = float("Nan")
    current_case_goal = 'Select one'
    find_providers_button = None
#    ## need to add this so session state if resetter in session state blah blah blah
    if 'resetter' not in st.session_state:
        st.session_state['resetter'] = False

    col1.header("Child Information")
    col2.write(WHITECOLORsmall, unsafe_allow_html=True)
    col2.write(WHITECOLORsmall, unsafe_allow_html=True)

    child_birthday = col1.date_input("Child's birthday- Not Mandatory ", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now(),on_change = set_session_resetter)
    child_gender = col2.selectbox("Child's gender", ['Select one', 'Male', 'Female'], on_change = set_session_resetter)
    child_race = col1.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'], on_change = set_session_resetter)
    child_hispanic = col2.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'],on_change = set_session_resetter)
    child_caretakerage = col1.number_input("Primary caretaker's age at the time of child's removal", min_value = 0, max_value = 100, step = 1,on_change = set_session_resetter)
    child_ctkfamst = col2.selectbox("What kind of caretaker was the child removed from?", ['Select one', 'Married Couple', 'Unmarried Couple', 'Single Female', 'Single Male'],on_change = set_session_resetter)


    if child_ctkfamst != 'Select one':
        col1.header("Prior Placement Information")
        col2.write(WHITECOLORsmall, unsafe_allow_html=True)
        col2.write(WHITECOLORsmall, unsafe_allow_html=True)
        placed_before = col1.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'], on_change = set_session_resetter)

        if placed_before == 'Yes':
             num_prev_placements = col2.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1, on_change = set_session_resetter)
        else:
            col2.subheader(" ")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
 
        if num_prev_placements > 0:
            child_num_prev_placements_good = col1.number_input('Previous placements with POSITIVE outcome', min_value = 0, max_value = num_prev_placements, step = 1,on_change = set_session_resetter)
            child_num_prev_placements_bad = col2.number_input('Previous placements with NEGATIVE outcome', min_value = 0, max_value = num_prev_placements, step = 1,on_change = set_session_resetter)

            child_date_of_first_placement = col1.date_input("First Placement Start Date", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now(),on_change = set_session_resetter)
            child_recent_placement_outcome = col2.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'],on_change = set_session_resetter)

        if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
            child_iswaiting = col1.selectbox("Is the child currently waiting for adoption?", ['Select one', 'Yes', 'No'],on_change = set_session_resetter)
            child_everadpt = col2.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'],on_change = set_session_resetter)

        if child_everadpt == 'Yes':
            child_everadpt_age = col1.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18,on_change = set_session_resetter)
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.subheader("")
   
                
    if  child_everadpt != 'Select one':

        col1.text("")
        col1.header("Disability Information")
        col2.write(WHITECOLORsmall,unsafe_allow_html=True) 
        col2.write(WHITECOLORsmall,unsafe_allow_html=True) 
        child_clindis = col1.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'],on_change = set_session_resetter)

        if child_clindis == 'Yes':
            col2.text("")
            col2.write("Check all that apply:")

            child_phydis_flag = col2.checkbox("Physically Disabled",on_change = set_session_resetter)
            child_vishear_flag = col2.checkbox("Visually or Hearing Impaired",on_change = set_session_resetter)
            col1.text("")
            col1.text("")
            child_mr_flag = col2.checkbox("Intellectually Disabled",on_change = set_session_resetter)
            child_emotdist_flag = col2.checkbox("Emotionally Disturbed",on_change = set_session_resetter)
            child_othermed_flag = col2.checkbox("Other Medically Diagnosed Condition",on_change = set_session_resetter)
            col2.text("")
                
            
        if ((child_clindis == 'Yes' and (child_mr_flag or child_vishear_flag or child_phydis_flag or child_emotdist_flag or child_othermed_flag))
            or  child_clindis == 'No' or child_clindis == 'Not yet determined'):
        
            if child_clindis !='Yes':
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("") 
                col2.text("")
                col2.text("")
                col2.text("")
     
            else:   
                col1.text("")
                col1.text("")
                col1.text("")
                col1.text("")
                col1.text("")
                col1.text("")
                col2.text("")
     

            col1.header("Removal Reasons")
            col2.write(WHITECOLOR, unsafe_allow_html=True)
            col2.write(WHITECOLORsmall,unsafe_allow_html=True)
            col1.write("Why did the child enter the foster care system? (Check all that apply)")
    

            physical_abuse = col1.checkbox('Physical Abuse',on_change = set_session_resetter)
            sexual_abuse = col1.checkbox('Sexual Abuse',on_change = set_session_resetter)
            emotional_abuse_neglect = col1.checkbox('Emotional Abuse',on_change = set_session_resetter)
            physical_neglect = col1.checkbox("Physical Neglect")
            medical_neglect = col1.checkbox("Medical Neglect",on_change = set_session_resetter)
            alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse",on_change = set_session_resetter)
            drug_abuse_child = col1.checkbox("Child's Drug Abuse",on_change = set_session_resetter)
            child_behavior_problem = col1.checkbox('Child Behavior Problem',on_change = set_session_resetter)
            child_disability = col1.checkbox('Child Disability',on_change = set_session_resetter)
            transition_to_independence = col1.checkbox("Transition to Independence",on_change = set_session_resetter)
            inadequate_supervision = col1.checkbox("Inadequate Supervision",on_change = set_session_resetter)
            adoption_dissolution = col1.checkbox("Adoption Dissolution",on_change = set_session_resetter)
            abandonment = col1.checkbox("Abandonment",on_change = set_session_resetter)
            labor_trafficking = col1.checkbox("Labor Trafficking")
            sexual_abuse_sexual_exploitation = col1.checkbox("Sexual Exploitation",on_change = set_session_resetter)

            prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse",on_change = set_session_resetter)
            prospective_sexual_abuse = col2.checkbox('Prospective Sexual Abuse',on_change = set_session_resetter)
            prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse",on_change = set_session_resetter)
            prospective_physical_neglect = col2.checkbox('Prospective Physical Neglect',on_change = set_session_resetter)
            prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect",on_change = set_session_resetter)
            alcohol_abuse_parent = col2.checkbox("Parent's Alcohol Abuse",on_change = set_session_resetter)
            drug_abuse_parent = col2.checkbox("Parent's Drug Abuse",on_change = set_session_resetter)
            incarceration_of_parent = col2.checkbox('Incarceration of Parent',on_change = set_session_resetter)
            death_of_parent = col2.checkbox('Death of Parent',on_change = set_session_resetter)
            domestic_violence = col2.checkbox("Domestic Violence",on_change = set_session_resetter)
            inadequate_housing = col2.checkbox("Inadequate Housing",on_change = set_session_resetter)
            caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope",on_change = set_session_resetter)
            relinquishment = col2.checkbox('Relinquishment',on_change = set_session_resetter)
            request_for_service = col2.checkbox('Request for Service',on_change = set_session_resetter)
            csec = col2.checkbox("CSEC",on_change = set_session_resetter)


            col1.header("Current placement information")
            current_case_goal = col1.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'],on_change = set_session_resetter)
        
        if current_case_goal != 'Select one':
            col1.text("")
            col1.write("Current placement's applicable payments",on_change = set_session_resetter)
            current_case_ivefc = col1.checkbox("Foster Care Payments",on_change = set_session_resetter)
            current_case_iveaa = col1.checkbox("Adoption Assistance",on_change = set_session_resetter)
            current_case_ivaafdc = col1.checkbox("TANF Payment (Temporary Assistance for Needy Families)",on_change = set_session_resetter)
            current_case_ivdchsup = col1.checkbox("Child Support Funds",on_change = set_session_resetter)
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("") 
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            current_case_xixmedcd = col2.checkbox("Medicaid",on_change = set_session_resetter)
            current_case_ssiother = col2.checkbox("SSI or Social Security Benefits",on_change = set_session_resetter)
            current_case_noa = col2.checkbox("Only State or Other Support",on_change = set_session_resetter)
            current_case_payments_none = col2.checkbox("None of the above apply",on_change = set_session_resetter)
            current_case_fcmntpay = col1.number_input("Monthly Foster Care Payment ($)", min_value = 200, step = 100,on_change = set_session_resetter)
            col1.text("")
            col1.text("")
            
            find_providers_button = st.button("Assesment of Risk")

#        ## Once the button is pressed, the resetter will be set to True and will be updated in the Session State
#        ## Recommender System output    
        if find_providers_button:
            if child_gender == 'Select one' or child_race == 'Select one' or child_clindis == 'Select one':
                st.error('Please fill in child\'s gender and race')
            else:
                st.session_state['resetter'] = True
			
            ## construct child record using user_input
        if st.session_state['resetter'] == True:
            child_input_record_data = {
            'PHYSICAL_ABUSE':[1.0 if physical_abuse else 0.0]
            ,'SEXUAL_ABUSE':[1.0 if sexual_abuse else 0.0]
            ,'EMOTIONAL_ABUSE_NEGLECT':[1.0 if emotional_abuse_neglect else 0.0]
            ,'ALCOHOL_ABUSE_CHILD':[1.0 if alcohol_abuse_child else 0.0]
            ,'DRUG_ABUSE_CHILD':[1.0 if drug_abuse_child else 0.0]
            ,'ALCOHOL_ABUSE_PARENT':[1.0 if alcohol_abuse_parent else 0.0]
            ,'DRUG_ABUSE_PARENT':[1.0 if drug_abuse_parent else 0.0]
            ,'PHYSICAL_NEGLECT':[1.0 if physical_neglect else 0.0]
            ,'DOMESTIC_VIOLENCE':[1.0 if domestic_violence else 0.0]
            ,'INADEQUATE_HOUSING':[1.0 if inadequate_housing else 0.0]
            ,'CHILD_BEHAVIOR_PROBLEM':[1.0 if child_behavior_problem else 0.0]
            ,'CHILD_DISABILITY':[1.0 if child_disability else 0.0]
            ,'INCARCERATION_OF_PARENT':[1.0 if incarceration_of_parent else 0.0]
            ,'DEATH_OF_PARENT':[1.0 if death_of_parent else 0.0]
            ,'CAREGIVER_INABILITY_TO_COPE':[1.0 if caregiver_inability_to_cope else 0.0]
            ,'ABANDONMENT':[1.0 if abandonment else 0.0]
            ,'TRANSITION_TO_INDEPENDENCE':[1.0 if transition_to_independence else 0.0]
            ,'INADEQUATE_SUPERVISION':[1.0 if inadequate_supervision else 0.0]
            ,'PROSPECTIVE_EMOTIONAL_ABUSE_NEGLECT':[1.0 if prospective_emotional_abuse_neglect else 0.0]
            ,'PROSPECTIVE_MEDICAL_NEGLECT':[1.0 if prospective_medical_neglect else 0.0]
            ,'PROSPECTIVE_PHYSICAL_ABUSE':[1.0 if prospective_physical_abuse else 0.0]
            ,'PROSPECTIVE_PHYSICAL_NEGLECT':[1.0 if prospective_physical_neglect else 0.0]
            ,'PROSPECTIVE_SEXUAL_ABUSE':[1.0 if prospective_sexual_abuse else 0.0]
            ,'RELINQUISHMENT':[1.0 if relinquishment else 0.0]
            ,'REQUEST_FOR_SERVICE':[1.0 if request_for_service else 0.0]
            ,'ADOPTION_DISSOLUTION':[1.0 if adoption_dissolution else 0.0]
            ,'MEDICAL_NEGLECT':[1.0 if medical_neglect else 0.0]
            ,'CSEC':[1.0 if csec else 0.0]
            ,'LABOR_TRAFFICKING':[1.0 if labor_trafficking else 0.0]
            ,'SEXUAL_ABUSE_SEXUAL_EXPLOITATION':[1.0 if sexual_abuse_sexual_exploitation else 0.0]
            ,'RACE_WHITE':[1.0 if child_race == 'White' else 0.0]
            ,'RACE_BLACK':[1.0 if child_race == 'Black' else 0.0]
            ,'RACE_ASIAN':[1.0 if child_race == 'Asian' else 0.0]
            ,'RACE_UNKNOWN':[0.0]
            ,'RACE_HAWAIIAN':[1.0 if child_race == 'Pacific Islander' else 0.0]
            ,'RACE_AMERICAN_INDIAN':[1.0 if child_race == 'Native American' else 0.0]
            ,'RACE_MULTI_RCL':[1.0 if child_race == 'Multi-Racial' else 0.0]
            ,'HISPANIC':[1.0 if child_hispanic == 'Yes' else 2.0]
            ,'AGE_AT_PLACEMENT_BEGIN':[round((datetime.datetime.date(datetime.datetime.now()) - child_birthday).days / 365, 2)]
            ,'NEW_REMOVAL':[1.0 if placed_before == 'Yes' else 0.0]
            #     #,REMOVAL_LENGTH #Need to make reflective as of placement begin date   
            #     #,PLACEMENT_NUMBER #Need to apply after using James's flattened version
            ,'CHILD_NUM_PREV_PLACEMENTS':[float(num_prev_placements)]
            ,'CHILD_NUM_PREV_PLACEMENTS_GOOD':[float(child_num_prev_placements_good)]
            ,'CHILD_NUM_PREV_PLACEMENTS_NEUTRAL':[max(float(num_prev_placements - child_num_prev_placements_good - child_num_prev_placements_bad), 0)]
            ,'CHILD_NUM_PREV_PLACEMENTS_BAD':[float(child_num_prev_placements_bad)]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_1.0':[1 if child_recent_placement_outcome == 'Positive' else 0]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_2.0':[1 if child_recent_placement_outcome == 'Neutral' else 0]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_3.0':[1 if child_recent_placement_outcome == 'Negative' else 0]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_nan':[0]
            ,'CHILD_DAYS_SINCE_FIRST_PLACEMENT':[float((datetime.datetime.date(datetime.datetime.now()) - child_date_of_first_placement).days)]
            ,'CHILD_NUM_PREV_PLACEMENTS_GOOD_PERC':[round(child_num_prev_placements_good / float("Nan") if num_prev_placements == 0 else num_prev_placements,6)]
            ,'CHILD_NUM_PREV_PLACEMENTS_NEUTRAL_PERC':[round(max(float(num_prev_placements - child_num_prev_placements_good - child_num_prev_placements_bad), 0) / float("Nan") if num_prev_placements == 0 else num_prev_placements,6)]
            ,'CHILD_NUM_PREV_PLACEMENTS_BAD_PERC':[round(child_num_prev_placements_bad / float("Nan") if num_prev_placements == 0 else num_prev_placements,6)]
            #     #,'MOVE_MILES'
            #     #,'ROOMMATE_COUNT'
            ,'IVEFC':[float("Nan") if current_case_payments_none else (1.0 if current_case_ivefc else 0.0)]
            ,'IVEAA':[float("Nan") if current_case_payments_none else (1.0 if current_case_iveaa else 0.0)]
            ,'IVAAFDC':[float("Nan") if current_case_payments_none else (1.0 if current_case_ivaafdc else 0.0)]
            ,'IVDCHSUP':[float("Nan") if current_case_payments_none else (1.0 if current_case_ivdchsup else 0.0)]
            ,'XIXMEDCD':[float("Nan") if current_case_payments_none else (1.0 if current_case_xixmedcd else 0.0)]
            ,'SSIOTHER':[float("Nan") if current_case_payments_none else (1.0 if current_case_ssiother else 0.0)]
            ,'NOA':[float("Nan") if current_case_payments_none else (1.0 if current_case_noa else 0.0)]
            ,'FCMNTPAY'  :[float("Nan") if current_case_payments_none else float(current_case_fcmntpay)]
            ,'CLINDIS':[1.0 if child_clindis == 'Yes' else 2.0]
            ,'MR':[1.0 if child_mr_flag else 0.0]
            ,'VISHEAR':[1.0 if child_vishear_flag else 0.0]
            ,'PHYDIS':[1.0 if child_phydis_flag else 0.0]
            ,'EMOTDIST':[1.0 if child_emotdist_flag else 0.0]
            ,'OTHERMED':[1.0 if child_othermed_flag else 0.0]
            ,'CASEGOAL_1':[1.0 if current_case_goal == 'Reunification' else 0.0]
            ,'CASEGOAL_2':[1.0 if current_case_goal == 'Live with Other Relatives' else 0.0]
            ,'CASEGOAL_3':[1.0 if current_case_goal == 'Adoption' else 0.0]
            ,'CASEGOAL_4':[1.0 if current_case_goal == 'Long Term Foster Care' else 0.0]
            ,'CASEGOAL_5':[1.0 if current_case_goal == 'Emancipation' else 0.0]
            ,'CASEGOAL_6':[1.0 if current_case_goal == 'Guardianship' else 0.0]
            ,'CASEGOAL_7':[1.0 if current_case_goal == 'Goal Not Yet Established' else 0.0]
            ,'CASEGOAL_99':[0.0]
            ,'ISWAITING':[1.0 if child_iswaiting == 'Yes' else 0.0]
            ,'EVERADPT_1.0':[1.0 if child_everadpt == 'Yes' else 0.0]
            ,'EVERADPT_2.0':[1.0 if child_everadpt == 'No' else 0.0]
            #     #,'EVERADPT_3.0'
            ,'AGEADOPT_0.0':[1.0 if child_everadpt != 'Yes' else 0.0]
            ,'AGEADOPT_1.0':[1.0 if child_everadpt_age <= 2 else 0.0]
            ,'AGEADOPT_2.0':[1.0 if 2 < child_everadpt_age <= 5 else 0.0]
            ,'AGEADOPT_3.0':[1.0 if 5 < child_everadpt_age <= 12 else 0.0]
            ,'AGEADOPT_4.0':[1.0 if 12 < child_everadpt_age else 0.0]
            #     #,'AGEADOPT_5.0'
            #     #,'AGEADOPT_nan'
            ,'CTKFAMST_1.0':[1.0 if child_ctkfamst == 'Married Couple' else 0.0]
            ,'CTKFAMST_2.0':[1.0 if child_ctkfamst == 'Unmarried Couple' else 0.0]
            ,'CTKFAMST_3.0':[1.0 if child_ctkfamst == 'Single Female' else 0.0]
            ,'CTKFAMST_4.0':[1.0 if child_ctkfamst == 'Single Male' else 0.0]
            ,'CARETAKER_AGE':[float(child_caretakerage)]
            }
            #st.write(child_input_record_data)
            # Create child record input dataframe
            child_input_record_df = pd.DataFrame(child_input_record_data)


            ### RUN RECOMMENDER MODEL ###
            #regroup relevant user input for recommender model
            input_age = FMModel.regroup_age(child_birthday)
            input_race = FMModel.regroup_race(child_race, child_hispanic)
            input_placement = FMModel.regroup_placement(num_prev_placements)
            input_disability = FMModel.regroup_disability(child_clindis, child_mr_flag, child_vishear_flag, child_phydis_flag, child_emotdist_flag, child_othermed_flag)
            input_gender = FMModel.regroup_gender(child_gender)

            #loading configuration and datasets
            device, templatechilddf, ratingsdf, agelookupdf, racelookupdf, disabilitylookupdf, placementlookupdf, genderlookupdf, lenmodel, lenfeatures = FMModel.load_and_prep_datasets()

            #loading the model
            modelinfer = FMModel.load_model(lenmodel = lenmodel, lenfeatures = lenfeatures, device = device)

            #load providers 
            providers, provider_biases, provider_embeddings = FMModel.load_providers(ratingsdf = ratingsdf, modelinfer = modelinfer, device = device)

            #get user parameters from UI 
            childid, ageid,raceid,disability,placement,gender = FMModel.get_lookups(templatechilddf = templatechilddf, agelookupdf = agelookupdf, racelookupdf = racelookupdf, disabilitylookupdf = disabilitylookupdf, placementlookupdf = placementlookupdf, genderlookupdf = genderlookupdf, age = input_age,race = input_race, disability = input_disability, placement = input_placement, gender = input_gender)
            #st.write(childid, ageid,raceid,disability,placement,gender)
            
            #store output into variable
            recommender_output = FMModel.get_recommendations(modelinfer = modelinfer, device = device, providers = providers, provider_biases = provider_biases, provider_embeddings = provider_embeddings, childid = childid, raceid = raceid, ageid = ageid, disability = disability, placement = placement, gender = gender, topN = 12)
            #st.write(recommender_output)
            ### FINISH RUNNING RECOMMENDER MODEL ###


            ### SET UP DURATION MODEL ###
            providers_lookup = DurationModel.load_provider_lookup_table()
            recommended_providers = recommender_output.merge(providers_lookup, how = 'left', left_on = 'PROVIDER_ID', right_on = 'PROVIDER_ID')
            recommended_providers_features = recommended_providers[DurationModel.FOSTER_FEATURES].reset_index(drop=True)
            child_input_features = pd.concat([child_input_record_df]*recommended_providers_features.shape[0], ignore_index = True)
            placements_to_predict = pd.concat([child_input_features, recommended_providers_features], axis =1)
            #st.write(placements_to_predict)
            ### FINISH SET UP OF DURATION MODEL ###


            ### RUN DURATION AND PROBABILITY MODELS ###
            duration_error_table = DurationModel.load_duration_error_table()
            duration_model = DurationModel.load_duration_model()
            probability_model = DurationModel.load_positive_probability_model()
            duration_prediction = DurationModel.get_duration(duration_model, duration_error_table, placements_to_predict)
            probability_prediction = DurationModel.get_probability_of_good_outcome(probability_model, placements_to_predict)
            final_providers = pd.concat([recommended_providers, duration_prediction, probability_prediction], axis = 1)
            # st.write(final_providers)
            ### FINISH RUNNING DURATION AND PROBABILITY MODELS ###


            ### FORMAT OUTPUT ###
            # st.write(recommended_providers)
            # st.write(duration_prediction)
            st.text('')
            st.text('')
            st.text('')
            st.text('')
            st.title('Risk Assesment and Providers Information')
            providers = st.beta_container()
            
            with providers:
                provcols =  st.columns(3)
                button_dict1 = {}
                button_dict2 = {}
                button_dict3 = {}
                
                for index, row in final_providers.iterrows():
                    def risk(n):
                        if n > 8.0:
                            return 'Low'
                        if n > 5.0:
                            return 'Medium'
                        if n < 4.0:
                            return 'Medium'
                        if n > 3.0:
                            return 'Medium'
                        if n >= 2.0:
                            return 'High'
                        if n < 2.0:
                            return 'High Risk'
                    mods = index%3
                    if mods == 0:
                        with provcols[0]:
                        
                            #html1 = "Unknown" if type(row["PROVIDER_NAME"])==float else '<em> <b>'+ row["PROVIDER_NAME"] + '</em> </b>'+ '    (Provider ID: ' + str(row["PROVIDER_ID"]) + ')'
                            #html = str(index + 1) + ". " + html1
                            #html1 = "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else str(round(row["PROVIDER_NUM_PREV_PLACEMENTS"]))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + str(int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + html1
                            #html = html + '<br>' + "Provider Strengths: " + '<b>' + row["FLAGS"] + '</b>'
                            #html1 = "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)
                            #html2 = html + '<br>' + "Track Record for reunification/adoption/guardianship: " + str(html1) + '%'
                            #html = html2 + '<br>' + "Match Rating: " + str(round(row.RATING,2)) + "/5"
                            #html = html + '<br>'+ "Estimated Stay Duration: " + '<b>' + str(int(round(row["Predicted Duration"],0))) + '</b>' + " days"
                            #html = html + '<br>'+ "Probability of Positive Outcome: " + '<b>' + str(round(row["Probability of Good Outcome"]*100,2)) + "%" + '</b>'
                            html = html + '<br>'+ "Risk: " + '<b>' + str(risk(float(row["Probability of Good Outcome"]*100)))  + '</b>' 
                            st.write(HTML_WRAPPER1.format(html), unsafe_allow_html=True)
                            #button_dict2["string{}".format(index)] = st.button("Risk Assesment with Providers", key = str(index))
                            #if button_dict2["string{}".format(index)]:
                                #DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            #st.markdown("---")
                            
                    elif  mods == 1:
                        with provcols[0]:
                             #html1 = "Unknown" if type(row["PROVIDER_NAME"])==float else '<em> <b>'+ row["PROVIDER_NAME"] + '</em> </b>'+ '    (Provider ID: ' + str(row["PROVIDER_ID"]) + ')'
                            #html = str(index + 1) + ". " + html1
                            #html1 = "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else str(round(row["PROVIDER_NUM_PREV_PLACEMENTS"]))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + str(int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + html1
                            #html = html + '<br>' + "Provider Strengths: " + '<b>' + row["FLAGS"] + '</b>'
                            #html1 = "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)
                            #html2 = html + '<br>' + "Track Record for reunification/adoption/guardianship: " + str(html1) + '%'
                            #html = html2 + '<br>' + "Match Rating: " + str(round(row.RATING,2)) + "/5"
                            #html = html + '<br>'+ "Estimated Stay Duration: " + '<b>' + str(int(round(row["Predicted Duration"],0))) + '</b>' + " days"
                            #html = html + '<br>'+ "Probability of Positive Outcome: " + '<b>' + str(round(row["Probability of Good Outcome"]*100,2)) + "%" + '</b>'
                            html = html + '<br>'+ "Risk: " + '<b>' + str(risk(float(row["Probability of Good Outcome"]*100)))  + '</b>' 
                            st.write(HTML_WRAPPER1.format(html), unsafe_allow_html=True)
                            #button_dict2["string{}".format(index)] = st.button("Risk Assesment with Providers", key = str(index))
                            #if button_dict2["string{}".format(index)]:
                                #DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            #st.markdown("---")
                            ##st.markdown("---")
                            
                    elif mods == 2: 
                        with provcols[0]:
                            #html1 = "Unknown" if type(row["PROVIDER_NAME"])==float else '<em> <b>'+ row["PROVIDER_NAME"] + '</em> </b>'+ '    (Provider ID: ' + str(row["PROVIDER_ID"]) + ')'
                            #html = str(index + 1) + ". " + html1
                            #html1 = "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else str(round(row["PROVIDER_NUM_PREV_PLACEMENTS"]))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + str(int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + html1
                            #html = html + '<br>' + "Provider Strengths: " + '<b>' + row["FLAGS"] + '</b>'
                            #html1 = "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)
                            #html2 = html + '<br>' + "Track Record for reunification/adoption/guardianship: " + str(html1) + '%'
                            #html = html2 + '<br>' + "Match Rating: " + str(round(row.RATING,2)) + "/5"
                            #html = html + '<br>'+ "Estimated Stay Duration: " + '<b>' + str(int(round(row["Predicted Duration"],0))) + '</b>' + " days"
                            #html = html + '<br>'+ "Probability of Positive Outcome: " + '<b>' + str(round(row["Probability of Good Outcome"]*100,2)) + "%" + '</b>'
                            html = html + '<br>'+ "Risk: " + '<b>' + str(risk(float(row["Probability of Good Outcome"]*100)))  + '</b>' 
                            st.write(HTML_WRAPPER1.format(html), unsafe_allow_html=True)
                            #button_dict2["string{}".format(index)] = st.button("Risk Assesment with Providers", key = str(index))
                            #if button_dict2["string{}".format(index)]:
                                #DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            #st.markdown("---")t_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            #st.markdown("---")
                        
#                button_dict = {}
#                for index, row in final_providers.iterrows():
#                    st.write(str(index + 1),". ", "Unknown" if type(row["PROVIDER_NAME"])==float else row["PROVIDER_NAME"], '    (Provider ID: ', row["PROVIDER_ID"], ") ------- ", row["FLAGS"])
#                    # st.write("Flags: ", row["FLAGS"])
#                    st.write("Number of Children Fostered: ", "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
#                    st.write("Provider Strengths: ", "No Red Flags" if row["FLAGS"] == 'No Flags' else row["FLAGS"])
#                    st.write("Track Record for reunification/adoption/guardianship: ", "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1), '%')
#                    st.write("Match Rating: ", round(row.RATING,2), "/5")
#                    st.write("Estimated Stay Duration: ", int(round(row["Predicted Duration"],0)), "days")
#                    st.write("Probability of Positive Outcome: ", round(row["Probability of Good Outcome"]*100,2), "%")
#                    #button_dict["string{}".format(index)] = st.button("Risk Assesment with Providers", key = str(index),on_click =  DurationModel.get_probability_distribution, args = (placements_to_predict.iloc[[index]], probability_model))
#                    button_dict["string{}".format(index)] = st.button("Risk Assesment with Providers", key = str(index))
#                    if button_dict["string{}".format(index)]:
#                        DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
#                    st.text('')
#                    st.text('')

    return None
    
def cs_home():
    st.write('Home Page')
	#components.html(vis_link_one)
    
	

    

def cs_architecture():
    st.write(BANNER,unsafe_allow_html=True) 
    st.write(BANNERsmall,unsafe_allow_html=True) 

    st.session_state['resetter'] = False
    st.title('E2E Pipilines and Models Specifications')
    st.text("")
    product1 = st.beta_container()
    product2, product3 =  st.columns(2)
    product4, product5 = st.columns(2)
    
def cs_performance():
    st.title('Model Performance')
    st.text("")
    st.text("")
    picture_jason = Image.open('learning_rate.PNG')
    picture_james = Image.open('cls_report.PNG')
    


    col1, col2, col3 = st.columns(3)

    col1.image(picture_jason, width = 300)
    col1.write('<div style="text-align: left"> <b> Learning Rate </b> </div>', unsafe_allow_html = True)
    col1.write('<div style="text-align: left"> Train and Test Loss </div>', unsafe_allow_html = True)
    col1.write('<div style="text-align: left"> <b> Risk Assesment </b> </div>', unsafe_allow_html = True)
    col1.text("")
    col1.text("")
    col1.text("")
    
    col2.image(picture_james, width = 300)
    col2.write('<div style="text-align: left"> <b> F1 Recall Precision </b> </div>', unsafe_allow_html = True)
    col2.write('<div style="text-align: left"> Reports </div>', unsafe_allow_html = True)
    col2.write('<div style="text-align: left"> <b> performance in metrics </b> </div>', unsafe_allow_html = True)
    col2.text("")
    col2.text("")
    col2.text("")

    



# def cs_model():
# 	st.write(BANNER,unsafe_allow_html=True) 
# 	st.write(BANNERsmall,unsafe_allow_html=True) 

# 	st.session_state['resetter'] = False
# 	st.title('Foster Care Abuse Risk')
# 	st.header('Features about Foster Care Abuse Risk')
# 	st.write('Process on creating this')
# 	model2 = XGBRegressor(objective ='reg:tweedie', tree_method = "gpu_hist", max_depth=12, n_estimators=200, predictor='cpu_predictor')
# 	model2.load_model("./XGBoost_regressor_2")
# 	placements_to_predict = pd.read_csv("./placements_to_predict.csv")
# 	st.write(placements_to_predict)
# 	new_df = model2.predict(placements_to_predict)
# 	st.write(new_df)



# Run main()

if __name__ == '__main__':
    main()
