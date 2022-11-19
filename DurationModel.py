import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
import datetime as dt
import pickle

from xgboost import XGBClassifier, XGBRegressor, plot_tree
import xgboost as xgb

import numpy as np


FOSTER_FEATURES = [                            
'PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC'
,'PROVIDER_NUM_PREV_PLACEMENTS_NEUTRAL_PERC'
,'PROVIDER_NUM_PREV_PLACEMENTS_BAD_PERC'
,'PROVIDER_DAYS_SINCE_FIRST_PLACEMENT'
,'PROVIDER_NUM_PREV_PLACEMENTS'
,'PROVIDER_NUM_PREV_PLACEMENTS_GOOD'
,'PROVIDER_NUM_PREV_PLACEMENTS_NEUTRAL'
,'PROVIDER_NUM_PREV_PLACEMENTS_BAD'
,'PROVIDER_PREV_PLACEMENT_OUTCOME_1.0'
,'PROVIDER_PREV_PLACEMENT_OUTCOME_2.0'
,'PROVIDER_PREV_PLACEMENT_OUTCOME_3.0'
,'PLACEMENT_(NULL)'
,'PLACEMENT_FOSTER_FAMILY_HOME_(NON-RELATIVE)'
,'PLACEMENT_FOSTER_FAMILY_HOME_(RELATIVE)'
,'PLACEMENT_GROUP_HOME'
,'PLACEMENT_INSTITUTION'
,'PLACEMENT_PRE-ADOPTIVE_HOME'
,'PLACEMENT_RUNAWAY'
,'PLACEMENT_SUPERVISED_INDEPENDENT_LIVING'
,'PLACEMENT_TRIAL_HOME_VISIT'
,'RF1AMAKN'
,'RF1ASIAN'
,'RF1BLKAA'
,'RF1NHOPI'
,'RF1WHITE'
,'RF1UTOD'
,'HOFCCTK1'
,'FOSTER_AGE'
]




@st.cache
def load_provider_lookup_table():
	PATH = "."
	dataset = pd.read_csv(PATH + "/data/PROVIDER_FEATURES_LU.csv", low_memory = False)
	return dataset


@st.cache
def load_duration_error_table():
	PATH = "."
	dataset = pd.read_csv(PATH + "/data/ERROR_TABLE.csv", low_memory = False)
	return dataset


def load_duration_model():
	PATH = "."

	# model = pickle.load(open(PATH + "/XGBoost_regressor", "rb"))
	# return model

	# ## loading gpu model
	model = XGBRegressor(objective ='reg:tweedie', tree_method = "gpu_hist", max_depth=12, n_estimators=200, predictor='cpu_predictor')
	model.load_model(PATH + "/model/XGBoost_regressor_2")

	## loading dumb cpu model
	# model = XGBRegressor(objective ='reg:tweedie', max_depth=2, n_estimators=5)
	# model.load_model(PATH + "/XGBoost_regressor_cpu_dumb")

	## loading cpu model
	# model = XGBRegressor(objective ='reg:tweedie', max_depth=9, n_estimators=200)
	# model.load_model(PATH + "/XGBoost_regressor_cpu")

	return model


def load_positive_probability_model():
	PATH = "."

	# model = pickle.load(open(PATH + "/XGBoost_classifier", "rb"))
	# return model

	## loading gpu model
	model = XGBClassifier(objective="multi:softprob", tree_method = "gpu_hist", max_depth=10, n_estimators=200, predictor='cpu_predictor')
	model.load_model(PATH + "/model/XGBoost_classifier_2")

	## loading cpu model
	# model = XGBClassifier(objective="multi:softprob", max_depth=10, n_estimators=200)
	# model.load_model(PATH + "/XGBoost_classifier_cpu")

	return model


# @st.cache
def get_duration(model, error_table, records):
    predicted_duration = pd.DataFrame(model.predict(records)).rename(columns={0:'Predicted Duration'})
    # predicted_duration = pd.merge(predicted_duration, error_table, on='key')[list(predicted_duration.columns)+list(error_table.columns)]
    predicted_duration = predicted_duration.assign(foo=1).merge(error_table.assign(foo=1)).drop('foo', 1)
    predicted_duration = predicted_duration.apply(pd.to_numeric)
    predicted_duration = predicted_duration[(predicted_duration['Predicted Duration'] >= predicted_duration['Predictions_Bins_Lower']) &
                                            (predicted_duration['Predicted Duration'] < predicted_duration['Predictions_Bins_Upper'])][['Predicted Duration', 'Absolute Error']].reset_index(drop=True)
    return predicted_duration



def get_probability_of_good_outcome(model_clf, records):
    y_pred_probs = model_clf.predict_proba(records)
    y_pred_probs_df = pd.DataFrame(y_pred_probs)
    y_pred_probs_df.columns = model_clf.classes_

    GOOD_OUTCOMES = ['Reunification w/Parent(s) including Non-',
              'Adoption Finalization',
              'Permanent Guardianship (Includes Guardia',
              'Placement with a fit and willing Relativ',
              'Child ages out (18 - 23 Years Old)',
              'Adoption Placement',
              'Reunited with Removal Home Caregiver-Not',
              'APPLA (Another Planned Permanent Living',
              'Voluntary Opt Out (EFC only)',
              'Child Ages Out Non-EFC',
              'Young Adult Ages Out (EFC only)',
              'Permanent Guardianship to Successor Guar',
              'Child Ages Out (18 - 23 Years Old)',
              'Entered Military Service']

    GOOD_OUTCOMES_TEST_DF = []
    for i in y_pred_probs_df.columns:
        if i in GOOD_OUTCOMES:
            GOOD_OUTCOMES_TEST_DF.append(i)
    TEMP_DF = y_pred_probs_df[GOOD_OUTCOMES_TEST_DF]
    return pd.DataFrame(TEMP_DF.sum(axis=1)).reset_index(drop=True).rename(columns={0:'Probability of Good Outcome'})


def get_probability_distribution(record, model_clf):
    y_pred_probs = model_clf.predict_proba(record)
    y_pred_probs_df = pd.DataFrame(y_pred_probs)
    y_pred_probs_df.columns = model_clf.classes_
    print(y_pred_probs_df.columns)
    GOOD_OUTCOMES = ['Reunification w/Parent(s) including Non-',
              'Adoption Finalization',
              'Permanent Guardianship (Includes Guardia',
              'Placement with a fit and willing Relativ',
              'Child ages out (18 - 23 Years Old)',
              'Adoption Placement',
              'Reunited with Removal Home Caregiver-Not',
              'APPLA (Another Planned Permanent Living',
              'Voluntary Opt Out (EFC only)',
              'Child Ages Out Non-EFC',
              'Young Adult Ages Out (EFC only)',
              'Permanent Guardianship to Successor Guar',
              'Child Ages Out (18 - 23 Years Old)',
              'Entered Military Service']

    GOOD_OUTCOMES_TEST_DF = []
    for i in y_pred_probs_df.columns:
        if i in GOOD_OUTCOMES:
            GOOD_OUTCOMES_TEST_DF.append(i)
    TEMP_DF = y_pred_probs_df[GOOD_OUTCOMES_TEST_DF]
    y_pred_probs_df['Probability of a Good Outcome'] = TEMP_DF.sum(axis=1)

    add_list = ['Reunification'
            ,'Placement with a Relative'
            ,'Adoption'
            ,'Permanent Guardianship'
            ,'Home Visit'
            ,'Child Ages Out'
            ,'Other Permanent Arrangement'
            ,'Administrative Disruption'
            ,'Change Requested'
            ,'Runaway'
            # ,'Change in EFC Supervised IL Arrangement'
            # ,'Correctional Facility with Aftercare'
            # ,'Entering EFC Supervised IL Arrangement'
            # ,'Move Made in Accordance with Case Plan Goal'
            # ,'Transfer to Other Agency'
            ,'Hospitalization'
            ,'Incarceration/Detention'
            ,'Overcapacity'
            ,'Provider No Longer Licensed'
            ,'Other Placement Disruption'
            ,'Other'
            # ,'Voluntary Opt Out'
                ]
    drop_list = [
                ['Reunification w/Parent(s) including Non-','Reunited with Removal Home Caregiver-Not']
                ,['Placement with a fit and willing Relativ']
                ,['Adoption Finalization', 'Adoption Placement']
                ,['Permanent Guardianship (Includes Guardia','Permanent Guardianship to Successor Guar']
                ,['Trial Home Visit from Court-Ordered Plcm']
                ,['Child ages out (18 - 23 Years Old)','Child Ages Out Non-EFC','Young Adult Ages Out (EFC only)','Voluntary Opt Out (EFC only)','No Longer EFC Eligible', 'Emancipation', 'Entered Military Service', 'Marriage']
                ,['APPLA (Another Planned Permanent Living ']
                ,['Entering EFC Supervised IL Arrangement/P', 'Change in EFC Supervised IL Arrangement/','Move Made in Accordance with Case Plan G','Transfer to Other Agency (i.e. Out of Co','Dismissed by Court', 'Adoption Disruption']
                ,['Provider Requested Change','Child Requested Change','Parent/Relative/Guardian Requested Chang']
                ,['Runaway - Closing Case','Runaway - NOT Closing Case', 'Absconded', 'Abducted']
                # ,['Change in EFC Supervised IL Arrangement/']
                # ,['Child in Correctional Facility W/Afterca']
                # ,['Entering EFC Supervised IL Arrangement/P']
                # ,['Move Made in Accordance with Case Plan G']
                # ,['Transfer to Other Agency (i.e. Out of Co']
                ,['Hospitalization (more than 30 days)', 'Death of Child']
                ,['Placement Disruption']
                ,['Incarceration/Detention', 'Child in Correctional Facility W/Afterca']
                ,['Placement Overcapacity']
                ,['Provider No Longer Licensed']
                ,['Duplicate','Duplicate Provider Clean-up','Birthday Batch','Other']
                # ,['Voluntary Opt Out (EFC only)']
                ]


    for i in range(len(add_list)):
        y_pred_probs_df[add_list[i]] = y_pred_probs_df[drop_list[i]].sum(axis=1)
        y_pred_probs_df.drop(columns=drop_list[i],inplace=True)

    GOOD_OUTCOMES_V2 = ['Reunification',
                'Adoption',
                'Permanent Guardianship',
                'Placement with a Relative',
                'Child Ages Out',
                'Other Permanent Arrangement',
                'Home Visit'
                # 'Voluntary Opt Out',
                # 'Entered Military Service'
                ]
    NEUTRAL_OUTCOMES_V2 = ['Administrative Disruption']

    plt.rc('font', size=12)

    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    fig1, ax1 = plt.subplots(figsize=(22,10))

    cols = y_pred_probs_df.columns

    good_cols = []
    for i in range(len(cols)):
        if cols[i] in GOOD_OUTCOMES_V2:
            good_cols.append(cols[i])

    neutral_cols = []
    for i in range(len(cols)):
        if cols[i] in NEUTRAL_OUTCOMES_V2:
            neutral_cols.append(cols[i])

    not_good_cols = []
    for i in range(len(cols)):
        if cols[i] not in good_cols and cols[i] not in neutral_cols:
            not_good_cols.append(cols[i])

    df_good = y_pred_probs_df[good_cols].copy()
    df_neutral = y_pred_probs_df[neutral_cols].copy()
    df_not_good = y_pred_probs_df[not_good_cols].copy()
    df_not_good.drop(columns='Probability of a Good Outcome', inplace=True)

    ax1.barh(df_good.columns, df_good.iloc[0], color = '#1a9988ff')
    ax1.barh(df_neutral.columns, df_neutral.iloc[0], color = '#808080')
    ax1.barh(df_not_good.columns, df_not_good.iloc[0], color = '#eb5600')

    ax1.invert_yaxis()
    ax1.set_ylabel('Placement Outcomes', fontsize = 14)
    ax1.set_xlabel('Probability', fontsize = 14)
    ax1.set_title('Outcome Probability Breakdown', fontsize = 20)

    # ax1.barh(df_good.iloc[0], df_good.columns, color = '#1a9988ff')
    # ax1.barh(df_neutral.iloc[0], df_neutral.columns, color = '#595959')
    # ax1.barh(df_not_good.iloc[0], df_not_good.columns, color = '#eb5600')

    # ax1.set_xlabel('Placement Outcomes', fontsize=14)
    # ax1.set_yticklabels(list(df_good.columns)+list(df_not_good.columns), rotation=45, ha='right')
    # ax1.set_ylabel('Probability', fontsize=14)
    # ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))




        # ax.barh(y_pos, performance, xerr=error, align='center')
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(people)
        # ax.invert_yaxis()  # labels read top-to-bottom
        # ax.set_xlabel('Performance')
        # ax.set_title('How fast do you want to go today?')


    # plt.show()
    #return fig1
    st.pyplot(fig1)










