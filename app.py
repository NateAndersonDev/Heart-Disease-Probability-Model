from cProfile import label
from doctest import DocFileTest
from turtle import color, title
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import altair as alt
from PIL import Image

#Loading model
hd_model = pickle.load(open("hd_model.pkl", "rb"))

#loading the encoder
#one_hot = pickle.load(open("oen_hot.pkl", "rb"))

#reading in dataset
hd_data = pd.read_csv("heart_2020_cleaned.csv")

#Yes_no_list = ['Have you smoked at least 100 cigarettes in your entire life?', 'More than 14 drinks per week (Men), 7 drinks per week (women)?',
#'Have you had a stroke?', 'Serious dificulty walking or climbing stairs?', 'Do you have Diabeties?', 'Any exercise in the past 30 days?', 'Do you have Asthma?', 'Do you have Kidney Disease?', 'Do you have Skin Cancer?' ]

#question_list =['Smoking', 'AlcoholDrinking', 'Stroke', 'WalkingDifficulty', 'Diabeties', 'Exercise', 'Asthma', 'KidneyDisease', 'SkinCancer']



#creating dataframe of dataset
df = pd.DataFrame(hd_data)

#Function that converts occurances of heartdisease to the probability of heartdisease normalized for the sample size given the feature
def convert_occurances_to_prob(main_df, feature):
    working_df = main_df[main_df['HeartDisease'] =='Yes'][feature]
    working_df= working_df.value_counts().reset_index(name='Occurances')
    total_df = main_df[feature].value_counts()
    total_df_index = total_df.reset_index(name = 'Total')
    
    for i in range(len(working_df)) :
    #print(df_smokers.loc[i, "index"])
        category_name = working_df.loc[i, "index"]
        value = working_df.loc[i, "Occurances"]
        total = total_df_index.loc[total_df_index['index'] == category_name]['Total'].values[0]
        probability = value/total
        working_df.loc[working_df['index']== category_name, 'Occurances'] = probability
    
    return working_df

#Function to create bar charts
def create_bar_chart(chart_df):

    chart= alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('index', axis=alt.Axis(title= '')),
        y=alt.Y('Occurances', axis=alt.Axis(title= 'Probability'))
    )
    return chart

#creating polynomial fit plot with regression line for BMI and heart disease

#setting up dataframe
BMI_prob_df = convert_occurances_to_prob(hd_data, 'BMI')


#defining degree of polynomial fits
degree_list = [1,2]

base = alt.Chart(BMI_prob_df).mark_circle(color="green").encode(
        alt.X("index", axis=alt.Axis(title= 'BMI')), alt.Y("Occurances", axis=alt.Axis(title= 'Probability'))
)

polynomial_fit = [
    base.transform_regression(
        "index", "Occurances", method="poly", order=order, as_=["index", str(order)]
    )
    .mark_line()
    .transform_fold([str(order)], as_=["degree", "Occurances"])
    .encode(alt.Color("degree:N"))
    for order in degree_list
]

poly_fit_BMI = alt.layer(base, *polynomial_fit)

#******************** setting up dummy variables**************************#
#converting Yes/ No to boolean 1,0
# yes = 1
# no = 0
conv_hd_data = hd_data.replace(to_replace = ['Yes','No'], value = ['1','0'])

# using dictonary comprehesion to create map for age categories and resultant labled numbers
AgeLabels = conv_hd_data.AgeCategory.astype('category').cat.categories.tolist()
agelabelmap = {'AgeCategory' : {k: v for k,v in zip(AgeLabels,list(range(1,len(AgeLabels)+1)))}}


# using dictonary comprehesion to create map for General health and resultant labled numbers
Healthlabelmap = {'GenHealth': {'Excellent': 1, 'Fair': 4, 'Good': 3, 'Poor': 5, 'Very good': 2}}

# replacing labels per values in the map
conv_hd_data.replace(agelabelmap, inplace=True)
conv_hd_data.replace(Healthlabelmap, inplace=True)

# create x features matrix
X = conv_hd_data.drop("HeartDisease", axis=1)

all_dummy = pd.get_dummies(X, columns=['Race', 'Diabetic', 'Sex'])

#Function for transforming user input data into a dataframe the model can read correctly 
def converting_df_to_model_input(dataframe):

    dataframe.replace({False :0, True: 1}, inplace = True)
    AgeLabels = hd_data['AgeCategory'].astype('category').cat.categories.tolist()
    agelabelmap = {'AgeCategory' : {k: v for k,v in zip(AgeLabels,list(range(1,len(AgeLabels)+1)))}}
    Healthlabelmap = {'GenHealth': {'Excellent': 1, 'Fair': 4, 'Good': 3, 'Poor': 5, 'Very good': 2}}

    dataframe.replace(agelabelmap, inplace=True)
    dataframe.replace(Healthlabelmap, inplace=True)

    convert_dict = {'Smoking' : object, 'AlcoholDrinking' : object, 'Stroke' : object, 'DiffWalking' :object, 'Diabetic' : object, 'PhysicalActivity' :object, 'Asthma' :object, 'KidneyDisease' :object, 'SkinCancer' : object }
    
    X = dataframe.drop("HeartDisease", axis=1)

    x_conv = X.astype(convert_dict)

    x_conv_dummy = pd.get_dummies(x_conv, columns=['Race', 'Diabetic', 'Sex'])

    missing_cols = set (all_dummy.columns) - set(x_conv_dummy.columns)
    for cat in missing_cols:
        x_conv_dummy[cat] =0

    x_conv_dummy= x_conv_dummy[all_dummy.columns]

    return x_conv_dummy

# #function to create scatter plot with heart disease
# def create_scatter_plot_hd(feature):
#     hasHd_df = hd_data[hd_data['HeartDisease'] =='Yes'][feature]
#     hasHd_df = hasHd_df.value_counts().reset_index(name='Occurances')

    
#     chart= alt.Chart(hasHd_df).mark_circle(color='red', opacity=.2).encode(
#         x='index',
#         y='Occurances',
#     )

#     return chart
# #function to create scatter plot with no heart disease
# def create_scatter_plot_noHd(feature):
#     noHd_df = hd_data[hd_data['HeartDisease'] =='No'][feature]
#     noHd_df = noHd_df.value_counts().reset_index(name='Occurances')

    
#     chart= alt.Chart(noHd_df).mark_circle().encode(
#         x='index',
#         y='Occurances',
        
#     )
#     return chart



st.write("""
# Welcome to the simple heart disease predictor.

**NOTE:** this is a sample application for my Capstone project, and not a valid predictor of heart disease

***
""")

####################### Form for data collection ##################################
#'Have you smoked at least 100 cigarettes in your entire life?', 'More than 14 drinks per week (Men), 7 drinks per week (women)?',
#'Have you had a stroke?', 'Serious dificulty walking or climbing stairs?', 'Do you have Diabeties?', 'Any exercise in the past 30 days?', 'Do you have Asthma?', 'Do you have Kidney Disease?', 'Do you have Skin Cancer?'
form = st.form(key='Submitform')
smoked= form.checkbox(label=f'Have you smoked at least 100 cigarettes in your entire life?')
drinks= form.checkbox(label=f'More than 14 drinks per week (Men), 7 drinks per week (women)?')
stroke= form.checkbox(label=f'Have you had a stroke?')
diff_walking= form.checkbox(label=f'Serious dificulty walking or climbing stairs')
diabeties= form.checkbox(label=f'Do you have Diabeties')
exercies=form.checkbox(label=f'Any exercise in the past 30 days?')
asthma=form.checkbox(label=f'Do you have Asthma?')
kidney_disease=form.checkbox(label=f'Do you have Kidney Disease?')
skin_cancer=form.checkbox(label=f'Do you have Skin Cancer?')
Bmi = form.slider('Body Mass Index : ', min_value = 0.0, max_value = 70.0, value = 25.0)
phys_health = form.slider('How many days in the last 30 days was your physical health " not good " : ', min_value = 0.0, max_value = 30.0, value = 5.0)
mental_health = form.slider('How many days in the last 30 days was your mental health " not good " : ', min_value = 0.0, max_value = 30.0, value = 5.0)
sleep_hours = form.slider('Average hours of sleep in a 24hr period',  min_value = 0.0, max_value = 24.0, value = 8.0)

age_category = form.selectbox("Age category", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
sex = form.selectbox("Sex", ['Male', 'Female'])
race = form.selectbox("Race", ['White', 'Hispanic', 'Black', 'Asain', 'American Indian/Alaskan Native', 'Other'])
gen_health = form.selectbox("General Health", ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
submit_button = form.form_submit_button(label='Submit')

if submit_button:
    # st.write('Input Data: ')
    # st.write(f'Body mass index: {Bmi}')
    # st.write(f'Physical Health: {phys_health}')
    # st.write(f'Mental Health: {mental_health}')
    # st.write(f'Average hours of sleep: {sleep_hours}')
    # st.write(f'Age category: {age_category}')
    # st.write(f'Sex: {sex}')
    # st.write(f'Race: {race}')
    # st.write(f'General Health: {gen_health}')
    # st.write(f'Skin Cancer {skin_cancer}')

    user_df = pd.DataFrame({'HeartDisease' : 'False', 
    'BMI':Bmi,
    'Smoking': smoked, 
    'AlcoholDrinking': drinks, 
    'Stroke': stroke,
    'PhysicalHealth': phys_health, 
    'MentalHealth': mental_health, 
    'DiffWalking': diff_walking, 
    'Sex': sex, 
    'AgeCategory': age_category, 
    'Race': race,
    'Diabetic': diabeties,
    'PhysicalActivity': exercies,
    'GenHealth': gen_health,
    'SleepTime': sleep_hours,
    'Asthma': asthma, 
    'KidneyDisease': kidney_disease,
    'SkinCancer': skin_cancer},
    index=[0])


    user_df_to_run = converting_df_to_model_input(user_df)
    
    results = hd_model.predict_proba(user_df_to_run)

    prob_hd = results[0][1]
    prob_no_hd = results[0][0]

    st.write(""" ## Your results: """)
    st.write(""" ### Probability of having heart disease: """ )
    st.write("{:.2%}".format(prob_hd))
    st.write(""" ### Probability of NOT having heart disease: """ )
    st.write("{:.2%}".format(prob_no_hd))
    st.write("""
    """)


#Creating dataframe that will hold the values from the user input when the submit button is pressed 
st.write("""
***
## Characteristics of the model

#### ROC curve

""")

st.image(Image.open('ROC curver.png'))

st.write("""

#### Confusion Matrix

Actual values as columns, predicted values as rows

""")

st.image(Image.open('ConfusionMatrix.png'))


st.write("""

#### Classification Report

""")

st.image(Image.open('ClassificationReport.PNG'))

st.write("""

## Risk factors identified in the data set 

- Smoking
- Sex
- Age
- BMI
    
***

""")

st.altair_chart(create_bar_chart(convert_occurances_to_prob(hd_data, 'Smoking')).properties(title='Probabilities of Heart Disease given the respondant smokes'), use_container_width=True)
st.altair_chart(create_bar_chart(convert_occurances_to_prob(hd_data, 'Sex')).properties(title='Probabilities Heart Disease given sex'), use_container_width=True)
st.altair_chart(create_bar_chart(convert_occurances_to_prob(hd_data, 'AgeCategory')).properties(title='Probabilities Heart Disease given age'), use_container_width=True)
st.altair_chart((poly_fit_BMI).properties(title='Probability of heart disease and BMI, polynomial regression'), use_container_width=True)

