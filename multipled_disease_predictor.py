# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:10:45 2024

@author: apoorva
"""

#importing the dependencies
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

st.set_page_config(
        page_title="CureIT - The Healthcare Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Report a bug': "mailto:apoorva.cse1@gmail.com",
            'About': "Created by team LoopBreakers, CureIt is an advance healthcare support featuring machine learning to diagnose your statistical health reports, analyse them and predict the possibility of various diseases."
        }
    )

# CSS 
# custom_css = """
# <style>
# /* Body background with gradient */
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(to right, #020024, #090979);
# }

# /* Sidebar background with gradient, curved edges, and elevation */
# [data-testid="stSidebar"] {
#     background: linear-gradient(to bottom, #020024, #090979);
#     border-radius: 15px;
#     box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
#     margin-top:5px;
# }

# /* Padding to prevent content from touching the edges */
# [data-testid="stSidebar"] > div:first-child {
#     padding: 10px;
# }

# [data-testid="stContainer"] {
#     background: linear-gradient(to bottom, #020024, #090979);
#     border-radius: 15px;
#     box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
#     margin-top:5px;
# }
# </style>
# """

# st.markdown(custom_css, unsafe_allow_html=True)

import streamlit.components.v1 as components

custom_css = """
<style>
/* Body background with gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #020024, #090979);
}

/* Sidebar background with gradient, curved edges, and elevation */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #020024, #090979);
    border-radius: 15px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    margin-top:5px;
}

/* Padding to prevent content from touching the edges */
[data-testid="stSidebar"] > div:first-child {
    padding: 10px;
}

[data-testid="stContainer"] {
    background: linear-gradient(to bottom, #020024, #090979);
    border-radius: 15px;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    margin-top:5px;
}
</style>
"""

# Inject CSS into the app
components.html(f"""
    {custom_css}
""", height=0)


#loading the saved models
diabetes_model=pickle.load(open('models/diabetes_model.sav','rb'))
parkinsons_model=pickle.load(open('models/parkinson_model.sav','rb'))
cancer_model=pickle.load(open('models/cancer_model.sav','rb'))
heart_model=pickle.load(open('models/heart_model.sav','rb'))
medicine_predictor=pickle.load(open('models/svc.sav','rb'))

#loading saved scaler transformations
diabetesScaler = pickle.load(open('scaler_files/scalerDiabetes.pkl','rb'))
heartScaler = pickle.load(open('scaler_files/scalerHeart.pkl','rb'))
parkinsonScaler = pickle.load(open('scaler_files/scalerParkinson.pkl','rb'))
breastCancerScaler = pickle.load(open('scaler_files/scalerCancer.pkl','rb'))

#loading datasets for medicine predictor
sys_des = pd.read_csv("medicine_predictor/symtoms_df.csv")
precautions = pd.read_csv("medicine_predictor/precautions_df.csv")
workout = pd.read_csv("medicine_predictor/workout_df.csv")
description = pd.read_csv("medicine_predictor/description.csv")
medications = pd.read_csv("medicine_predictor/medications.csv")
diets = pd.read_csv("medicine_predictor/diets.csv")

#sidebar for navigation
with st.sidebar:
    selected=option_menu('CureIT',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinson\'s Prediction',
                          'Breast Cancer Prediction',
                          'Medicine Recommendation'
                         ],
                         menu_icon=['hospital-fill'],
                         icons=['activity','heart-pulse-fill','person-heart','virus','clipboard2-pulse'],
                         default_index=0,
                         orientation="horizontal"
                         )

# Predictor dataset table 
def breast_cancer():
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar")
        data = {
           'Feature': [
               'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
               'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
               'Radius Error', 'Texture Error', 'Perimeter Error', 'Area Error', 'Smoothness Error',
               'Compactness Error', 'Concavity Error', 'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error',
               'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
               'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
           ],
           'Description': [
               'Mean of distances from center to points on the perimeter',
               'Standard deviation of gray-scale values',
               'Mean of perimeter of cell nuclei',
               'Mean of area of cell nuclei',
               'Mean of local variation in radius lengths',
               'Mean of local variation in gray-scale values',
               'Mean of local variation in area of cell nuclei',
               'Mean of local variation in radius lengths of concave portions of contour',
               'Mean of local variation in radius lengths',
               'Mean of local variation in area of cell nuclei',
               'Standard error of the mean of distances from center to points on the perimeter',
               'Standard error of the mean of gray-scale values',
               'Standard error of the mean of perimeter of cell nuclei',
               'Standard error of the mean of area of cell nuclei',
               'Standard error of the mean of local variation in radius lengths',
               'Standard error of the mean of local variation in gray-scale values',
               'Standard error of the mean of local variation in area of cell nuclei',
               'Standard error of the mean of local variation in radius lengths of concave portions of contour',
               'Standard error of the mean of local variation in radius lengths',
               'Standard error of the mean of local variation in area of cell nuclei',
               'Largest mean value of the distance from center to points on the perimeter',
               'Largest mean value of gray-scale values',
               'Largest mean value of perimeter of cell nuclei',
               'Largest mean value of area of cell nuclei',
               'Largest mean value of local variation in radius lengths',
               'Largest mean value of local variation in gray-scale values',
               'Largest mean value of local variation in area of cell nuclei',
               'Largest mean value of local variation in radius lengths of concave portions of contour',
               'Largest mean value of local variation in radius lengths',
               'Largest mean value of local variation in area of cell nuclei'
           ]
       }
        df = pd.DataFrame(data)
        
        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
        }])
    )
        
def diabetes_predictor():
    with st.container():
        st.title("Diabetes Predictor")
        st.write("Please connect this app to your healthcare provider to predict your risk of developing diabetes based on various health factors. This app utilizes a machine learning model to assess the likelihood of diabetes onset. You can also input and adjust your health metrics such as BMI, glucose levels, and blood pressure using the sliders in the sidebar")
        data = {
        'Attribute': [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
            ],
        'Description': [
            'Number of times pregnant',
            'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
            'Diastolic blood pressure (mm Hg)',
            'Triceps skin fold thickness (mm)',
            '2-Hour serum insulin (mu U/ml)',
            'Body mass index (weight in kg/(height in m)^2)',
            'Diabetes pedigree function',
            'Age (years)',
            'Class variable (0 or 1) indicating whether the individual has diabetes or not'
            ]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
            }])
        )   

def heart_disease_predictor():
    with st.container():
        st.title("Heart Disease Predictor")
        st.write("Connect this app to your cardiovascular specialist to assess your risk of heart disease based on medical data. Using a machine learning model, this app predicts the probability of heart disease occurrence. Easily modify input parameters like cholesterol levels, blood pressure, and age using the sidebar sliders to understand your heart health better")
        data = {
        'Attribute': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ],
        'Description': [
            'Age (in years)',
            'Sex (0 = female, 1 = male)',
            'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
            'Resting blood pressure (in mm Hg)',
            'Serum cholesterol (in mg/dl)',
            'Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)',
            'Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)',
            'Maximum heart rate achieved',
            'Exercise induced angina (0 = no, 1 = yes)',
            'ST depression induced by exercise relative to rest',
            'Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
            'Number of major vessels (0-3) colored by flourosopy',
            'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)',
            'Presence of heart disease (0 = no, 1 = yes)'
        ]
    }

        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
        }])
    )

def parkinsons_predictor_model():
    with st.container():
        st.title("Parkinsons Detector")
        st.write("Link this app to your neurologist to predict the likelihood of Parkinson's disease based on clinical data. With a machine learning algorithm, this app estimates the probability of Parkinson's onset. Adjust the features such as tremors, rigidity, and bradykinesia using the sidebar sliders for a personalized assessment")
        data = {
        'Feature': [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR',
            'RPDE', 'D2',
            'DFA',
            'spread1', 'spread2', 'PPE'
        ],
        'Description': [
            'Average vocal fundamental frequency',
            'Maximum vocal fundamental frequency',
            'Minimum vocal fundamental frequency',
            'Several measures of variation in fundamental frequency',
            'Several measures of variation in fundamental frequency (absolute)',
            'Variation in fundamental frequency - Relative amplitude perturbation',
            'Variation in fundamental frequency - Period perturbation quotient',
            'Variation in fundamental frequency - Jitter:DDP',
            'Several measures of variation in amplitude',
            'Several measures of variation in amplitude (in dB)',
            'Amplitude perturbation quotient - Three-point average',
            'Amplitude perturbation quotient - Five-point average',
            'Variation in amplitude',
            'Amplitude perturbation quotient - Three-point average (in dB)',
            'Noise to tonal components ratio',
            'Harmonic to noise ratio',
            'Nonlinear dynamical complexity measure',
            'Nonlinear dynamical complexity measure',
            'Signal fractal scaling exponent',
            'Nonlinear measure of fundamental frequency variation',
            'Nonlinear measure of fundamental frequency variation',
            'Nonlinear measure of fundamental frequency variation'
        ]
    }

    # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Display the DataFrame as a table using Streamlit
        st.write(
        df
        .style
        .set_properties(**{'max-width': '100%', 'font-size': '1vw'})
        .set_table_styles([{
            'selector': 'tr:hover',
            'props': 'background-color: #ffff99;'
        }])
    )

def medicine_recommendation_system():
    with st.container():
        st.title("Medicine Recommendation System")
        st.write("Feed in your symptoms to get recommendation about tests,medicines and effective solutions for the same")
        
        
        
def preprocess_input_diabetes(input_data):
    # Transform the input features using the loaded scaler
    input_data_scaled = diabetesScaler.transform(input_data)
    return input_data_scaled

def predict_diabetes(input_data):
    input_data_scaled = preprocess_input_diabetes(input_data)
    # Make predictions using the loaded model
    predictions = diabetes_model.predict(input_data_scaled)
    return predictions

if (selected == 'Diabetes Prediction'):
    diabetes_predictor()
    # Display input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        glucose = st.text_input('Glucose Level')
    with col3:
        blood_pressure = st.text_input('Blood Pressure Value')
    with col1:
        skin_thickness = st.text_input('Skin Thickness')
    with col2:
        insulin = st.text_input('Insulin Level')
    with col3:
        bmi = st.text_input('BMI Level')
    with col1:
        diabetes_pedigree_function = st.text_input('Diabetes Pedigree Value')
    with col2:
        age = st.text_input('Age Value')

    # Check if all fields are filled
    fields_filled = all([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

    if not fields_filled:
        st.error("Please fill in all the fields.")
    else:
        # Convert input data to float and reshape as needed
        if st.button('Diabetes Disease Test results'):
            input_data = [[float(pregnancies), float(glucose), float(blood_pressure), float(skin_thickness), float(insulin), float(bmi), float(diabetes_pedigree_function), float(age)]]
            diabetes_prediction = predict_diabetes(input_data)
            if diabetes_prediction[0] == 1:
                diabetes_diagnosis = 'The person is Diabetic'
            else:
                diabetes_diagnosis = 'The person is Non-Diabetic'
        st.success(diabetes_diagnosis)

    
def preprocess_input_heart(input_data):
    # Transform the input features using the loaded scaler
    input_data_scaled = heartScaler.transform(input_data)
    return input_data_scaled

def predict_heart(input_data):
    input_data_scaled = preprocess_input_heart(input_data)
    # Make predictions using the loaded model
    predictions = heart_model.predict(input_data_scaled)
    return predictions

# Heart Prediction Page
if (selected == 'Heart Disease Prediction'):
    heart_disease_predictor()
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.text_input('Enter your Age')
    with col2:
        sex = st.text_input('Enter 0 if Female and 1 if Male')
    with col3:
        cp = st.text_input('CP Value')
    with col1:
        trest = st.text_input('Resting Blood Pressure Value')
    with col2:
        chol = st.text_input('Cholestrol Level')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar Value')
    with col1:
        restecg = st.text_input('Resting electrocardiographic results ')
    with col2:
        thalach = st.text_input('Maximum heart rate achieved Value')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('OldPeak Value')
    with col2:
        slope = st.text_input('Slope Value')
    with col3:
        ca = st.text_input('Number of major vessels (0-3) colored by flourosopy')
    with col1:
        thal = st.text_input('Thal Value')
    fields_filled = all([age,sex,cp,trest,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    if not fields_filled:
         st.error("Please fill in all the fields.")
    else:
        if st.button('Heart Disease Test Results'):
            input_data = [[float(age),float(sex),float(cp),float(trest),float(chol),float(fbs),float(restecg),float(thalach),float(exang),float(oldpeak),float(slope),float(ca),float(thal)]]
            heart_prediction = predict_heart(input_data)
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has high chance of heart disease'
            else:
                heart_diagnosis = 'The person has low chance of heart disease'
        st.success(heart_diagnosis)
        #creating a button for prediction
        

def preprocess_input_parkinson(input_data):
    # Transform the input features using the loaded scaler
    input_data_scaled = parkinsonScaler.transform(input_data)
    return input_data_scaled

def predict_parkinson(input_data):
    input_data_scaled = preprocess_input_parkinson(input_data)
    # Make predictions using the loaded model
    predictions = parkinsons_model.predict(input_data_scaled)
    return predictions

# Parkinson's Prediction Page
if (selected == 'Parkinson\'s Prediction'):
    parkinsons_predictor_model()
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Fo = st.text_input('Enter MDVP:Fo value')
    with col2:
        Fhi = st.text_input('Enter MDVP:Fhi value')
    with col3:
        Flo = st.text_input('Enter MDVP:Flo value')
    with col1:
        Jitter = st.text_input('Enter MDVP:Jitter value')
    with col2:
        Jitter2 = st.text_input('Enter Jitter(Abs) value')
    with col3:
        RAP = st.text_input('Enter MDVP:RAP value')
    with col1:
        PPQ = st.text_input('Enter MDVP:PPQ value')
    with col2:
        Jitter3 = st.text_input('Enter Jitter(DDP) value')
    with col3:
        Shimmer = st.text_input('Enter MDVP(Shimmer) Value')
    with col1:
        Shimmer2 = st.text_input('Enter the Shimmer(db) Value')
    with col2:
        APQ3 = st.text_input('Enter the Shimmer(APQ3) Value')
    with col3:
        APQ5 = st.text_input('Enter the Shimmer(APQ5) Value')
    with col1:
        APQ = st.text_input('Enter the MDVP(APQ) Value')
    with col2:
        DDA = st.text_input('Enter the Shimmer (DDA) Value')
    with col3:
        NHR = st.text_input('Enter the NHR Value')
    with col1:
        HNR = st.text_input('Enter the HNR Value')
    with col2:
        RPDE = st.text_input('Enter the RPDE Value')
    with col3:
        DFA = st.text_input('Enter the DFA Value')
    with col1:
        spread1 = st.text_input('Enter the Spread1 Value')
    with col2:
        spread2 = st.text_input('Enter the Spread2 Value')
    with col3:
        D2 = st.text_input('Enter the D2 Value')
    with col1:
        PPE = st.text_input('Enter the PPE Value')
    fields_filled = all([Fo,Fhi,Flo,Jitter,Jitter2,RAP,PPQ,Jitter3,Shimmer,Shimmer2,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])

    if not fields_filled:
         st.error("Please fill in all the fields.")
    else:
        parkinsons_disease_diagnosis = ''
        #creating a button for prediction
        if st.button('Parkinsons Disease Test Results'):
            input_data = [[float(Fo),float(Fhi),float(Flo),float(Jitter),float(Jitter2),float(RAP),float(PPQ),float(Jitter3),float(Shimmer),float(Shimmer2),float(APQ3),float(APQ5),float(APQ),float(DDA),float(NHR),float(HNR),float(RPDE),float(DFA),float(spread1),float(spread2),float(D2),float(PPE)]]
            parkinson_prediction = predict_parkinson(input_data)
            if parkinson_prediction[0] == 1:
                parkinson_diagnosis = 'The person has chances of Parkinsons Disease'
            else:
                parkinson_diagnosis = 'The person is Healthy'
        st.success(parkinson_diagnosis)

def preprocess_input_cancer(input_data):
    # Transform the input features using the loaded scaler
    input_data_scaled = breastCancerScaler.transform(input_data)
    return input_data_scaled

def predict_cancer(input_data):
    input_data_scaled = preprocess_input_cancer(input_data)
    # Make predictions using the loaded model
    predictions = cancer_model.predict(input_data_scaled)
    return predictions

# Breast Cancer Prediction Page
if (selected == 'Breast Cancer Prediction'):
    breast_cancer()
    col1,col2,col3 = st.columns(3)
    
    with col1:
        mean_radius = st.text_input('Mean Radius')
        mean_texture = st.text_input('Mean Texture')
        mean_perimeter = st.text_input('Mean Perimeter')
        mean_area = st.text_input('Mean Area')
        mean_smoothness = st.text_input('Mean Smoothness')
        mean_compactness = st.text_input('Mean Compactness')
        mean_concavity = st.text_input('Mean Concavity')
        mean_concave_points = st.text_input('Mean Concave Points')
        mean_symmetry = st.text_input('Mean Symmetry')
        mean_fractal_dimension = st.text_input('Mean Fractal Dimension')
    with col2:
        radius_error = st.text_input('Radius Error')
        texture_error = st.text_input('Texture Error')
        perimeter_error = st.text_input('Perimeter Error')
        area_error = st.text_input('Area Error')
        smoothness_error = st.text_input('Smoothness Error')
        compactness_error = st.text_input('Compactness Error')
        concavity_error = st.text_input('Concavity Error')
        concave_points_error = st.text_input('Concave Points Error')
        symmetry_error = st.text_input('Symmetry Error')
        fractal_dimension_error = st.text_input('Fractal Dimension Error')
    with col3:
        worst_radius = st.text_input('Worst Radius')
        worst_texture = st.text_input('Worst Texture')
        worst_perimeter = st.text_input('Worst Perimeter')
        worst_area = st.text_input('Worst Area')
        worst_smoothness = st.text_input('Worst Smoothness')
        worst_compactness = st.text_input('Worst Compactness')
        worst_concavity = st.text_input('Worst Concavity')
        worst_concave_points = st.text_input('Worst Concave Points')
        worst_symmetry = st.text_input('Worst Symmetry')
        worst_fractal_dimension = st.text_input('Worst Fractal Dimension')
        
    fields_filled = all([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
        radius_error, texture_error, perimeter_error, area_error, smoothness_error,
        compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
        worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension])

    if not fields_filled:
         st.error("Please fill in all the fields.")
    else:
        bc_disease_diagnosis = ''
        #creating a button for prediction
        if st.button('Breast Cancer Disease Test Results'):
            input_data = [[float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area), float(mean_smoothness),
                                                               float(mean_compactness), float(mean_concavity), float(mean_concave_points), float(mean_symmetry),float(mean_fractal_dimension),
                                                               float(radius_error), float(texture_error), float(perimeter_error), float(area_error), float(smoothness_error),
                                                               float(compactness_error), float(concavity_error), float(concave_points_error), float(symmetry_error), float(fractal_dimension_error),
                                                               float(worst_radius), float(worst_texture), float(worst_perimeter), float(worst_area), float(worst_smoothness),
                                                               float(worst_compactness), float(worst_concavity), float(worst_concave_points), float(worst_symmetry), float(worst_fractal_dimension)]]
            bc_disease_prediction = predict_cancer(input_data)
            if bc_disease_prediction[0]=='M':
                bc_disease_diagnosis = 'The tumor is Benign'
            else:
                bc_disease_diagnosis = 'The tumor is Malignant'
        st.success(bc_disease_diagnosis)
    
    
# medicine recommendation functions
# helper function
def helper(dis):
  desc = description[description['Disease']==dis]['Description']
  desc=" ".join([w for w in desc])
  pre = precautions[precautions['Disease']==dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
  pre = [col for col in pre.values]
  med = medications[medications['Disease']==dis]['Medication']
  med = [med for med in med.values]
  diet = diets[diets['Disease']==dis]['Diet']
  diet = [diet for diet in diet.values]
  wrkout = workout[workout['disease']==dis]['workout']
  return desc,pre,med,diet,wrkout
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
# model prediction function
def get_predicted_value(patient_symptoms):
  input_vector=np.zeros(len(symptoms_dict))
  for item in patient_symptoms:
    input_vector[symptoms_dict[item]] = 1
  return diseases_list[medicine_predictor.predict([input_vector])[0]]

# medicine recommendation system
if (selected == 'Medicine Recommendation'):
    medicine_recommendation_system()
    # test 1
    symptoms = st.text_input('Enter your symptoms')
    if st.button('Medication Prediction Results'):
        user_symptoms = [s.strip().lower() for s in symptoms.split(',') ]
        user_symptoms = [sym.strip("[]' ") for sym in user_symptoms ]
        predicted_disease = get_predicted_value(user_symptoms)
        desc,pre,med,diet,wrkout = helper(predicted_disease)

        # result print
        st.write("### Predicted Disease")
        st.write(predicted_disease)
    
        st.write("### Description")
        st.write(desc)
        
        st.write("### Precautions")
        precautions = [(i+1, p_i) for i, p_i in enumerate(pre[0])]
        precautions_df = pd.DataFrame(precautions, columns=["#", "Precaution"])
        st.table(precautions_df)
        
        st.write("### Medications")
        medications = [(i+1, m_i) for i, m_i in enumerate(med)]
        medications_df = pd.DataFrame(medications, columns=["#", "Medication"])
        st.table(medications_df)
        
        st.write("### Workout")
        workouts = [(i+1, w_i) for i, w_i in enumerate(wrkout)]
        workouts_df = pd.DataFrame(workouts, columns=["#", "Workout"])
        st.table(workouts_df)
        
        st.write("### Diets")
        diets = [(i+1, d_i) for i, d_i in enumerate(diet)]
        diets_df = pd.DataFrame(diets, columns=["#", "Diet"])
        st.table(diets_df)
        
    
    
