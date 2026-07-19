
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title='Wellness Tourism Predictor', layout='centered')
st.title('🌴 Wellness Tourism Predictor')

HF_DATASET_REPO = 'vgoveas/tourism-dataset'
HF_MODEL_REPO = 'vgoveas/tourism-prediction-model'

@st.cache_resource
def load_artifacts():
    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename='best_model.joblib', repo_type='model')
        le_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename='label_encoders.joblib', repo_type='dataset')
        return joblib.load(model_path), joblib.load(le_path)
    except Exception as e:
        return None, None

model, label_encoders = load_artifacts()

if model is None or label_encoders is None:
    st.warning('⌛ Loading application components... Please refresh in a moment if this persists.')
    st.stop()

with st.sidebar:
    st.header('Customer Details')
    age = st.number_input('Age', 18, 90, 35)
    typeofcontact = st.selectbox('Contact Type', options=list(label_encoders['TypeofContact'].classes_))
    citytier = st.selectbox('City Tier', [1, 2, 3])
    durationofpitch = st.number_input('Pitch Duration (min)', 1, 60, 10)
    occupation = st.selectbox('Occupation', options=list(label_encoders['Occupation'].classes_))
    gender = st.selectbox('Gender', options=list(label_encoders['Gender'].classes_))
    numberofpersonvisiting = st.number_input('Persons Visiting', 1, 10, 2)
    numberoffollowups = st.number_input('Follow-ups', 0, 10, 3)
    productpitched = st.selectbox('Product Pitched', options=list(label_encoders['ProductPitched'].classes_))
    preferredpropertystar = st.selectbox('Property Star', [3, 4, 5])
    maritalstatus = st.selectbox('Marital Status', options=list(label_encoders['MaritalStatus'].classes_))
    numberoftrips = st.number_input('Annual Trips', 0, 50, 5)
    passport = st.selectbox('Passport', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    pitchsatisfactionscore = st.slider('Satisfaction Score', 1, 5, 3)
    owncar = st.selectbox('Own Car', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    numberofchildrenvisiting = st.number_input('Children Visiting', 0, 5, 0)
    designation = st.selectbox('Designation', options=list(label_encoders['Designation'].classes_))
    monthlyincome = st.number_input('Monthly Income', 0, 200000, 50000)

if st.button('Generate Prediction'):
    input_dict = {
        'Age': age, 'TypeofContact': label_encoders['TypeofContact'].transform([typeofcontact])[0],
        'CityTier': citytier, 'DurationOfPitch': durationofpitch,
        'Occupation': label_encoders['Occupation'].transform([occupation])[0],
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'NumberOfPersonVisiting': numberofpersonvisiting, 'NumberOfFollowups': numberoffollowups,
        'ProductPitched': label_encoders['ProductPitched'].transform([productpitched])[0],
        'PreferredPropertyStar': preferredpropertystar,
        'MaritalStatus': label_encoders['MaritalStatus'].transform([maritalstatus])[0],
        'NumberOfTrips': numberoftrips, 'Passport': passport, 'PitchSatisfactionScore': pitchsatisfactionscore,
        'OwnCar': owncar, 'NumberOfChildrenVisiting': numberofchildrenvisiting,
        'Designation': label_encoders['Designation'].transform([designation])[0],
        'MonthlyIncome': monthlyincome
    }
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.success(f'Target Identified: Likely to purchase (Prob: {proba:.2f})')
    else:
        st.info(f'Unlikely to purchase (Prob: {proba:.2f})')
