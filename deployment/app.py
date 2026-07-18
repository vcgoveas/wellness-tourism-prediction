
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title='Wellness Tourism Predictor', layout='wide')
st.title('🌴 Wellness Tourism Package Purchase Prediction')

HF_DATASET_REPO = 'vgoveas/tourism-dataset'
HF_MODEL_REPO = 'vgoveas/tourism-prediction-model'

@st.cache_resource
def load_artifacts():
    try:
        # Download from HF Hub (uses HF_TOKEN from environment automatically)
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename='best_model.joblib', repo_type='model')
        le_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename='label_encoders.joblib', repo_type='dataset')
        return joblib.load(model_path), joblib.load(le_path)
    except Exception as e:
        st.error(f'Initialization Error: {e}')
        return None, None

model, label_encoders = load_artifacts()

if model and label_encoders:
    st.success('System Ready')
    
    with st.sidebar:
        st.header('Customer Information')
        age = st.number_input('Age', min_value=18, max_value=90, value=35)
        typeofcontact = st.selectbox('Type of Contact', options=list(label_encoders['TypeofContact'].classes_))
        citytier = st.selectbox('City Tier', options=[1, 2, 3])
        durationofpitch = st.number_input('Duration of Pitch (min)', min_value=1, max_value=60, value=10)
        occupation = st.selectbox('Occupation', options=list(label_encoders['Occupation'].classes_))
        gender = st.selectbox('Gender', options=list(label_encoders['Gender'].classes_))
        numberofpersonvisiting = st.number_input('Persons Visiting', min_value=1, max_value=10, value=2)
        numberoffollowups = st.number_input('Follow-ups', min_value=0, max_value=10, value=3)
        productpitched = st.selectbox('Product Pitched', options=list(label_encoders['ProductPitched'].classes_))
        preferredpropertystar = st.selectbox('Property Star', options=[3, 4, 5])
        maritalstatus = st.selectbox('Marital Status', options=list(label_encoders['MaritalStatus'].classes_))
        numberoftrips = st.number_input('Annual Trips', min_value=0, max_value=50, value=5)
        passport = st.selectbox('Passport', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        pitchsatisfactionscore = st.slider('Satisfaction Score', min_value=1, max_value=5, value=3)
        owncar = st.selectbox('Own Car', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        numberofchildrenvisiting = st.number_input('Children Visiting', min_value=0, max_value=5, value=0)
        designation = st.selectbox('Designation', options=list(label_encoders['Designation'].classes_))
        monthlyincome = st.number_input('Monthly Income', min_value=0, value=50000)

        predict_button = st.button('Predict Purchase')

    if predict_button:
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
        
        st.subheader('Prediction Results')
        if prediction == 1:
            st.success(f'Likely to purchase! (Probability: {proba:.2f})')
        else:
            st.warning(f'Unlikely to purchase. (Probability: {proba:.2f})')
else:
    st.info('Please wait... Application artifacts are being synchronized.')
