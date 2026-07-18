
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download, login

# Configure page for HF health checks
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("🌴 Wellness Tourism Package Purchase Prediction")

HF_DATASET_REPO = "vgoveas/tourism-dataset"
HF_MODEL_REPO = "vgoveas/tourism-prediction-model"

@st.cache_resource
def load_artifacts():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN environment variable not found.")
        st.stop()
    login(token=hf_token)
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib", repo_type="model")
    le_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="label_encoders.joblib", repo_type="dataset")
    return joblib.load(model_path), joblib.load(le_path)

try:
    model, label_encoders = load_artifacts()
    st.success("Artifacts loaded successfully!")
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

with st.sidebar:
    st.header("Customer Information")
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    typeofcontact = st.selectbox("Type of Contact", options=list(label_encoders['TypeofContact'].classes_))
    citytier = st.selectbox("City Tier", options=[1, 2, 3])
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)
    occupation = st.selectbox("Occupation", options=list(label_encoders['Occupation'].classes_))
    gender = st.selectbox("Gender", options=list(label_encoders['Gender'].classes_))
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    productpitched = st.selectbox("Product Pitched", options=list(label_encoders['ProductPitched'].classes_))
    preferredpropertystar = st.selectbox("Preferred Property Star", options=[3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", options=list(label_encoders['MaritalStatus'].classes_))
    numberoftrips = st.number_input("NumberOfTrips Annually", min_value=0, max_value=50, value=5)
    passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    owncar = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", options=list(label_encoders['Designation'].classes_))
    monthlyincome = st.number_input("Monthly Income", min_value=0, value=50000)

    # Encode categorical features
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
    predict_button = st.button("Predict Purchase")

if predict_button:
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Results")
    if prediction == 1:
        st.success(f"Likely to purchase! Probability: {proba:.2f}")
    else:
        st.warning(f"Unlikely to purchase. Probability: {proba:.2f}")
