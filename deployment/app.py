
import sys
print("DEBUG: app.py started execution. (via sys.stderr)", file=sys.stderr)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import HfHubHTTPError # Import for specific error handling

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("🌴 Wellness Tourism Package Purchase Prediction")
st.markdown("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Configuration from environment variables or hardcoded for the deployed app
HF_DATASET_REPO = "vgoveas/tourism-dataset"
HF_MODEL_REPO = "vgoveas/tourism-prediction-model"

st.write(f"[DEBUG] HF_DATASET_REPO: {HF_DATASET_REPO}")
st.write(f"[DEBUG] HF_MODEL_REPO: {HF_MODEL_REPO}")

# Load the model and label encoders
@st.cache_resource
def load_artifacts():
    st.write("[DEBUG] Starting to load artifacts...")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        st.error("[CRITICAL ERROR] HF_TOKEN environment variable not found. Please ensure it's set as a Space Secret.")
        st.stop() # Stop the app if token is missing
    else:
        st.write(f"[DEBUG] HF_TOKEN found. Length: {len(hf_token)} (masked: {hf_token[:5]}...{hf_token[-5:]})")
        try:
            login(token=hf_token)
            st.write("[DEBUG] Hugging Face login successful.")
        except Exception as e:
            st.error(f"[CRITICAL ERROR] Hugging Face login failed with provided HF_TOKEN: {e}")
            st.stop() # Stop if login fails

    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib", repo_type="model")
        st.write(f"[DEBUG] Model downloaded to: {model_path}")

        label_encoders_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="label_encoders.joblib", repo_type="dataset")
        st.write(f"[DEBUG] Label encoders downloaded to: {label_encoders_path}")

        model = joblib.load(model_path)
        st.write("[DEBUG] Model loaded successfully.")

        label_encoders = joblib.load(label_encoders_path)
        st.write("[DEBUG] Label encoders loaded successfully.")

        st.write("[DEBUG] Artifacts loaded successfully.")
        return model, label_encoders
    except HfHubHTTPError as http_e:
        st.error(f"[ERROR] Hugging Face Hub HTTP Error during artifact download: {http_e}. This often means permissions issues or wrong repo/file ID. Check your HF_TOKEN and repository access.")
        st.stop()
    except Exception as e:
        st.error(f"[ERROR] Failed to load artifacts: {e}")
        st.stop() # Stop the app if artifacts can't be loaded

model, label_encoders = load_artifacts()

st.write("[DEBUG] Application initialization complete.")

with st.sidebar:
    st.header("Customer Information")

    # Input fields
    # Use st.session_state to persist values if needed, but for initial debug, direct input is fine
    age = st.number_input("Age", min_value=18, max_value=90, value=35, key='age')
    typeofcontact_options = list(label_encoders['TypeofContact'].classes_)
    typeofcontact = st.selectbox("Type of Contact", options=typeofcontact_options, key='typeofcontact')
    citytier = st.selectbox("City Tier", options=[1, 2, 3], key='citytier')
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10, key='durationofpitch')
    occupation_options = list(label_encoders['Occupation'].classes_)
    occupation = st.selectbox("Occupation", options=occupation_options, key='occupation')
    gender_options = list(label_encoders['Gender'].classes_)
    gender = st.selectbox("Gender", options=gender_options, key='gender')
    numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, key='numberofpersonvisiting')
    numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, key='numberoffollowups')
    productpitched_options = list(label_encoders['ProductPitched'].classes_)
    productpitched = st.selectbox("Product Pitched", options=productpitched_options, key='productpitched')
    preferredpropertystar = st.selectbox("Preferred Property Star", options=[3, 4, 5], key='preferredpropertystar')
    maritalstatus_options = list(label_encoders['MaritalStatus'].classes_)
    maritalstatus = st.selectbox("Marital Status", options=maritalstatus_options, key='maritalstatus')
    numberoftrips = st.number_input("NumberOfTrips Annually", min_value=0, max_value=50, value=5, key='numberoftrips')
    passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='passport')
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, key='pitchsatisfactionscore')
    owncar = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='owncar')
    numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, key='numberofchildrenvisiting')
    designation_options = list(label_encoders['Designation'].classes_)
    designation = st.selectbox("Designation", options=designation_options, key='designation')
    monthlyincome = st.number_input("Monthly Income", min_value=0, value=50000, key='monthlyincome')

    # Encode categorical features
    encoded_typeofcontact = label_encoders['TypeofContact'].transform([typeofcontact])[0]
    encoded_occupation = label_encoders['Occupation'].transform([occupation])[0]
    encoded_gender = label_encoders['Gender'].transform([gender])[0]
    encoded_productpitched = label_encoders['ProductPitched'].transform([productpitched])[0]
    encoded_maritalstatus = label_encoders['MaritalStatus'].transform([maritalstatus])[0]
    encoded_designation = label_encoders['Designation'].transform([designation])[0]

    # Create a DataFrame for prediction
    input_df = pd.DataFrame([{
        'Age': age,
        'TypeofContact': encoded_typeofcontact,
        'CityTier': citytier,
        'DurationOfPitch': durationofpitch,
        'Occupation': encoded_occupation,
        'Gender': encoded_gender,
        'NumberOfPersonVisiting': numberofpersonvisiting,
        'NumberOfFollowups': numberoffollowups,
        'ProductPitched': encoded_productpitched,
        'PreferredPropertyStar': preferredpropertystar,
        'MaritalStatus': encoded_maritalstatus,
        'NumberOfTrips': numberoftrips,
        'Passport': passport,
        'PitchSatisfactionScore': pitchsatisfactionscore,
        'OwnCar': owncar,
        'NumberOfChildrenVisiting': numberofchildrenvisiting,
        'Designation': encoded_designation,
        'MonthlyIncome': monthlyincome,
    }])

    predict_button = st.button("Predict Purchase")

if predict_button:
    st.write("[DEBUG] Prediction button clicked. Making prediction...")
    try:
        prediction_proba = model.predict_proba(input_df)[0][1] # Probability of purchasing (class 1)
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Results")
        if prediction == 1:
            st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba:.2f})")
        else:
            st.warning(f"The customer is unlikely to purchase the package. (Probability: {prediction_proba:.2f})")

        st.write(f"Raw Prediction: {prediction}")
        st.write(f"Purchase Probability: {prediction_proba:.2f}")

        st.markdown("__Note__: A probability > 0.5 indicates a likely purchase.")
    except Exception as e:
        st.error(f"[ERROR] Prediction failed: {e}")
