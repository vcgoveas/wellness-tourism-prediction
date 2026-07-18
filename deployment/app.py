
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Configuration from environment variables or hardcoded for the deployed app
HF_DATASET_REPO = "vgoveas/tourism-dataset"
HF_MODEL_REPO = "vgoveas/tourism-prediction-model"

# Load the model and label encoders
@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib", repo_type="model")
    label_encoders_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="label_encoders.joblib", repo_type="dataset")

    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)
    return model, label_encoders

model, label_encoders = load_artifacts()

# --- Streamlit UI ---
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("🌴 Wellness Tourism Package Purchase Prediction")
st.markdown("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

with st.sidebar:
    st.header("Customer Information")

    # Input fields
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
