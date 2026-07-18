
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from huggingface_hub import hf_hub_download, HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
from sklearn.metrics import accuracy_score

# Configuration from environment variables or hardcoded
HF_USERNAME = os.environ.get('HF_USERNAME', 'vgoveas')
HF_DATASET_REPO = os.environ.get('HF_DATASET_REPO', 'vgoveas/tourism-dataset')
HF_MODEL_REPO = os.environ.get('HF_MODEL_REPO', 'vgoveas/tourism-prediction-model')
HF_TOKEN = os.environ.get('HF_TOKEN')

# --- Debugging HF_TOKEN ---
if HF_TOKEN:
    print(f"HF_TOKEN environment variable found. Length: {len(HF_TOKEN)} (masked: {HF_TOKEN[:5]}...{HF_TOKEN[-5:]})")
else:
    print("HF_TOKEN environment variable is NOT set.")

# Authenticate with Hugging Face by explicitly logging in
login(token=HF_TOKEN)
api = HfApi() # HfApi will now pick up the token from the authenticated session

# --- 1. Load Data and Label Encoders ---
print("Loading data and label encoders from Hugging Face...")
X_train_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="Xtrain.csv", repo_type="dataset")
X_test_path  = hf_hub_download(repo_id=HF_DATASET_REPO, filename="Xtest.csv", repo_type="dataset")
y_train_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="ytrain.csv", repo_type="dataset")
y_test_path  = hf_hub_download(repo_id=HF_DATASET_REPO, filename="ytest.csv", repo_type="dataset")
label_encoders_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="label_encoders.joblib", repo_type="dataset")

X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test  = pd.read_csv(y_test_path).values.ravel()
label_encoders = joblib.load(label_encoders_path)
print("Data and label encoders loaded.")

# --- 2. Define Preprocessor ---
numeric_features = X_train.columns.tolist()
preprocessor = make_column_transformer((StandardScaler(), numeric_features))

# --- 3. Define and Train Best Model (Random Forest with tuned hyperparameters) ---
print("Training the best Random Forest model...")
best_rf_model_params = {
    'randomforestclassifier__max_depth': None,
    'randomforestclassifier__max_features': 'sqrt',
    'randomforestclassifier__min_samples_leaf': 1,
    'randomforestclassifier__min_samples_split': 2,
    'randomforestclassifier__n_estimators': 300
}

rf_model = RandomForestClassifier(random_state=42)
# Apply the parameters directly to the model after creating the pipeline

# Create the full pipeline
pipeline = make_pipeline(preprocessor, rf_model)

# Set the best parameters found during GridSearchCV
pipeline.set_params(**best_rf_model_params)

pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate Model (Optional, can be logged to MLflow if tracking is enabled in CI/CD) ---
# In a real CI/CD, you might log metrics here to a remote MLflow server
# For simplicity, we'll just print a confirmation.
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy of the trained model: {accuracy:.4f}")

# --- 5. Save Model and Label Encoders for Deployment ---
MODEL_BUILDING_DIR = "model_building"
DEPLOYMENT_DIR = "deployment"
os.makedirs(MODEL_BUILDING_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

MODEL_PATH_LOCAL = os.path.join(DEPLOYMENT_DIR, "best_model.joblib") # Save to deployment dir for hosting.py
LABEL_ENCODERS_PATH_LOCAL = os.path.join(DEPLOYMENT_DIR, "label_encoders.joblib")

joblib.dump(pipeline, MODEL_PATH_LOCAL)
joblib.dump(label_encoders, LABEL_ENCODERS_PATH_LOCAL)
print(f"Trained model saved locally to {MODEL_PATH_LOCAL}")
print(f"Label encoders saved locally to {LABEL_ENCODERS_PATH_LOCAL}")

# --- 6. Upload Model to Hugging Face Model Hub --- 
print(f"Uploading model to Hugging Face Model Hub: {HF_MODEL_REPO}...")
try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Model repo '{HF_MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating model repo: {HF_MODEL_REPO}")
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=MODEL_PATH_LOCAL,
    path_in_repo="best_model.joblib",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)
print(f"Model uploaded to https://huggingface.co/{HF_MODEL_REPO}/tree/main")
