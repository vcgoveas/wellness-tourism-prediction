
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

# --- Configuration from environment variables ---
HF_USERNAME = os.environ.get('HF_USERNAME')
HF_DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
HF_MODEL_REPO = f"{HF_USERNAME}/tourism-prediction-model"
HF_TOKEN = os.environ.get('HF_TOKEN')

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
if not HF_USERNAME:
    raise ValueError("HF_USERNAME environment variable not set.")

api = HfApi(token=HF_TOKEN)

# --- Helper Functions (replicated from notebook) ---
# Placeholder for model_performance_classification if needed for logging metrics
# The original notebook version used display(), which won't work in a script.
# For pipeline, we might just focus on saving the best model and relevant metrics.

# --- Main Pipeline Script ---
def run_pipeline():
    print("Starting MLOps pipeline...")

    # 1. Load Data
    print("Downloading data from Hugging Face...")
    X_train_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="Xtrain.csv", repo_type="dataset")
    X_test_path  = hf_hub_download(repo_id=HF_DATASET_REPO, filename="Xtest.csv", repo_type="dataset")
    y_train_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="ytrain.csv", repo_type="dataset")
    y_test_path  = hf_hub_download(repo_id=HF_DATASET_REPO, filename="ytest.csv", repo_type="dataset")
    le_path      = hf_hub_download(repo_id=HF_DATASET_REPO, filename="label_encoders.joblib", repo_type="dataset")

    X_train = pd.read_csv(X_train_path)
    X_test   = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test  = pd.read_csv(y_test_path).values.ravel()
    label_encoders = joblib.load(le_path)
    print("Data loaded.")

    # 2. Define Preprocessor (from notebook)
    numeric_features = X_train.columns.tolist()
    preprocessor = make_column_transformer((StandardScaler(), numeric_features))

    # 3. Model Training (using the best identified model from notebook - Random Forest)
    print("Training the best model (Random Forest)...")
    best_rf_model = make_pipeline(preprocessor, RandomForestClassifier(random_state=42,
                                                                    n_estimators=300,
                                                                    max_depth=None,
                                                                    min_samples_split=2,
                                                                    min_samples_leaf=1,
                                                                    max_features='sqrt'))
    best_rf_model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Save and Register Model to Hugging Face Model Hub
    print("Saving and registering model to Hugging Face Model Hub...")
    model_filename = "best_model.joblib"
    local_model_path = os.path.join("model_building", model_filename)
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    joblib.dump(best_rf_model, local_model_path)

    try:
        api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        print(f"Model repo '{HF_MODEL_REPO}' already exists.")
    except HfHubHTTPError:
        create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)
        print(f"Created model repo: {HF_MODEL_REPO}")

    api.upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo=model_filename,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded to Hugging Face Model Hub: https://huggingface.co/{HF_MODEL_REPO}/tree/main")
    print("MLOps pipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline()
