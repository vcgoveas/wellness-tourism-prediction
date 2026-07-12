
import os
import shutil
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import HfHubHTTPError

# Configuration
HF_USERNAME = "vgoveas"
HF_SPACE_REPO = "vgoveas/wellness-tourism-prediction"
DEPLOYMENT_DIR = "deployment"

# Hugging Face login (token should be set in env or passed)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN environment variable not set. Please set it.")
    exit(1)

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

def push_to_hf_space():
    print(f"
--- Pushing files to Hugging Face Space: {HF_SPACE_REPO} ---")

    # 1. Create the Hugging Face Space repo if it doesn't exist
    try:
        api.repo_info(repo_id=HF_SPACE_REPO, repo_type="space")
        print(f"Space repo '{HF_SPACE_REPO}' already exists.")
    except HfHubHTTPError:
        print(f"Creating Space repo: {HF_SPACE_REPO}")
        create_repo(repo_id=HF_SPACE_REPO, repo_type="space", private=False, space_sdk="docker")
        print("Space created. Please allow a few moments for the Space to initialize.")

    # 2. Upload all files from the deployment directory
    for root, _, files in os.walk(DEPLOYMENT_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            # Calculate path in repo relative to DEPLOYMENT_DIR
            path_in_repo = os.path.relpath(local_path, DEPLOYMENT_DIR)
            print(f"Uploading {local_path} as {path_in_repo}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=HF_SPACE_REPO,
                repo_type="space",
            )
    print(f"
All deployment files pushed to {HF_SPACE_REPO}")
    print(f"Check your Space at: https://huggingface.co/spaces/{HF_SPACE_REPO}")

if __name__ == "__main__":
    push_to_hf_space()
