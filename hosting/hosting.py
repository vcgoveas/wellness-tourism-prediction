
import os
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import HfHubHTTPError

HF_SPACE_REPO = "vgoveas/wellness-tourism-prediction"
DEPLOYMENT_DIR = "deployment"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("HF_TOKEN not set.")
    exit(1)

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

def push_to_hf_space():
    try:
        api.repo_info(repo_id=HF_SPACE_REPO, repo_type="space")
    except HfHubHTTPError:
        create_repo(repo_id=HF_SPACE_REPO, repo_type="space", private=False, space_sdk="docker")

    for root, _, files in os.walk(DEPLOYMENT_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            path_in_repo = os.path.relpath(local_path, DEPLOYMENT_DIR)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=HF_SPACE_REPO,
                repo_type="space",
            )
    print(f"Pushed to {HF_SPACE_REPO}")

if __name__ == "__main__":
    push_to_hf_space()
