
import os
from huggingface_hub import HfApi, login

HF_SPACE_REPO = 'vgoveas/wellness-tourism-prediction'
DEPLOYMENT_DIR = 'deployment'
HF_TOKEN = os.environ.get('HF_TOKEN')

if HF_TOKEN:
    login(token=HF_TOKEN)
    api = HfApi()
    api.upload_folder(folder_path=DEPLOYMENT_DIR, repo_id=HF_SPACE_REPO, repo_type='space')
