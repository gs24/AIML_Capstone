
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, login


def get_repo_id(repo_id, repo_type):
    if repo_type == "dataset":
        return f"{repo_id}-data"
    return repo_id

def login_to_huggingface():
    hf_token = get_env_variable("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    login(hf_token)


def get_env_variable(var_name):
    load_dotenv()
    return os.getenv(var_name)

def create_hf_repo(repo_id,repo_type):
    api = HfApi()

    data_repo_id = get_repo_id(repo_id, repo_type)

    # Create a new repository on Hugging Face Hub
    api.create_repo(
        repo_id=data_repo_id,
        repo_type=repo_type,
        private=False,
        exist_ok=True,
    )