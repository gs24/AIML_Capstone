import os

from datasets import load_dataset
from huggingface_hub import HfApi
from utils import create_hf_repo, get_env_variable, get_repo_id, login_to_huggingface


def load_raw_data():
    repo = os.getenv("HF_REPO_DATA")
    print("DEBUG HF_REPO_DATA:", repo)
    if repo is None:
        raise ValueError("HF_REPO_DATA not set")
    
    dataset = load_dataset(repo)

    df = dataset["train"].to_pandas()
    
    return df

def upload_raw_data_to_huggingface(repo_id, repo_type):

    api = HfApi()
    create_hf_repo(repo_id, repo_type="dataset")
    data_repo_id = get_repo_id(repo_id, repo_type="dataset")

    api.upload_folder(
        folder_path="data/raw",
        repo_id=data_repo_id,
        repo_type="dataset",
        path_in_repo="raw"
    )


def upload_processed_data_to_huggingface(repo_id, repo_type):

    api = HfApi()
    create_hf_repo(repo_id, repo_type="dataset")
    data_repo_id = get_repo_id(repo_id, repo_type="dataset")

    api.upload_folder(
        folder_path="data/processed",
        repo_id=data_repo_id,
        repo_type="dataset",
        path_in_repo="processed"
    )

def upload_to_huggingface_hub(file_path, repo_name):
    try:
        login_to_huggingface()
        create_hf_repo(repo_name, repo_type="dataset")

        #Load the dataset
        dataset = load_dataset("csv",
                               data_files=file_path)

        #Push the dataset to Hugging Face Hub
        dataset.push_to_hub(repo_name)

        print(f"Dataset uploaded to Hugging Face Hub: {repo_name}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")

def get_processed_data_from_huggingface():
    repo = os.getenv("HF_REPO_DATA")
    print("DEBUG HF_REPO_DATA:", repo)
    if repo is None:
        raise ValueError("HF_REPO_DATA not set")

    dataset = load_dataset(repo, data_dir="processed")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()


    return train_df, test_df