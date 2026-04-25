from datasets import load_dataset
from huggingface_hub import HfApi
from utils import create_hf_repo, get_env_variable, get_repo_id, login_to_huggingface

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


if __name__ == "__main__":
    repo_id = get_env_variable("HF_REPO")
    upload_raw_data_to_huggingface(repo_id=repo_id, repo_type="dataset")