
import os
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model/model.pkl",
    path_in_repo=os.getenv("MODEL_FILENAME"),
    repo_id=os.getenv("HF_REPO_MODEL"),
    token=os.getenv("HF_TOKEN")
)

print("Model uploaded successfully!")
