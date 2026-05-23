from huggingface_hub import HfApi
import os

api = HfApi()

api.upload_file(
    path_or_fileobj="model/engine_maintenance_model.pkl",
    path_in_repo="engine_maintenance_model.pkl",
    repo_id="gsri24/engine-maintenance-model",
    token=os.getenv("HF_TOKEN")
)