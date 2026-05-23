import os

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

from src.preprocess import FEATURES, validate_columns


def load_model():
    path = hf_hub_download(
                repo_id=os.getenv("HF_REPO_MODEL"),
                filename=os.getenv("MODEL_FILENAME")
            )
    return joblib.load(path)


def predict(input_data: dict):
    model = load_model()

    df = pd.DataFrame([input_data])

    validate_columns(df)

    df = df[FEATURES]  # enforce order

    pred = model.predict(df)

    return pred[0]