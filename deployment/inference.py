
import os

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

from preprocess import FEATURES, validate_columns


def load_model():
    path = hf_hub_download(
                repo_id='gsri24/engine-maintenance-model',
                filename='engine_maintenance_model.pkl'
            )
    return joblib.load(path)


def predict(input_data: dict):
    model = load_model()

    df = pd.DataFrame([input_data])

    validate_columns(df)

    df = df[FEATURES]  # enforce order

    pred = model.predict(df)

    return pred[0]
    
