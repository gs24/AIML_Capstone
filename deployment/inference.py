import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os


def load_model():
    load_dotenv()
    model_path = hf_hub_download(repo_id=os.getenv('HF_REPO_MODEL'), 
                                 filename=os.getenv('MODEL_FILENAME', default="model.joblib"
                                                    ))
                                 
    model = joblib.load(model_path)
    return model


def predict(input_data:dict):
    model = load_model()
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    print(input_df)
    print(model.predict(input_df))
    print(model.predict_proba(input_df))
    return prediction[0]