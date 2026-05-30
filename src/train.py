

import os
import joblib
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from huggingface_hub import login
import os



from preprocess import split_features_target, test_train_split, validate_columns
from data_hf_process import  load_raw_data, upload_processed_data_to_huggingface,get_processed_data_from_huggingface

login(token=os.getenv("HF_TOKEN"))

# def load_data():
#     repo = os.getenv("HF_REPO_DATA")
#     print("DEBUG HF_REPO_DATA:", repo)
#     if repo is None:
#         raise ValueError("HF_REPO_DATA not set")

#     dataset = load_dataset(repo)

#     train_df = dataset["train"].to_pandas()
#     test_df = dataset["test"].to_pandas()

#     return train_df, test_df


def build_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def preprocess_split_and_upload(data_df):
    validate_columns(data_df)

    train_df, test_df = test_train_split(data_df)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    upload_processed_data_to_huggingface(repo_id=os.getenv("HF_REPO_DATA"))

def train_and_select(data_df):
    train_df,test_df = get_processed_data_from_huggingface()
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    rf = build_pipeline(RandomForestClassifier(random_state=42))
    xgb = build_pipeline(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_f1 = f1_score(y_test, rf.predict(X_test))
    xgb_f1 = f1_score(y_test, xgb.predict(X_test))

    print("RF F1:", rf_f1)
    print("XGB F1:", xgb_f1)
    if abs(rf_f1 - xgb_f1) < 0.01:
        print("Performance is almost similar: Selecting Random Forest for simplicity")
        best = rf
    else:
        best = rf if rf_f1 >= xgb_f1 else xgb
   

    if rf_f1 >= xgb_f1:
        print("Selected Model: Random Forest")
        best = rf
    else:
        print("Selected Model: XGBoost")
        best = xgb

    return best, X_train, y_train

#Although XGBoost achieved a slightly higher F1-score, 
# the difference was negligible. Therefore, Random Forest was selected as the final model due to its simplicity, faster inference, and ease of deployment.

def tune_model(model, X_train, y_train):
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10]
    }

    grid = GridSearchCV(model, param_grid, cv=3, scoring="f1")
    grid.fit(X_train, y_train)

    print("Best Params:", grid.best_params_)
    return grid.best_estimator_


def save_model(model):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("Model saved!")


if __name__ == "__main__":
    data_df = load_raw_data()
    preprocess_split_and_upload(data_df)
    model, X_train, y_train = train_and_select(data_df)
    best_model = tune_model(model, X_train, y_train)
    save_model(best_model)


