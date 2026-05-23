import os
import joblib
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from preprocess import split_features_target, validate_columns


def load_data():
    repo = os.getenv("HF_REPO_DATA")

    if repo is None:
        raise ValueError("HF_REPO_DATA not set")

    dataset = load_dataset(repo, data_dir="processed")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, test_df


def build_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def train_and_select(train_df, test_df):
    validate_columns(train_df)
    validate_columns(test_df)

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

    best = rf if rf_f1 >= xgb_f1 else xgb

    return best, X_train, y_train


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
    train_df, test_df = load_data()
    model, X_train, y_train = train_and_select(train_df, test_df)
    best_model = tune_model(model, X_train, y_train)
    save_model(best_model)