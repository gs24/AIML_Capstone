from xml.parsers.expat import model

import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

import joblib
from dotenv import load_dotenv
from datasets import load_dataset
from xgboost import XGBClassifier
from huggingface_hub import HfApi


def load_data_from_hf():
    '''
        Loading the data from Hugging Face Hub
    '''

    load_dotenv()
    repo_name = os.getenv('REPO_NAME')

    datasets = load_dataset(repo_name, data_dir="processed")

    train_df = datasets['train'].to_pandas()
    test_df = datasets['test'].to_pandas()

    return train_df, test_df


def prepare_data_for_model_building(train_df, test_df):

    X_train = train_df.drop('Engine Condition', axis=1)
    y_train = train_df['Engine Condition']

    X_test = test_df.drop('Engine Condition', axis=1)
    y_test = test_df['Engine Condition']

    return X_train, X_test, y_train, y_test


def get_models():
    models = {
        'RandomForest': RandomForestClassifier(random_state=79),
        'XGBoost': XGBClassifier(random_state=79)
    }
    return models

def evaluate_baseline_models(models, X_train, y_train, X_test, y_test):
    baseline_metrics = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        baseline_metrics[model_name] = metrics

    return baseline_metrics


def select_best_model(baseline_metrics):
    '''
    Selecting the best model based on F1 Score, as it is a good balance between precision and recall, especially for imbalanced datasets.
    '''
    best_model_name = max(baseline_metrics, key=lambda x: baseline_metrics[x]['f1_score'])
    print(f"Best Model: {best_model_name} with F1 Score: {baseline_metrics[best_model_name]['f1_score']:.4f}")
    return best_model_name


def get_models_parameters_grid_for_hyperparameter_tuning(model):
    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif isinstance(model, XGBClassifier):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    else:
        raise ValueError("Unsupported model type for hyperparameter tuning.")
    return param_grid

def hyperparameter_tuning(model, X_train, y_train):
    param_grid = get_models_parameters_grid_for_hyperparameter_tuning(model)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Model Name: ", model.__class__.__name__)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics


def final_evaluation(model, X_test, y_test):
    print("Final Evaluation of the Best Model:")
    metrics = evaluate_model(model, X_test, y_test)

    print("Final Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    return metrics


def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model saved")


def upload_model_to_hf(model_path="models/model.pkl"):
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    repo_name = os.getenv("HF_REPO") + "-model"

    api = HfApi()

    # Create repo if not exists
    api.create_repo(
        repo_id=repo_name,
        repo_type="model",
        exist_ok=True,
        token=token
    )

    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_name,
        token=token
    )

    print(f"Model uploaded to Hugging Face: {repo_name}")

if __name__ == "__main__":
    train_df, test_df = load_data_from_hf()
    X_train, X_test, y_train, y_test = prepare_data_for_model_building(train_df, test_df)

    models = get_models()
    baseline_metrics = evaluate_baseline_models(models, X_train, y_train, X_test, y_test)

    best_model_name = select_best_model(baseline_metrics)
    best_model = models[best_model_name]

    tuned_model = hyperparameter_tuning(best_model, X_train, y_train)

    final_metrics = final_evaluation(tuned_model, X_test, y_test)

    save_model(tuned_model, best_model_name)
    upload_model_to_hf()