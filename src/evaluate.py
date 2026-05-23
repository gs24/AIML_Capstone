import os
import joblib
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import split_features_target


def load_test():
    repo = os.getenv("HF_REPO_DATA")

    if repo is None:
        raise ValueError("HF_REPO_DATA not set")

    dataset = load_dataset(repo, data_dir="processed")
    test_df = dataset["test"].to_pandas()

    return split_features_target(test_df)


def load_model():
    return joblib.load("model/model.pkl")


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\nEvaluation:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))


if __name__ == "__main__":
    X_test, y_test = load_test()
    model = load_model()
    evaluate(model, X_test, y_test)
