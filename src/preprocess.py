import pandas as pd

TARGET = "Engine Condition"

FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]


def split_features_target(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def validate_columns(df):
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")