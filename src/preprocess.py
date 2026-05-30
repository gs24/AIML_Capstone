

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Engine Condition"

FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]
def test_train_split(data_df):
    validate_columns(data_df)

    X = data_df[FEATURES]
    y = data_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # important for classification
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df

def split_features_target(df):
    X = df[FEATURES]
    y = df[TARGET]

    
    return X, y




def validate_columns(df):
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")



