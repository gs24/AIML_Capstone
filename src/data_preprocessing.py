from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from dotenv import load_dotenv

from huggingface_hub  import HfApi

def load_data():
    # Load environment variables
    load_dotenv()
    
    repo_name = os.getenv('REPO_NAME')

    if not repo_name:
        raise ValueError("REPO_NAME environment variable not set.")
    
    dataset = load_dataset(repo_name)

    df = dataset['train'].to_pandas()

    return df

def preprocess_data(df):

    #Removing duplicates
    df = df.drop_duplicates()

    #There are no missing values in the dataset, so we can skip that.

    #No outliers removal as the outliers are required for the model to predict the failure

    return df


def split_data(df):
    
    X = df.drop('Engine Condition', axis=1)
    y = df['Engine Condition']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=79)

    return X_train, X_test, y_train, y_test

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def save_data(X_train,X_test, y_train, y_test):
    make_dir("data/processed")
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Data saved successfully in data/processed directory.")

def upload_data_to_hf():

    load_dotenv()
    repo_name = os.getenv('REPO_NAME')
    if not repo_name:
        raise ValueError("REPO_NAME environment variable not set.")
    
    api = HfApi()

    # Upload the processed data to Hugging Face Hub
    api.upload_folder(
        folder_path="data/processed",
        repo_id=repo_name,
        repo_type="dataset",
        path_in_repo="processed"
    )

    print("Data uploaded successfully to Hugging Face Hub.")


if __name__ == "__main__":
    '''
    ## Data Preprocessing
        - No missing values were found in the dataset
        - Duplicate records were removed
        - Outliers were retained as they represent real-world engine failure scenarios
        - The dataset was split into training and testing sets using an 80:20 ratio
        - Stratified sampling was used to maintain class distribution
        - Processed datasets were saved and uploaded to Hugging Face for reproducibility
    '''
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(X_train,X_test, y_train, y_test)
    upload_data_to_hf()