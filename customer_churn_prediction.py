# import libraries
import os
import csv
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_FILE = 'Telco-Customer-Churn.csv'


def load_data(data_file):
    data = pd.read_csv(data_file)
    return data


def add_extra_futures(df):
    # Add average monthly spend
    # Add +1 to tenure to prevent division with 0
    df["avg_monthly_spend"] = pd.to_numeric(df["TotalCharges"], errors='coerce') / (df["tenure"] + 1)
    
    # Add month-to-month contract risk
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # Add number of services customer has
    services = [
        "PhoneService", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "MultipleLines"
    ]

    # Handle internet service values
    df["num_services"] = df[services].isin(["Yes", "DSL", "Fiber optic"]).sum(axis=1)
     

def preprocess_data(df):
    # Get input features and target values from raw data with added features
    X = df.drop(["customerID", "Churn"])
    y = df["Churn"]

    # Get numeric features and categorical features for transformation
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    # Initialize transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Initialize Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            # For numeric features use numeric transformer
            ("num", numeric_transformer, numeric_features)
            # For categorical features use categorical transformer
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Return features, targets and preprocessor
    return X, y, preprocessor

# def train_model()


# def evaluate_predictions()


def main():
    raw_data = load_data(DATA_FILE)
    add_extra_futures(raw_data)


if __name__ == '__main__':
    main()
