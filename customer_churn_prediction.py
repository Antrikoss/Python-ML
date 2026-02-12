import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc


DATA_FILE = 'Telco-Customer-Churn.csv'
TEST_SIZE = 0.20


def load_data(data_file):
    data = pd.read_csv(data_file)

    # Drop rows with missing values
    data["TotalCharges"] = data["TotalCharges"].replace(" ", np.nan)
    data.dropna(subset=["TotalCharges"], inplace=True)
    return data


def add_extra_futures(df):
    # Add average monthly spend
    # Add +1 to tenure to prevent division with 0
    df["avg_monthly_value"] = pd.to_numeric(
        df["TotalCharges"].str.strip(), errors="coerce") / (df["tenure"] + 1)

    # Add month-to-month contract risk
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # Add number of services customer has
    services = [
        "PhoneService", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "MultipleLines"
    ]

    # Handle internet service values
    df["num_services"] = df[services].isin(
        ["Yes", "DSL", "Fiber optic"]).sum(axis=1)


def preprocess_data(df):
    # Get input features and target values from raw data with added features
    X = df.drop(columns=["customerID", "Churn"])
    # Encode labels
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Get numeric features and categorical features for transformation
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(
        include=["object", "category", "str"]).columns

    # Initialize transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Initialize Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            # Scale numeric features
            ("num", numeric_transformer, numeric_features),
            # For categorical features use categorical transformer
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Return features, targets and preprocessor
    return X, y, preprocessor


def train_models(X_train, y_train, preprocessor):
    # Initialize models
    models = {
        "logistic_reg": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    trained_models = dict()

    # For both models
    for name, model in models.items():
        # Create pipeline
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Fit model
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

    return trained_models


def evaluate_predictions(X_test, y_test, model, model_name):
    # Make predictions
    predictions = model.predict(X_test)
    # Probability for positive class
    y_score = model.predict_proba(X_test)[:, 1]

    # Print Accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"\nAccuracy: {accuracy:.4f}%")

    # Print ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_score)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=["No", "Yes"], zero_division=0))

    # Calculate RUC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # Plot RUC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()

    # Save plot
    plot_name = None
    if model_name == "Logistic Regression":
        plot_name = "Logistic_Regression"
    else:
        plot_name = "Random_Forest"
    plt.savefig(f"{plot_name}.jpg")

def main():
    # Load raw data
    print("Loading data...")
    raw_data = load_data(DATA_FILE)

    print("\nPreprocessing data...")
    # Add extra features
    add_extra_futures(raw_data)

    # Preprocess data
    X, y, preprocessor = preprocess_data(raw_data)

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    print("\nTraining models...")
    # Train Logistic Regression and Random Forest models
    trained_models = train_models(X_train, y_train, preprocessor)

    print("\nEvaluating the models...")
    # Print evaluation for Logistic Regression model
    print("\nLogistic Regression Model:")
    evaluate_predictions(
        X_test, y_test, trained_models["logistic_reg"], "Logistic Regression")

    # Print evaluation for Random Forest model
    print("\nRandom Forest Model:")
    evaluate_predictions(
        X_test, y_test, trained_models["random_forest"], "Random Forest")
    
    print(f"\nRUC curve plots saved at: 'Logistic_Reg_RUC.jpg' & 'Random_Forest_RUC.jpg")


if __name__ == '__main__':
    main()
