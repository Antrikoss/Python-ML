# Customer Churn Prediction

This project uses machine learning to predict customer churn based on the **Telco Customer Churn dataset**. It also compares the performance of two classification models: **Logistic Regression** and **Random Forest**.

Dataset:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Features

- Data cleaning and preprocessing using **Pandas**
- Model training using:
  - Logistic Regression
  - Random Forest
- Model evaluation using:
  - Accuracy
  - ROC-AUC score
  - Classification report
- Visualization:
  - ROC curve with AUC score

## Prerequisites

Make sure you have Python installed, along with the following libraries:

- NumPy  
- Pandas  
- scikit-learn  
- Matplotlib  

You can install all dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/Antrikoss/Python-ML.git
cd Python-ML
```

2. Run the script:

```bash
python customer_churn_prediction.py
```

## Output

The script will:

- Train both models
- Print evaluation metrics (accuracy, ROC-AUC, classification report)
- Display ROC curves for model comparison

## Project Structure

```
Python-ML/
│── customer_churn_prediction.py
|── Telco-Customer-Churn.csv
│── README.md
```
