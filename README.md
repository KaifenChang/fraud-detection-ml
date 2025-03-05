# Credit Card Fraud Detection

Adapting the top-down learning method to credit card fraud detection.
The file `fraud_detection.py` is the code of the top-down learning method, which is generated by Claude Sonnet 3.7 Agent.
The file `note.md` is the note of the top-down learning method.

***
**The following is the original README.md file of the project.**

This project implements a machine learning pipeline for credit card fraud detection using the Credit Card Fraud Detection dataset.

## Overview

The script performs the following tasks:
- Loads and cleans the dataset
- Performs exploratory data analysis (EDA)
- Handles imbalanced data using SMOTE and undersampling
- Trains baseline models (Logistic Regression and XGBoost)
- Evaluates models using Precision-Recall curves and F1-scores

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Dataset

The script uses the Credit Card Fraud Detection dataset (`creditcard.csv`), which should be placed in the same directory as the script. The dataset contains transactions made by credit cards, where:
- Class 0: Non-fraudulent transactions
- Class 1: Fraudulent transactions

## Usage

To run the fraud detection script:

```bash
python fraud_detection.py
```

## Output

The script generates several visualization files:
- `class_distribution.png`: Distribution of fraudulent vs. non-fraudulent transactions
- `correlation_matrix.png`: Correlation matrix of features
- `amount_distribution.png`: Distribution of transaction amounts
- `time_distribution.png`: Distribution of transaction times
- `confusion_matrix_logistic_regression.png`: Confusion matrix for Logistic Regression
- `confusion_matrix_xgboost.png`: Confusion matrix for XGBoost
- `pr_curve_logistic_regression.png`: Precision-Recall curve for Logistic Regression
- `pr_curve_xgboost.png`: Precision-Recall curve for XGBoost
- `model_comparison.png`: Comparison of F1 scores between models

## Model Performance

The script evaluates models using:
- Precision, Recall, and F1-score
- Precision-Recall curves
- Confusion matrices

These metrics are particularly useful for imbalanced classification problems like fraud detection. 
***
# fraud-detection-ml
