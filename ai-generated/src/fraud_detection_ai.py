#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Credit Card Fraud Detection

This script performs fraud detection on the Credit Card Fraud Detection dataset.
It includes data loading, exploratory data analysis, handling of imbalanced data,
model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, auc, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_clean_data(file_path):
    """
    Load and clean the credit card fraud dataset.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Loading and cleaning data...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values found: {missing_values}")
        # Handle missing values if needed
        df = df.dropna()
    else:
        print("No missing values found.")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Duplicate rows found: {duplicates}")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")
    
    print(f"Dataset shape: {df.shape}")
    return df

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
    """
    print("\nPerforming exploratory data analysis...")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Class distribution
    class_distribution = df['Class'].value_counts()
    print("\nClass distribution:")
    print(class_distribution)
    
    fraud_percentage = class_distribution[1] / len(df) * 100
    print(f"Percentage of fraudulent transactions: {fraud_percentage:.4f}%")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.savefig('class_distribution.png')
    
    # Feature correlation with the target
    correlations = df.corr()['Class'].sort_values(ascending=False)
    print("\nTop 10 features correlated with Class:")
    print(correlations[:11])  # Including Class itself
    
    # Visualize feature correlations
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Distribution of Amount
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Distribution of Transaction Amount')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Amount by Class')
    plt.tight_layout()
    plt.savefig('amount_distribution.png')
    
    # Distribution of Time
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Time'], bins=50, kde=True)
    plt.title('Distribution of Time')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Class', y='Time', data=df)
    plt.title('Time by Class')
    plt.tight_layout()
    plt.savefig('time_distribution.png')
    
    print("EDA completed. Visualizations saved.")

def prepare_data(df):
    """
    Prepare the data for modeling.
    
    Args:
        df (pd.DataFrame): The dataset
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\nPreparing data for modeling...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def handle_imbalanced_data(X_train, y_train, sampling_strategy=0.1):
    """
    Handle imbalanced data using SMOTE and undersampling.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        sampling_strategy (float): Ratio of minority to majority class after resampling
        
    Returns:
        tuple: Resampled X_train, y_train
    """
    print("\nHandling imbalanced data...")
    
    # Print class distribution before resampling
    print("Class distribution before resampling:")
    print(pd.Series(y_train).value_counts())
    
    # Create a pipeline with SMOTE and RandomUnderSampler
    over = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    
    # Apply the resampling
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    
    # Print class distribution after resampling
    print("Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts())
    
    return X_resampled, y_resampled

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    """
    Train and evaluate a model.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        model_name (str): Name of the model
        model: The model to train
        
    Returns:
        tuple: Trained model, predictions, and probabilities
    """
    print(f"\nTraining {model_name}...")
    
    # Start timer
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # End timer
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
    
    # Evaluate the model
    print(f"\n{model_name} Evaluation:")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'pr_curve_{model_name.lower().replace(" ", "_")}.png')
    
    return model, y_pred, y_prob

def main():
    """
    Main function to run the fraud detection pipeline.
    """
    print("Credit Card Fraud Detection")
    print("=" * 30)
    
    # Load and clean data
    df = load_and_clean_data('creditcard.csv')
    
    # Perform EDA
    perform_eda(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Handle imbalanced data
    X_resampled, y_resampled = handle_imbalanced_data(X_train, y_train)
    
    # Train and evaluate Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model, lr_pred, lr_prob = train_and_evaluate_model(
        X_resampled, X_test, y_resampled, y_test, 
        "Logistic Regression", lr_model
    )
    
    # Train and evaluate XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        scale_pos_weight=10  # To handle class imbalance
    )
    xgb_model, xgb_pred, xgb_prob = train_and_evaluate_model(
        X_resampled, X_test, y_resampled, y_test, 
        "XGBoost", xgb_model
    )
    
    # Compare models
    print("\nModel Comparison:")
    models = ["Logistic Regression", "XGBoost"]
    f1_scores = [
        f1_score(y_test, lr_pred),
        f1_score(y_test, xgb_pred)
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, f1_scores)
    plt.title('Model Comparison - F1 Score')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.savefig('model_comparison.png')
    
    print("Fraud detection analysis completed.")

if __name__ == "__main__":
    main() 