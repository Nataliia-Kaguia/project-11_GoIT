# 📡 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## Project Overview
This project is a Machine Learning solution designed to predict customer churn for a telecommunications company. By analyzing historical customer data (demographics, usage patterns, and subscription details), the model identifies users who are likely to cancel their service.

The goal is to provide actionable insights, allowing the business to proactively retain high-risk customers.

## Features & Pipeline

The project implements a complete End-to-End Machine Learning pipeline:

### 1. Data Preprocessing (`train_model.py`)
* **Missing Value Imputation:** Handling missing data in `download_avg`, `upload_avg`, and `remaining_contract` using **Median Imputation** to be robust against outliers.
* **Categorical Encoding:** Binary features (`is_tv_subscriber`, etc.) are processed directly.
* **Feature Scaling:** Applied **Standardization (`StandardScaler`)** to normalize continuous variables, ensuring consistent scale across features like bandwidth usage and bill amount.

### 2. Model Development
* **Algorithm:** `RandomForestClassifier` was chosen for its high accuracy and ability to handle non-linear relationships.
* **Hyperparameter Tuning:** Implemented `GridSearchCV` to optimize `n_estimators` and `max_depth`.
* **Cross-Validation:** Used 3-fold cross-validation to ensure model stability.

### 3. Integration & Inference (`predict.py`)
* **Model Persistence:** The trained model, scalar, and imputation values are serialized into a `churn_model.pkl` file.
* **Inference Engine:** A dedicated script simulates an interface for new client data, automatically applying the saved preprocessing steps to generate a real-time prediction.

---

## Project Structure

```text
├── internet_service_churn.csv   # Raw dataset
├── train_model.py               # Script for preprocessing and model training
├── predict.py                   # Script for inference (predicting new clients)
├── churn_model.pkl              # Saved model artifacts (generated after training)
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation