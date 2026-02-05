import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def predict_new_client(client_data):
    try:
        artifacts = joblib.load('churn_model_artifacts.pkl')
    except FileNotFoundError:
        return "ERROR: No Model File Found"

    model = artifacts['model']
    scaler = artifacts['scaler']
    fill_values = artifacts['fill_values']
    feature_names = artifacts['feature_names']

    input_df = pd.DataFrame([client_data])

    input_df = input_df.fillna(value=fill_values)

    input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result_text = "HIGH Probability of Customer Churn" if prediction == 1 else "LOW Probability of Customer Churn"
    
    print("=" * 30)
    print(f"Forecast Result: {result_text}")
    print(f"Probability of Churn: {probability:.2%}")
    print("=" * 30)

    plt.figure(figsize=(6, 2))
    plt.barh(['Churn Probability'], [probability], color='red' if prediction == 1 else 'green')
    plt.xlim(0, 1)
    plt.title(f"Client Forecast: {result_text}")
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    new_client = {
        'is_tv_subscriber': 1,
        'is_movie_package_subscriber': 0,
        'subscription_age': 2.5,
        'bill_avg': 25,
        'reamining_contract': 0.1,
        'service_failure_count': 2,
        'download_avg': 15.5,
        'upload_avg': 2.1,
        'download_over_limit': 0
    }

    predict_new_client(new_client)