import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "training_data", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "training_data", "feature_names.pkl")

# Завантажуємо модель, scaler та список ознак
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)


def predict_churn(client_data):
    """Прогноз відтоку для одного клієнта"""
    # Перетворюємо словник у DataFrame
    df = pd.DataFrame([client_data])

    # Перевірка на наявність всіх ознак
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Відсутні ознаки: {missing_features}")

    # Використовуємо порядок ознак, як у тренованій моделі
    df = df[feature_names]

    numerical_cols = [
        "subscription_age",
        "bill_avg",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
    ]

    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])

    # Прогноз і ймовірність
    predictions_proba = model.predict_proba(df_scaled)[:, 1]
    predictions = model.predict(df_scaled)

    # Рівень ризику
    risk_level = (
        "Високий ризик"
        if predictions_proba[0] > 0.7
        else "Середній ризик" if predictions_proba[0] > 0.3 else "Низький ризик"
    )

    return {
        "churn_probability": predictions_proba[0],
        "churn_prediction": predictions[0],
        "risk_level": risk_level,
    }


