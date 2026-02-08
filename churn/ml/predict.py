import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "training_data", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "training_data", "feature_names.pkl")

# Завантажуємо через joblib, а не pickle
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)


def predict_churn(client_data):
    if isinstance(client_data, dict):
        if isinstance(list(client_data.values())[0], list):
            df = pd.DataFrame(client_data)
        else:
            df = pd.DataFrame([client_data])
    else:
        df = client_data.copy()
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Відсутні ознаки: {missing_features}")
    df = df[feature_names]
    numerical_cols = ['subscription_age', 'bill_avg', 'reamining_contract', 
                      'service_failure_count', 'download_avg', 'upload_avg']
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
    predictions_proba = model.predict_proba(df_scaled)[:, 1]
    predictions = model.predict(df_scaled)
    result = pd.DataFrame({
        'churn_probability': predictions_proba,
        'churn_prediction': predictions,
        'risk_level': ['Високий ризик' if p > 0.7 else 'Середній ризик' if p > 0.3 else 'Низький ризик' 
                       for p in predictions_proba]
    })
    
    return result


def predict_from_csv(csv_path, output_path=None):
    df = pd.read_csv(csv_path)
    print(f"\nЗавантажено {len(df)} клієнтів з файлу: {csv_path}")
    predictions = predict_churn(df)
    result = pd.concat([df, predictions], axis=1)
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"✓ Результати збережені: {output_path}")
    return result


def pred_for_one_client():
    new_client = {
        'is_tv_subscriber': 1,              # Є підписка на ТБ
        'is_movie_package_subscriber': 0,   # Немає кінопакету
        'subscription_age': 3.5,            # Вік підписки 3.5 місяці
        'bill_avg': 45,                     # Середній рахунок $45
        'reamining_contract': 0,            # Немає контракту (місяць-до-місяця)
        'service_failure_count': 2,         # 2 збої сервісу
        'download_avg': 75.5,               # Середнє завантаження 75.5 GB
        'upload_avg': 8.2,                  # Середнє вивантаження 8.2 GB
        'download_over_limit': 1            # 1 раз перевищив ліміт
    }
    print("\nДані клієнта:")
    for key, value in new_client.items():
        print(f"  {key}: {value}")
    prediction = predict_churn(new_client)
    print("\n" + "-" * 70)
    print("РЕЗУЛЬТАТ ПРОГНОЗУ:")
    print("-" * 70)
    print(f"Ймовірність відтоку: {prediction['churn_probability'].values[0]:.2%}")
    print(f"Прогноз: {'ВІДТІК' if prediction['churn_prediction'].values[0] == 1 else 'ЗАЛИШИТЬСЯ'}")
    print(f"Рівень ризику: {prediction['risk_level'].values[0]}")

def pred_for_multiple_clients():
    multiple_clients = {
        'is_tv_subscriber': [1, 0, 1, 0],
        'is_movie_package_subscriber': [1, 0, 0, 1],
        'subscription_age': [8.5, 2.3, 11.2, 5.7],
        'bill_avg': [30, 15, 40, 25],
        'reamining_contract': [1.5, 0, 2.0, 0.5],
        'service_failure_count': [0, 3, 1, 2],
        'download_avg': [50, 10, 80, 35],
        'upload_avg': [5, 1, 9, 4],
        'download_over_limit': [0, 0, 2, 1]}
    print(f"\nКількість клієнтів: {len(multiple_clients['is_tv_subscriber'])}")
    predictions = predict_churn(multiple_clients)
    print("\n" + "-" * 70)
    print("РЕЗУЛЬТАТИ ПРОГНОЗІВ:")
    print("-" * 70)
    results_df = pd.DataFrame(multiple_clients)
    results_df = pd.concat([results_df, predictions], axis=1)
    print(results_df[['subscription_age', 'bill_avg', 'reamining_contract', 
                    'churn_probability', 'churn_prediction', 'risk_level']].to_string())
    print("\n" + "-" * 70)
    print("СТАТИСТИКА:")
    print("-" * 70)
    print(f"Загальна кількість клієнтів: {len(predictions)}")
    print(f"Прогнозується відтік: {predictions['churn_prediction'].sum()} клієнтів")
    print(f"Залишаться: {len(predictions) - predictions['churn_prediction'].sum()} клієнтів")
    print(f"Середня ймовірність відтоку: {predictions['churn_probability'].mean():.2%}")
    print("\nРозподіл за рівнями ризику:")
    risk_counts = predictions['risk_level'].value_counts()
    for risk_level, count in risk_counts.items():
        print(f"  {risk_level}: {count} клієнтів")

if __name__ == "__main__":
    pred_for_one_client()
    pred_for_multiple_clients()
