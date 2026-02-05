import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

def train_churn_model(input_file):
    df = pd.read_csv(input_file)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    fill_values = {
        'reamining_contract': df['reamining_contract'].median(),
        'download_avg': df['download_avg'].median(),
        'upload_avg': df['upload_avg'].median()
    }
    df = df.fillna(value=fill_values)

    X = df.drop('churn', axis=1)
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }

    print("Start Training The Model With Cross-Validation...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test_scaled)
    
    print("\n===== Assessment Results =====")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("\nReport:")
    print(classification_report(y_test, y_pred))

    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'fill_values': fill_values,
        'feature_names': X.columns.tolist()
    }
    joblib.dump(artifacts, 'churn_model_artifacts.pkl')
    print("\nModel and Parameters Saved in 'churn_model_artifacts.pkl'")

if __name__ == "__main__":
    train_churn_model('internet_service_churn.csv')