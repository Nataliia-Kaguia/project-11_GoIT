import os
import pickle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_X_PATH = os.path.join(BASE_DIR, "training_data", "X_train.csv")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "training_data", "Y_train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load data ---
X = pd.read_csv(TRAIN_X_PATH)
y = pd.read_csv(TRAIN_Y_PATH).values.ravel()  # ensure 1D array

# --- Split data for validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale numerical features ---
numerical_cols = [
    "subscription_age",
    "bill_avg",
    "reamining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])

# --- Save scaler ---
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# --- Define models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
}

best_model = None
best_f1 = 0
model_infos = {}

# --- Train and evaluate models ---
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    report = classification_report(y_val, y_pred, output_dict=True)
    f1 = report["1"]["f1-score"]  # use F1-score for positive class (churn)
    print(classification_report(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred)

    model_infos[name] = {"f1_score": f1, "confusion_matrix": cm}

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# --- Save the best model with compression ---
joblib.dump(best_model, os.path.join(MODEL_DIR, "churn_model.pkl"), compress=3)

# --- Save model info ---
with open(os.path.join(MODEL_DIR, "model_info.pkl"), "wb") as f:
    pickle.dump({"best_model": best_model_name, "models": model_infos}, f)

print(f"\n✅ Best model: {best_model_name} with F1-score = {best_f1:.3f}")
