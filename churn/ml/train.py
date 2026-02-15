import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------- Paths -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_X_PATH = os.path.join(BASE_DIR, "training_data", "X_train.csv")
TRAIN_Y_PATH = os.path.join(BASE_DIR, "training_data", "Y_train.csv")

# ------------------- Load Data -------------------
X = pd.read_csv(TRAIN_X_PATH)
y = pd.read_csv(TRAIN_Y_PATH).values.ravel()  # 1D array

# ------------------- Split Data -------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------- Define Models -------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
}

# ------------------- Pipelines -------------------
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline(
        [("scaler", StandardScaler()), ("model", model)]  # масштабування числових фіч
    )

# ------------------- Train & Evaluate -------------------
best_model = None
best_model_name = None
best_f1 = 0
model_infos = {}

for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]  # для ROC-AUC

    # Метрики
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    f1 = report["1"]["f1-score"]
    roc_auc = roc_auc_score(y_val, y_prob)

    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")

    model_infos[name] = {
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
    }

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# ------------------- Save Best Model & Info -------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "churn_model.pkl"))
print(f"✅ Best model '{best_model_name}' saved as 'churn_model.pkl'!")

model_info = {"best_model": best_model_name, "models": model_infos}
joblib.dump(model_info, os.path.join(MODEL_DIR, "model_info.pkl"))
print("✅ Model info saved as 'model_info.pkl'!")
