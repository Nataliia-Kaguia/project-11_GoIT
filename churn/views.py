from django.shortcuts import render
import pickle
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .forms import ChurnPredictionForm
from churn.ml.predict import predict_churn

# --- Paths ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.pkl")
X_TEST_PATH = os.path.join(BASE_DIR, "training_data", "X_test.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "training_data", "Y_test.csv")
SCALER_PATH = os.path.join(BASE_DIR, "training_data", "scaler.pkl")

# --- Load model & scaler & model_info ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_info = joblib.load(MODEL_INFO_PATH)  # dict with best_model and metrics

# --- Load test data ---
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()  # 1D array

# --------------------- Views ---------------------


def home_view(request):
    return render(request, "churn/home.html")


def feature_names_view(request):
    file_path = "churn/ml/training_data/feature_names.pkl"

    with open(file_path, "rb") as f:
        feature_names = pickle.load(f)

    context = {"feature_names": feature_names}
    return render(request, "churn/feature_names.html", context)


def model_metrics_view(request):
    # --- Predictions ---
    y_pred = model.predict(X_test)

    # --- Confusion matrix & metrics ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # --- Plot confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({model_info['best_model']})")

    # --- Convert plot to base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # --- Send context to template ---
    context = {
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2%}",
        "recall": f"{recall:.2%}",
        "f1_score": f"{f1:.2%}",
        "cm_base64": cm_base64,
        "best_model": model_info["best_model"],
    }

    return render(request, "churn/model_metrics.html", context)


def predict_view(request):
    form = ChurnPredictionForm()
    message = None
    probability = None

    if request.method == "POST":
        form = ChurnPredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # --- Get prediction ---
            result_df = predict_churn(data)
            probability = result_df["churn_probability"].values[0]
            prediction = result_df["churn_prediction"].values[0]
            risk_level = result_df["risk_level"].values[0]

            # --- Form message ---
            if prediction == 1:
                message = f"⚠️ Client WILL churn (Risk: {risk_level})"
            else:
                message = f"✅ Client will NOT churn (Risk: {risk_level})"

    return render(
        request,
        "churn/predict.html",
        {
            "form": form,
            "message": message,
            "probability": probability,
        },
    )
