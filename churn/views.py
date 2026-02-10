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
    roc_auc_score,
    roc_curve,
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
    # file_path = "churn/ml/training_data/feature_names.pkl"

    # with open(file_path, "rb") as f:
    #     feature_names = pickle.load(f)

    # context = {"feature_names": feature_names}

    feature_names = {
        "is_tv_subscriber": "Є підписка на ТБ (Так/Ні)",
        "is_movie_package_subscriber": "Є підписка на пакет фільмів (Так/Ні)",
        "subscription_age": "Вік підписки користувача у місяцях",
        "bill_avg": "Середній рахунок користувача за період",
        "reamining_contract": "Залишок місяців до закінчення контракту",
        "service_failure_count": "Кількість збоїв у наданні послуг",
        "download_avg": "Средній обсяг завантажених даних",
        "upload_avg": "Середній обсяг відвантажених даних",
        "download_over_limit": "Кількість разів, коли користувач перевищив ліміт завантаження",
    }
    return render(request, "churn/feature_names.html", {"feature_names": feature_names})
    # return render(request, "churn/feature_names.html", context)


def model_metrics_view(request):
    # --- Predictions ---
    y_pred = model.predict(X_test)

    # --- Probabilities for ROC ---
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # для моделей без predict_proba (наприклад, SVM без probability=True)
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # --- ROC Curve Plot ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2%}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    buf2.seek(0)
    roc_base64 = base64.b64encode(buf2.getvalue()).decode("utf-8")
    plt.close(fig2)

    context = {
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2%}",
        "recall": f"{recall:.2%}",
        "f1_score": f"{f1:.2%}",
        "roc_auc": f"{roc_auc:.2%}",
        "cm_base64": cm_base64,
        "roc_base64": roc_base64,
        "best_model": model_info.get("best_model", "N/A"),
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
            probability = result_df["churn_probability"]
            prediction = result_df["churn_prediction"]
            risk_level = result_df["risk_level"]

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
