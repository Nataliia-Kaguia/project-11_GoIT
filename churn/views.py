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


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
X_TEST_PATH = os.path.join(BASE_DIR, "training_data", "X_test.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "training_data", "Y_test.csv")
SCALER_PATH = os.path.join(BASE_DIR, "training_data", "scaler.pkl")

# Шлях до model_info.pkl
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.pkl")

# Завантажуємо model_info
with open(MODEL_INFO_PATH, "rb") as f:
    model_info = pickle.load(f)

numerical_cols = [
    "subscription_age",
    "bill_avg",
    "reamining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
]

model_info = joblib.load(MODEL_INFO_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def home_view(request):
    return render(request, "churn/home.html")


def feature_names_view(request):
    file_path = "churn/ml/training_data/feature_names.pkl"

    with open(file_path, "rb") as f:
        feature_names = pickle.load(f)

    context = {"feature_names": feature_names}
    return render(request, "churn/feature_names.html", context)


def model_metrics_view(request):
    # Вибираємо найкращу модель
    best_model = model_info["best_model"]
    best_metrics = model_info["models"][best_model]

    # Отримуємо confusion matrix
    cm = best_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    # Малюємо heatmap з TN, FP, FN, TP
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({best_model})")

    # Конвертуємо графік у base64 для шаблону
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # Передаємо у шаблон
    context = {
        "accuracy": None,  # можна додати, якщо обчислювати окремо
        "precision": None,
        "recall": None,
        "f1_score": best_metrics["f1_score"],
        "cm_base64": cm_base64,
        "best_model": best_model,
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

            # Викликаємо predict_churn → отримаємо DataFrame
            result_df = predict_churn(data)

            # Беремо перший рядок
            probability = result_df["churn_probability"].values[0]
            prediction = result_df["churn_prediction"].values[0]
            risk_level = result_df["risk_level"].values[0]

            # Формуємо повідомлення для шаблону
            if prediction == 1:
                message = f"⚠️ Client WILL churn (Рівень ризику: {risk_level})"
            else:
                message = f"✅ Client will NOT churn (Рівень ризику: {risk_level})"

    return render(
        request,
        "churn/predict.html",
        {
            "form": form,
            "message": message,
            "probability": probability,
        },
    )
