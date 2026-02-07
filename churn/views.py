from django.shortcuts import render
import pickle
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

numerical_cols = [
    "subscription_age",
    "bill_avg",
    "reamining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


def home_view(request):
    return render(request, "churn/home.html")


def feature_names_view(request):
    file_path = "churn/ml/training_data/feature_names.pkl"

    with open(file_path, "rb") as f:
        feature_names = pickle.load(f)

    context = {"feature_names": feature_names}
    return render(request, "churn/feature_names.html", context)


def model_metrics_view(request):
    # 1. Завантажуємо тестові дані
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    # 2. Масштабуємо числові колонки
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 3. Прогноз на тестовому наборі
    y_pred = model.predict(X_test_scaled)

    # 4. Обчислюємо метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 5. Готуємо підписи для TN / FP / FN / TP
    tn, fp, fn, tp = cm.ravel()
    labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])

    # 6. Створюємо heatmap із підписами
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (TN / FP / FN / TP)")

    # 7. Зберігаємо графік у буфер
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # 8. Передаємо у шаблон
    context = {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1_score": round(f1, 3),
        "cm_base64": cm_base64,
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
