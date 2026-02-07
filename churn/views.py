import pickle
from django.shortcuts import render


def home_view(request):
    return render(request, "churn/home.html")


def feature_names_view(request):
    file_path = "churn/ml/training_data/feature_names.pkl"

    with open(file_path, "rb") as f:
        feature_names = pickle.load(f)

    context = {"feature_names": feature_names}
    return render(request, "churn/feature_names.html", context)


def model_metrics_view(request):

    # TODO: Реалізувати логіку для отримання метрик моделі та передати їх у шаблон

    return render(
        request, "churn/model_metrics.html", {"message": "Тут будуть метрики моделі"}
    )


def predict_view(request):
    # тимчасово просто відображаємо сторінку
    # TODO: Реалізувати логіку для обробки форми прогнозування та відображення результатів
    return render(
        request, "churn/predict.html", {"message": "Тут буде форма для прогнозування"}
    )
