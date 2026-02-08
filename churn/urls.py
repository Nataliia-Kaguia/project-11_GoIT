from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_view, name="home"),
    path("features/", views.feature_names_view, name="feature_names"),
    path("metrics/", views.model_metrics_view, name="model_metrics"),
    path("predict/", views.predict_view, name="predict"),
]
