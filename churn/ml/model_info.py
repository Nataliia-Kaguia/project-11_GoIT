import pickle
import os

BASE_DIR = "/home/sergio/Desktop/project-11_GoIT/churn/ml"
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.pkl")

with open(MODEL_INFO_PATH, "rb") as f:
    model_info = pickle.load(f)
print(type(model_info))
print(model_info)
