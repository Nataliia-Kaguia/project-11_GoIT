import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

np.random.seed(42)

X_train = pd.read_csv('training_data/X_train.csv')
X_test = pd.read_csv('training_data/X_test.csv')
y_train = pd.read_csv('training_data/y_train.csv').values.ravel()
y_test = pd.read_csv('training_data/y_test.csv').values.ravel()

#Створення моделі
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

#Кросс валідація
cv_auc = cross_val_score(
    model,
    X_train,
    y_train,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
).mean()

print(f'CV ROC-AUC (fast): {cv_auc:.4f}')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

print(metrics)

epochs = range(1, len(model.loss_curve_) + 1)
plt.figure(figsize=(9, 4))
plt.plot(epochs, model.loss_curve_)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graphics/training_loss.png', dpi=150)
plt.close()

if hasattr(model, 'validation_scores_'):
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, model.validation_scores_)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphics/validation_accuracy.png', dpi=150)
    plt.close()

with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/model_info.pkl', 'wb') as f:
    pickle.dump({
        'metrics': metrics,
        'cv_auc_fast': cv_auc,
        'n_iter': model.n_iter_,
        'architecture': model.hidden_layer_sizes
    }, f)

