import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
np.random.seed(42)

# Завантажуємо оброблені дані з preprocess.py
X_train = pd.read_csv('training_data/X_train.csv')
X_test = pd.read_csv('training_data/X_test.csv')
y_train = pd.read_csv('training_data/y_train.csv').values.ravel()
y_test = pd.read_csv('training_data/y_test.csv').values.ravel()

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Архітектура: 64 → 32 → 16 нейронів
    activation='relu',                # Функція активації ReLU
    solver='adam',                    # Оптимізатор Adam
    learning_rate_init=0.001,         # Початкова швидкість навчання
    max_iter=200,                     # Максимум 200 ітерацій (епох)
    early_stopping=True,              # Рання зупинка при перенавчанні
    validation_fraction=0.2,          # 20% для валідації
    n_iter_no_change=15,              # Зупинка після 15 епох без покращення
    random_state=42,                  # Для відтворюваності
    verbose=True                      # Показувати прогрес
)

# Розраховуємо class_weight для балансування класів
# Це допомагає моделі краще навчатися на незбалансованих даних
class_weight = {
    0: len(y_train) / (2 * np.sum(y_train == 0)),
    1: len(y_train) / (2 * np.sum(y_train == 1))
}

# Навчаємо модель
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
test_accuracy = model.score(X_test, y_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

# Графіки точності та втрат
if hasattr(model, 'loss_curve_'):
    plt.figure(figsize=(10, 5))
    plt.plot(model.loss_curve_, linewidth=2, color='blue')
    plt.xlabel('Епоха', fontsize=12)
    plt.ylabel('Втрати (Loss)', fontsize=12)
    plt.title('Втрати під час навчання', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphics/training_loss.png', dpi=150, bbox_inches='tight')
    print("✓ Збережено: training_loss.png")
    plt.close()

if hasattr(model, 'validation_scores_'):
    plt.figure(figsize=(10, 5))
    plt.plot(model.validation_scores_, linewidth=2, color='green')
    plt.xlabel('Епоха', fontsize=12)
    plt.ylabel('Точність валідації', fontsize=12)
    plt.title('Точність на валідаційному наборі', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphics/validation_accuracy.png', dpi=150, bbox_inches='tight')
    print("✓ Збережено: validation_accuracy.png")
    plt.close()

# Зберігаємо модель у форматі pickle
with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
model_info = {
    'n_iter': model.n_iter_,
    'best_loss': model.best_loss_,
    'best_validation_score': model.best_validation_score_,
    'test_accuracy': test_accuracy,
    'test_auc': test_auc,
    'architecture': model.hidden_layer_sizes
}

with open('model/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
