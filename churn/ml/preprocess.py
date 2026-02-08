import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv('internet_service_churn.csv')

# Клієнтів без контракту заповнив нулями
df['reamining_contract'] = df['reamining_contract'].fillna(0)

# 'download_avg' та 'upload_avg' - заповнюємо медіаною
df['download_avg'] = df['download_avg'].fillna(df['download_avg'].median())
df['upload_avg'] = df['upload_avg'].fillna(df['upload_avg'].median())

# Видаляємо колонку 'id' - вона не потрібна для прогнозування
df = df.drop('id', axis=1)

# Цільова змінна - чи відтік клієнта відбувся (1) чи ні (0)
X = df.drop('churn', axis=1)
y = df['churn']

# Розділяємо дані: 80% - для тренування, 20% - для тестування
# stratify=y - зберігає пропорцію класів у обох наборах
# random_state=42 - для відтворюваності результатів
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y 
)

# Визначаємо числові колонки для стандартизації
# Бінарні змінні (0/1) залишаємо без змін
numerical_cols = ['subscription_age', 'bill_avg', 'reamining_contract', 
                  'service_failure_count', 'download_avg', 'upload_avg']

# Створюємо об'єкт StandardScaler
# Він трансформує дані: (x - середнє) / стандартне_відхилення
# Результат: середнє = 0, стандартне відхилення = 1
scaler = StandardScaler()
scaler.fit(X_train[numerical_cols])
X_train_scaled = X_train.copy()
X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])

# Трансформуємо тестовий набір (використовуючи параметри з тренувального)
X_test_scaled = X_test.copy()
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Зберігаємо тренувальні та тестові набори
X_train_scaled.to_csv('training_data/X_train.csv', index=False)
X_test_scaled.to_csv('training_data/X_test.csv', index=False)
y_train.to_csv('training_data/Y_train.csv', index=False)
y_test.to_csv('training_data/Y_test.csv', index=False)

# Зберігаємо scaler для використання при прогнозуванні нових даних
with open('training_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Зберігаємо назви ознак для використання в predict.py
feature_names = list(X.columns)
with open('training_data/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
