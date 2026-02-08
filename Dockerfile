# Базовий образ з Python 3.11
FROM python:3.11-slim

# Встановлюємо системні залежності
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Копіюємо файл requirements
COPY requirements.txt .

# Встановлюємо Python-залежності
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копіюємо весь проект у контейнер
COPY . .

# Експортуємо порт для Django
EXPOSE 8000

# Команда для запуску Django development server
# Для продакшн замість "runserver" можна використовувати Gunicorn
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
