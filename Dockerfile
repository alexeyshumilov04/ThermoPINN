# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системные зависимости, включая OpenMP (необходим для работы генератора данных)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями Python и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код проекта
COPY . .

# Компилируем C++ генератор синтетических данных
RUN cd synthetic_data && make

# Команда по умолчанию (запуск обучения модели)
CMD ["python", "scripts/train_forward.py", "--config", "configs/forward_fnn.yaml"]