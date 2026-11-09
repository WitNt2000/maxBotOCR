# Указываем базовый образ с Python.
# "slim" версия меньше по размеру, что хорошо для продакшена
FROM python:3.11-slim-bookworm

# --- НОВЫЙ БЛОК: Устанавливаем системные зависимости для OpenCV ---
# apt-get update обновляет список пакетов, а apt-get install ставит нужные библиотеки.
# Флаг --no-install-recommends не даёт скачивать лишний мусор.
# rm -rf /var/lib/apt/lists/* очищает кэш, чтобы образ не занимал лишнее место.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
# ------------------------------------------------------------------

# Устанавливаем рабочую директорию
WORKDIR /bots/maxBotOCR

# Копируем только requirements.txt, чтобы использовать кэш
COPY requirements.txt .

# Устанавливаем зависимости
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем оставшиеся файлы проекта
COPY . .

# Указываем команду запуска бота
CMD ["python", "bot.py"]