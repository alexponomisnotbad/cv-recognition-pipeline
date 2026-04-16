FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Минимальные зависимости: NATS-клиент
RUN pip install --no-cache-dir nats-py

# Копируем код компаратора
COPY src/comparator ./comparator

CMD ["python", "-m", "comparator.main"]
