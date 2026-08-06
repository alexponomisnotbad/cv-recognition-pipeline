FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Репозиторий Microsoft
RUN mkdir -p /etc/apt/keyrings

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor \
    -o /etc/apt/keyrings/microsoft.gpg

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/microsoft-prod.list

# ODBC Driver 18
RUN apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18
# Минимальные зависимости: NATS + tables/DB
RUN pip install --no-cache-dir \
    nats-py \
    pandas \
    SQLAlchemy \
    psycopg2-binary \
    pymysql \
    oracledb \
    openpyxl \
    pyodbc

# Копируем код компаратора
COPY src/comparator ./

CMD ["python", "main.py"]
