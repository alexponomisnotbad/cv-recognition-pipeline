# ─────────────────────────────────────────────────────────────────────────────
# Container 1 – Vision Pipeline 
# Base:  Ubuntu 24.04 LTS (Noble Numbat) – CPU only
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System libs + Python 3.12 ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        git \
        wget \
        curl \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# torch CPU-only (значительно меньше образ, чем CUDA-версия)
RUN pip install --no-cache-dir --break-system-packages \
        --timeout 120 \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu

COPY src/vision-pipeline/requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --timeout 120 -r requirements.txt

RUN pip install --no-cache-dir --break-system-packages \
        --timeout 120 \
        onnxruntime==1.18.1

# ── Директории для весов моделей ──────────────────────────────────────────────
RUN mkdir -p /app/models/yolo /app/models/onnx /app/models/resnet

# ── Код приложения ────────────────────────────────────────────────────────────
COPY src/vision-pipeline/ .

CMD ["python", "main.py"]


