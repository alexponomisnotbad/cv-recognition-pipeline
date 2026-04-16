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
        onnxruntime==1.28.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --timeout 120 -r requirements.txt

# ── SAM2 (Meta) ───────────────────────────────────────────────────────────────
# Сначала ставим build-зависимости, чтобы не было таймаутов при сборке SAM2
RUN pip install --no-cache-dir --break-system-packages \
        --timeout 120 \
        setuptools>=61.0 wheel ninja
# Содержит SAM2ImagePredictor, SAM2VideoPredictor, build_sam2
RUN pip install  --break-system-packages \
        --timeout 120 \
        "git+https://github.com/facebookresearch/sam2.git@main"

# ── Директории для весов моделей ──────────────────────────────────────────────
RUN mkdir -p /app/models/yolo /app/models/sam1 /app/models/sam2

# ── Код приложения ────────────────────────────────────────────────────────────
COPY . .

CMD ["python", "main.py"]

