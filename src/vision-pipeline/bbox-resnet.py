"""

Для каждого изображения из INPUT_DIR:
    1. Находим деталь через YOLO (bbox).
    2. Вырезаем кроп по bbox.
    3. Классифицируем кроп с помощью ResNet‑50.
    4. Сохраняем картинку с рамкой и подписью + summary.json.

Переменные окружения:
    INPUT_DIR        – путь к входным изображениям (default: /app/input)
    OUTPUT_DIR       – куда сохранять результаты (default: /app/output/test_pipeline)
    YOLO_MODEL_PATH  – путь к .pt YOLO
    RESNET_CKPT      – путь к весам ResNet (state_dict, optional)
    DEVICE           – cpu | cuda (default: cpu)
    DETECTION_CONF   – порог уверенности YOLO (default: 0.25)
"""

import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.detector import YOLODetector
from utils.resnet import build_resnet50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pipeline")

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/app/input/test"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output/test_pipeline/resnet50"))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/app/models/yolo/part_detector/weights/best.pt")
RESNET_CKPT = os.getenv("RESNET_CKPT", "/app/models/resnet/resnet50_cls_430.pt")
DEVICE = os.getenv("DEVICE", "cpu")
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.25"))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


CLASS_NAMES = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]


def _preprocess_crop(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Подготовка кропа к подаче в ResNet‑50 (224x224, нормализация)."""

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(crop_rgb).float() / 255.0  # [H, W, 3]
    tensor = tensor.permute(2, 0, 1)  # [3, H, W]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    tensor = (tensor.to(device) - mean) / std
    return tensor.unsqueeze(0)  # [1, 3, 224, 224]


def _draw_overlay(frame: np.ndarray, bbox: list, cls_label: str, yolo_conf: float, cls_conf: float) -> np.ndarray:
    """Рисуем только bbox и подпись с классом ResNet и уверенностями."""

    out = frame.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)

    label = f"{cls_label} | Y:{yolo_conf:.2f} C:{cls_conf:.2f}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw, y1), (0, 220, 255), -1)
    cv2.putText(
        out,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )

    return out


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        logger.error("INPUT_DIR не найден: %s", INPUT_DIR)
        sys.exit(1)

    images = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        logger.error("В %s нет изображений", INPUT_DIR)
        sys.exit(1)

    logger.info("Загрузка YOLO: %s", YOLO_MODEL_PATH)
    detector = YOLODetector(YOLO_MODEL_PATH, confidence=DETECTION_CONF, device=DEVICE)

    logger.info("Загрузка ResNet‑50")
    model, device = build_resnet50(num_classes=len(CLASS_NAMES), pretrained=True, device=DEVICE)
    model.eval()

    if RESNET_CKPT:
        ckpt_path = Path(RESNET_CKPT)
        if ckpt_path.exists():
            logger.info("Загружаем веса ResNet из %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
        else:
            logger.warning("RESNET_CKPT указан (%s), но файл не найден", ckpt_path)

    summary: list[dict] = []

    logger.info("Начало обработки %d изображений...", len(images))

    for idx, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Не удалось прочитать %s", img_path.name)
            continue

        detections = detector.detect(frame)
        if not detections:
            logger.info("%s: YOLO не нашел объектов", img_path.name)
            cv2.imwrite(str(OUTPUT_DIR / img_path.name), frame)
            summary.append({
                "image": img_path.name,
                "objects": [],
            })
            continue

        # Берём детекцию с максимальной уверенностью
        best = max(detections, key=lambda d: d["confidence"])
        bbox = best["bbox"]
        yolo_conf = float(best["confidence"])

        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            logger.warning("%s: некорректный bbox %s", img_path.name, bbox)
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("%s: пустой кроп по bbox %s", img_path.name, bbox)
            continue

        with torch.no_grad():
            inp = _preprocess_crop(crop, device)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)[0]
            cls_idx = int(torch.argmax(probs).item())
            cls_conf = float(probs[cls_idx].item())

        cls_label = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"

        overlay_frame = _draw_overlay(frame, [x1, y1, x2, y2], cls_label, yolo_conf, cls_conf)
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), overlay_frame)

        # Формат полностью совместим с summary.json из сегментационного пайплайна:
        # [{
        #   "image": "...",
        #   "objects": [
        #       {"bbox": [...], "yolo_conf": ..., "sam_score": ..., "color": "..."}
        #   ]
        # }]
        summary.append({
            "image": img_path.name,
            "objects": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "yolo_conf": round(yolo_conf, 3),
                    # используем sam_score для уверенности ResNet
                    "sam_score": round(cls_conf, 3),
                    "color": cls_label,
                }
            ],
        })

        logger.info(
            "%s: class=%s conf=%.3f bbox=%s", img_path.name, cls_label, cls_conf, [x1, y1, x2, y2]
        )

    (OUTPUT_DIR / "summary_resnet.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
        
    logger.info("Готово! Результаты в %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
