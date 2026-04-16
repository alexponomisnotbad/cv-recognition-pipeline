"""
Валидация связки YOLO‑детектор + YOLO‑классификатор.

Для каждого изображения из INPUT_DIR:
    1. Находим деталь через YOLO‑детектор (bbox).
    2. Вырезаем кроп по bbox.
    3. Классифицируем кроп с помощью обученного YOLO‑классификатора.
    4. Сохраняем картинку с рамкой и подписью + summary_yolo_cls.json.

Переменные окружения:
    INPUT_DIR       – путь к входным изображениям (default: /app/input)
    OUTPUT_DIR      – куда сохранять результаты (default: /app/output/yolo_cls)
    DET_MODEL_PATH  – путь к .pt детектора YOLO (объект)
    CLS_MODEL_PATH  – путь к .pt классификатора YOLO (runs/classify/.../weights/best.pt)
    DEVICE          – cpu | cuda | 0 | 1 ... (default: cpu)
    DETECTION_CONF  – порог уверенности детектора (default: 0.25)
"""

import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.detector import YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yolo_val_class")

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output/yolo_cls"))

DET_MODEL_PATH = os.getenv("DET_MODEL_PATH", "/app/models/yolo/part_detector/weights/best.pt")
CLS_MODEL_PATH = os.getenv("CLS_MODEL_PATH", "/app/runs/classify/models/yolo_cls3/weights/best.pt")
DEVICE = os.getenv("DEVICE", "cpu")
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.25"))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _draw_overlay(frame: np.ndarray, bbox: list, cls_label: str, yolo_conf: float, cls_conf: float) -> np.ndarray:
    """Рисуем bbox и подпись с классом YOLO‑классификатора и уверенностями."""

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

    logger.info("Загрузка YOLO‑детектора: %s", DET_MODEL_PATH)
    detector = YOLODetector(DET_MODEL_PATH, confidence=DETECTION_CONF, device=DEVICE)

    logger.info("Загрузка YOLO‑классификатора: %s", CLS_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)

    summary: list[dict] = []

    logger.info("Начало обработки %d изображений...", len(images))

    for idx, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Не удалось прочитать %s", img_path.name)
            continue

        detections = detector.detect(frame)
        if not detections:
            logger.info("%s: YOLO‑детектор не нашёл объектов", img_path.name)
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

        # Классификация кропа с помощью YOLO‑классификатора
        # YOLO сам приведёт размер к нужному (imgsz=224 при обучении)
        cls_results = cls_model(crop, device=DEVICE, verbose=False)
        if not cls_results:
            logger.warning("%s: YOLO‑классификатор не вернул результатов", img_path.name)
            continue

        r = cls_results[0]
        if r.probs is None:
            logger.warning("%s: YOLO‑классификатор не содержит вероятностей (probs=None)", img_path.name)
            continue

        top1_idx = int(r.probs.top1)
        cls_conf = float(r.probs.top1conf)
        cls_label = r.names[top1_idx]

        overlay_frame = _draw_overlay(frame, [x1, y1, x2, y2], cls_label, yolo_conf, cls_conf)
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), overlay_frame)

        # Формат summary совместим с текущим metrics.py и summary.json:
        summary.append({
            "image": img_path.name,
            "objects": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "yolo_conf": round(yolo_conf, 3),
                    # sam_score используем для уверенности классификатора
                    "sam_score": round(cls_conf, 3),
                    "color": cls_label,
                }
            ],
        })

        logger.info(
            "%s: class=%s conf=%.3f bbox=%s",
            img_path.name,
            cls_label,
            cls_conf,
            [x1, y1, x2, y2],
        )

    (OUTPUT_DIR / "summary_yolo_cls.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Готово! Результаты и summary_yolo_cls.json в %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
