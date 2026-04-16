"""
Тест пайплайна YOLO + SAM1 + SAM2 + Цвет по локальным изображениям (без RTSP).

Скрипт читает изображения, прогоняет через:
  1. YOLO (детекция bbox)
  2. SAM1 (генерация начальной маски по bbox)
  3. SAM2 (уточнение маски)
  4. MultiColorClassifier (определение цвета - хром/черный)

Все результаты (изображения с отрисовкой) сохраняются в OUTPUT_DIR.

Переменные окружения:
  INPUT_DIR         - исходные изображения (default: /app/input)
  OUTPUT_DIR        - куда сохранять результат (default: /app/output/test_pipeline)
  YOLO_MODEL_PATH   - путь к .pt YOLO
  SAM1_CHECKPOINT   - путь к SAM1 весам
  SAM1_MODEL_TYPE   - vit_h | vit_l | vit_b
  SAM2_CONFIG       - конфиг SAM2
  SAM2_CHECKPOINT   - путь к SAM2 весам
  DEVICE            - cpu | cuda
  DETECTION_CONF    - минимальная уверенность YOLO (default: 0.25)

Запуск:
  docker exec -it vision-pipeline python /app/test_sam_from_yolo_output.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pipeline")

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output/test_pipeline"))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/app/models/yolo/best.pt")
SAM1_CHECKPOINT = os.getenv("SAM1_CHECKPOINT", "/app/models/sam1/sam_vit_h.pth")
SAM1_MODEL_TYPE = os.getenv("SAM1_MODEL_TYPE", "vit_h")
SAM2_CONFIG = os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "/app/models/sam2/sam2.1_hiera_large.pt")
DEVICE = os.getenv("DEVICE", "cpu")
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.25"))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _draw_overlay(frame: np.ndarray, bbox: list, mask: np.ndarray, yolo_conf: float, sam_score: float, color_label: str) -> np.ndarray:
    out = frame.copy()

    # Полупрозрачная маска зелёного цвета
    colored = np.zeros_like(out)
    colored[mask] = (0, 255, 0)
    out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)

    # Контур маски полупрозрачным красным цветом
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour_layer = np.zeros_like(out)
        cv2.drawContours(contour_layer, contours, -1, (0, 0, 255), 12)
        out = cv2.addWeighted(out, 1.0, contour_layer, 0.8, 0)


    # BBox жёлтого цвета
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)

    # Текст над BBox
    label = f"{color_label} | Y:{yolo_conf:.2f} S:{sam_score:.2f}"
    
    # Фон для текста, чтобы лучше читалось
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
    from utils.detector import YOLODetector
    from utils.segmentor import SegmentorPipeline
    from utils.multi_color_classifier import MultiColorClassifier

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

    logger.info("Загрузка SegmentorPipeline (SAM1+SAM2)")
    segmentor = SegmentorPipeline(
        sam2_config=SAM2_CONFIG,
        sam2_checkpoint=SAM2_CHECKPOINT,
        sam1_checkpoint=SAM1_CHECKPOINT,
        sam1_model_type=SAM1_MODEL_TYPE,
        device=DEVICE,
    )
    
    logger.info("Инициализация классификатора цветов")
    color_classifier = MultiColorClassifier()

    logger.info("Начало обработки %d изображений...", len(images))
    
    summary = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Не удалось прочитать %s", img_path.name)
            continue

        detections = detector.detect(frame)
        
        if not detections:
            logger.info("%s: YOLO не нашел объектов", img_path.name)
            cv2.imwrite(str(OUTPUT_DIR / img_path.name), frame)  # сохраняем как есть
            continue

        overlay_frame = frame.copy()
        found_objects = []

        # Обрабатываем каждую детекцию на кадре
        for i, d in enumerate(detections):
            bbox = d["bbox"]
            yolo_conf = d["confidence"]
            
            mask, sam_score = segmentor.segment(frame, bbox)
            
            if mask is None:
                logger.info("%s #%d: SAM не вернул маску", img_path.name, i)
                continue
                
            color_label = color_classifier.classify_colors(frame, mask)
            
            overlay_frame = _draw_overlay(overlay_frame, bbox, mask, yolo_conf, sam_score, color_label)
            
            found_objects.append({
                "bbox": bbox,
                "yolo_conf": round(float(yolo_conf), 3),
                "sam_score": round(float(sam_score), 3),
                "color": color_label
            })
            
        logger.info("%s: найдено %d объектов", img_path.name, len(found_objects))
        
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), overlay_frame)
        
        summary.append({
            "image": img_path.name,
            "objects": found_objects
        })

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Готово! Результаты в %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
