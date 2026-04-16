"""
YOLODetector – обёртка над Ultralytics YOLO для детекции детали.

Возвращает список детекций вида:
    {
        "bbox":        [x1, y1, x2, y2],   # int, пиксели
        "confidence":  float,
        "class_id":    int,
        "class_name":  str,
    }
"""

import logging
import os
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Предобученная модель – скачивается автоматически из ultralytics при первом запуске
PRETRAINED_FALLBACK = "yolov8n.pt"


class YOLODetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        from ultralytics import YOLO  # lazy import – не нужен при импорте модуля

        # Если кастомная модель не найдена – используем предобученную yolov8n
        if not os.path.exists(model_path):
            logger.warning(
                "Модель %s не найдена. Используем предобученную '%s' из коробки.",
                model_path, PRETRAINED_FALLBACK,
            )
            model_path = PRETRAINED_FALLBACK

        logger.info("Загружаем YOLO из %s (device=%s)", model_path, device)
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Запускаем инференс, возвращаем список детекций."""
        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections: List[Dict[str, Any]] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            cls_name = results.names.get(cls_id, str(cls_id))

            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

        return detections
