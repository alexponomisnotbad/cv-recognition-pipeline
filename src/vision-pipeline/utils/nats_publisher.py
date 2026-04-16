"""NATSPublisher – отправляет результат детекции + классификации.

Текущий формат сообщения (JSON):

{
    "frame_id":  int,
    "timestamp": float,               # unix-time
    "detection": {
        "class":      str,           # класс из YOLO
        "confidence": float,
        "bbox":       [x1, y1, x2, y2]
    },
    "classification": {
        "label":      str,           # класс из ResNet/ONNX-классификатора
        "confidence": float
    },
    "roi_jpeg": str                  # base64-JPEG кропа по bbox
}
"""

import asyncio
import base64
import json
import logging
import time
from typing import List, Optional

import cv2
import nats
import numpy as np

logger = logging.getLogger(__name__)


# ─── Вспомогательные функции ──────────────────────────────────────────────────


def encode_roi_jpeg_bbox(frame: np.ndarray, bbox: List[int]) -> str:
    """Вырезаем ROI по bbox, кодируем в JPEG и возвращаем base64-строку."""
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ""

    ok, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Основной класс ───────────────────────────────────────────────────────────

class NATSPublisher:
    def __init__(self, url: str, subject: str):
        self.url = url
        self.subject = subject
        self._nc: Optional[nats.NATS] = None

    async def connect(self) -> None:
        logger.info("Подключаемся к NATS: %s", self.url)
        self._nc = await nats.connect(
            self.url,
            reconnect_time_wait=2,
            max_reconnect_attempts=-1,   # бесконечные переподключения
        )
        logger.info("NATS: соединение установлено")

    async def publish(
        self,
        frame_id: int,
        frame: np.ndarray,
        bbox: List[int],
        det_class: str,
        det_conf: float,
        cls_label: str,
        cls_conf: float,
    ) -> None:
        assert self._nc is not None, "Вызовите connect() перед publish()"

        payload = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "detection": {
                "class": det_class,
                "confidence": det_conf,
                "bbox": bbox,
            },
            "classification": {
                "label": cls_label,
                "confidence": cls_conf,
            },
            "roi_jpeg": encode_roi_jpeg_bbox(frame, bbox),
        }

        try:
            await self._nc.publish(
                self.subject,
                json.dumps(payload).encode("utf-8"),
            )
        except Exception:
            logger.exception("Ошибка при публикации в NATS")

    async def close(self) -> None:
        if self._nc:
            logger.info("Закрываем соединение с NATS")
            await self._nc.close()
            self._nc = None
