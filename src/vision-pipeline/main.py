"""Online vision-pipeline (ONNX classifier)
=========================================

RTSP (MediaMTX) → ONNX YOLO (детекция) → вырезаем bbox → ONNX ResNet‑классификатор → NATS.

Переменные окружения:
    RTSP_URL           – адрес MediaMTX-потока (default: rtsp://mediamtx:8554/live/camera1)
    NATS_URL           – адрес NATS (default: nats://nats:4222)
    NATS_SUBJECT       – топик (default: vision.segmentation)
    YOLO_MODEL_PATH    – путь к YOLO-модели (.onnx или .pt, default: /app/models/yolo/part_detector/weights/best.onnx)
    DETECTION_CONF     – порог уверенности YOLO (default: 0.5)
    FRAME_SKIP         – обрабатывать каждый N-й кадр (default: 1)
    DEVICE             – cuda | cpu (default: cpu)
    CLASSIFIER_ONNX    – путь к ONNX-классификатору (default: /app/models/onnx/classifier-v1.onnx)
"""

import asyncio
import logging
import os
import sys

from typing import List

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vision_pipeline")

# ── Конфигурация через env ─────────────────────────────────────────────────────
RTSP_URL        = os.getenv("RTSP_URL",        "rtsp://mediamtx:8554/live/camera1")
NATS_URL        = os.getenv("NATS_URL",        "nats://nats:4222")
NATS_SUBJECT    = os.getenv("NATS_SUBJECT",    "vision.segmentation")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/app/models/yolo/part_detector/weights/best.onnx")
DETECTION_CONF  = float(os.getenv("DETECTION_CONF", "0.5"))
FRAME_SKIP      = int(os.getenv("FRAME_SKIP", "2"))
DEVICE          = os.getenv("DEVICE", "cpu")
CLASSIFIER_ONNX = os.getenv("CLASSIFIER_ONNX", "/app/models/onnx/classifier-v1.onnx")

CLASS_NAMES: List[str] = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]


async def main() -> None:
    from utils.rtsp_reader    import RTSPReader
    from utils.detector       import YOLODetector
    from utils.onnx_classifier import OnnxResnetClassifier
    from utils.nats_publisher import NATSPublisher

    logger.info("Инициализация компонентов (YOLO + Resnet-классификатор)...")

    detector = YOLODetector(YOLO_MODEL_PATH, confidence=DETECTION_CONF, device=DEVICE)
    classifier = OnnxResnetClassifier(CLASSIFIER_ONNX, CLASS_NAMES)

    publisher = NATSPublisher(NATS_URL, NATS_SUBJECT)
    await publisher.connect()

    reader = RTSPReader(RTSP_URL)

    logger.info(
        "Пайплайн запущен.\n"
        "  RTSP       : %s\n"
        "  NATS       : %s → %s\n"
        "  YOLO (pt)  : %s  (conf≥%.2f)\n"
        "  CLS (onnx) : %s",
        RTSP_URL, NATS_URL, NATS_SUBJECT,
        YOLO_MODEL_PATH, DETECTION_CONF,
        CLASSIFIER_ONNX,
    )

    frame_idx = 0

    try:
        async for frame in reader.stream():
            frame_idx += 1

            # Пропускаем кадры для снижения нагрузки
            if frame_idx % FRAME_SKIP != 0:
                continue

            # ── 1. Детекция (YOLO) ────────────────────────────────────────────
            detections = detector.detect(frame)
            if not detections:
                continue

            # Берём детекцию с максимальной уверенностью
            best = max(detections, key=lambda d: d["confidence"])
            bbox = best["bbox"]  # [x1, y1, x2, y2]
            yolo_conf = float(best["confidence"])

            # Нормализуем bbox в пределах кадра
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                logger.debug("frame %d: некорректный bbox %s", frame_idx, bbox)
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                logger.debug("frame %d: пустой кроп по bbox %s", frame_idx, bbox)
                continue

            # ── 2. Классификация детали (ONNX ResNet) ────────────────────────
            pred = classifier.predict(crop)

            # ── 3. Публикация в NATS ─────────────────────────────────────────
            await publisher.publish(
                frame_id=frame_idx,
                frame=frame,
                bbox=[x1, y1, x2, y2],
                det_class=best["class_name"],
                det_conf=yolo_conf,
                cls_label=pred.label,
                cls_conf=pred.score,
            )

            logger.info(
                "frame=%d  det_class=%s  det_conf=%.2f  cls_label=%s  cls_conf=%.3f  bbox=%s",
                frame_idx,
                best["class_name"],
                yolo_conf,
                pred.label,
                pred.score,
                [x1, y1, x2, y2],
            )

    except asyncio.CancelledError:
        logger.info("Получен сигнал остановки")
    finally:
        await publisher.close()
        logger.info("Завершение работы")


if __name__ == "__main__":
    asyncio.run(main())
