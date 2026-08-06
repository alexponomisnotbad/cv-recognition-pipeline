"""Online vision-pipeline (ONNX classifier)
=========================================

RTSP → ONNX YOLO (детекция) → вырезаем bbox → ONNX ResNet‑классификатор → NATS.

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
from pathlib import Path
import subprocess

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
RTSP_URL        = os.getenv("RTSP_URL",        "rtsp://rtsp:rtsp@10.135.141.44:554/axis-media/media.amp")
NATS_URL        = os.getenv("NATS_URL",        "nats://nats:4222")
NATS_SUBJECT    = os.getenv("NATS_SUBJECT",    "vision.segmentation")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/app/models/yolo/part_detector/weights/best.onnx")
DETECTION_CONF  = float(os.getenv("DETECTION_CONF", "0.5"))
FRAME_SKIP      = int(os.getenv("FRAME_SKIP", "2"))
DEVICE          = os.getenv("DEVICE", "cpu")
CLASSIFIER_ONNX = os.getenv("CLASSIFIER_ONNX", "/app/models/onnx/classifier-v1.onnx")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output/test_on_server"))

CLASS_NAMES: List[str] = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]
WIDTH = 960
HEIGHT = 540
FPS = 25


async def main() -> None:
    from utils.rtsp_reader    import RTSPReader
    from utils.detector       import YOLODetector
    from utils.onnx_classifier import OnnxResnetClassifier
    from utils.nats_publisher import NATSPublisher

    logger.info("Инициализация компонентов (YOLO + Resnet-классификатор)...")

    detector = YOLODetector(YOLO_MODEL_PATH, confidence=DETECTION_CONF, device=DEVICE)
    classifier = OnnxResnetClassifier(CLASSIFIER_ONNX, CLASS_NAMES)

    # publisher = NATSPublisher(NATS_URL, NATS_SUBJECT)
    # await publisher.connect()

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
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "verbose",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", f"{FPS}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "mpegts",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "rtsp://mediamtx:8554/avtozavod_cv"
        ],
        stdin=subprocess.PIPE,
    )

    try:
        async for frame in reader.stream():
            frame_idx += 1

            # Пропускаем кадры для снижения нагрузки
            if frame_idx % FRAME_SKIP != 0:
                continue
            
            # ── 1. Детекция (YOLO) ────────────────────────────────────────────
            detections = detector.detect(frame)
            if detections:
                

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
                # await publisher.publish(
                #     frame_id=frame_idx,
                #     frame=frame,
                #     bbox=[x1, y1, x2, y2],
                #     det_class=best["class_name"],
                #     det_conf=yolo_conf,
                #     cls_label=pred.label,
                #     cls_conf=pred.score,
                # )

                logger.info(
                    "frame=%d  det_class=%s  det_conf=%.2f  cls_label=%s  cls_conf=%.3f  bbox=%s",
                    frame_idx,
                    best["class_name"],
                    yolo_conf,
                    pred.label,
                    pred.score,
                    [x1, y1, x2, y2],
                )

                # ── 3. Рисуем результат на исходном кадре (frame) ──────────────
                
                # Цвет рамки и текста (например, зеленый: BGR = (0, 255, 0))
                color = (0, 255, 0)
                
                # Рисуем ограничивающую рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                
                # Формируем текст для отрисовки (например, класс и уверенность)
                label = f"{pred.label}: {pred.score:.2f}"
                font_scale = 0.6
                thickness = 2
                
                # Вычисляем размеры текста для создания красивой фоновой плашки
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Позиция текста (чуть выше рамки)
                text_x = x1
                text_y = max(0, y1 - 10)
                
                # Рисуем прямоугольник подложки для читаемости текста
                cv2.rectangle(
                    frame, 
                    (text_x, text_y - text_height - baseline), 
                    (text_x + text_width, text_y + baseline), 
                    (0, 0, 0), # Черный фон
                    cv2.FILLED
                )
                
                # Накладываем сам текст поверх плашки
                cv2.putText(
                    frame, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
                )

                # output_filename = f"frame_{frame_idx:06d}.jpg"
                # output_path = OUTPUT_DIR / output_filename

                # cv2.imwrite(str(output_path), frame)
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                ffmpeg.stdin.write(frame.tobytes())

            else:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                ffmpeg.stdin.write(frame.tobytes())


    except asyncio.CancelledError:
        logger.info("Получен сигнал остановки")
    finally:
        # await publisher.close()
        logger.info("Завершение работы")


if __name__ == "__main__":
    asyncio.run(main())
