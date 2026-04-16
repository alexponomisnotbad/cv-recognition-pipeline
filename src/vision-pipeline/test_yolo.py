"""
test_yolo.py – проверка YOLO на папке с фотографиями или RTSP-потоке.

Режимы (SOURCE_TYPE):
  images  – обработать все jpg/png из INPUT_DIR  (default)
  rtsp    – читать RTSP-стрим

Переменные окружения:
  SOURCE_TYPE       – images | rtsp  (default: images)
  INPUT_DIR         – папка с фото на хосте, смонтированная в контейнер
                      (default: /app/input)
  RTSP_URL          – RTSP-адрес (используется при SOURCE_TYPE=rtsp)
                      (default: rtsp://mediamtx:8554/live/camera1)
  YOLO_MODEL_PATH   – путь к .pt (если нет файла – auto yolov8n.pt)
  DETECTION_CONF    – порог уверенности (default: 0.25)
  OUTPUT_DIR        – куда сохранять аннотированные фото (default: /app/output)
  DEVICE            – cpu | cuda (default: cpu)
  SAVE_EVERY        – (только rtsp) сохранять кадр каждые N кадров (default: 30)
  MAX_FRAMES        – (только rtsp) 0 = бесконечно (default: 0)

Запуск:
  docker compose -f docker/docker-compose.yml run --rm yolo-test
"""

import logging
import os
import sys
import time
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yolo_test")

SOURCE_TYPE = os.getenv("SOURCE_TYPE", "images")   # images | rtsp
INPUT_DIR   = os.getenv("INPUT_DIR",   "/app/input")
RTSP_URL    = os.getenv("RTSP_URL",    "rtsp://mediamtx:8554/live/camera1")
MODEL_PATH  = os.getenv("YOLO_MODEL_PATH", "/app/models/yolo/weights/best.pt")
CONF        = float(os.getenv("DETECTION_CONF", "0.25"))
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  "/app/output")
DEVICE      = os.getenv("DEVICE",      "cpu")
SAVE_EVERY  = int(os.getenv("SAVE_EVERY", "30"))   # только для rtsp
MAX_FRAMES  = int(os.getenv("MAX_FRAMES", "0"))     # только для rtsp

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from utils.detector import YOLODetector

    detector = YOLODetector(MODEL_PATH, confidence=CONF, device=DEVICE, imgsz=640)

    logger.info("Модель   : %s", MODEL_PATH)
    logger.info("Conf     : %.2f", CONF)
    logger.info("Output   : %s", OUTPUT_DIR)

    if SOURCE_TYPE == "images":
        _run_images(detector)
    else:
        _run_rtsp(detector)


# ─── Режим: папка с фотографиями ─────────────────────────────────────────────

def _run_images(detector) -> None:
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        logger.error("INPUT_DIR не найден: %s", INPUT_DIR)
        sys.exit(1)

    images = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        logger.error(
            "В папке %s нет изображений (%s)",
            INPUT_DIR, ", ".join(IMAGE_EXTS),
        )
        sys.exit(1)

    logger.info("Найдено изображений: %d  →  %s", len(images), INPUT_DIR)

    class_stats: dict[str, int] = {}
    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Не удалось прочитать: %s", img_path.name)
            continue

        detections = detector.detect(frame)

        for d in detections:
            class_stats[d["class_name"]] = class_stats.get(d["class_name"], 0) + 1

        if detections:
            summary = ", ".join(
                f"{d['class_name']}={d['confidence']:.2f}" for d in detections
            )
            logger.info("[%d/%d] %s  →  %s", idx, len(images), img_path.name, summary)
        else:
            logger.info("[%d/%d] %s  →  (ничего не найдено)", idx, len(images), img_path.name)

        # Сохраняем аннотированный кадр с тем же именем
        annotated = _draw(frame, detections)
        out_path = os.path.join(OUTPUT_DIR, img_path.name)
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    elapsed = time.time() - t_start
    _print_stats(class_stats, len(images), elapsed)


# ─── Режим: RTSP-поток ───────────────────────────────────────────────────────

def _run_rtsp(detector) -> None:
    logger.info("RTSP     : %s", RTSP_URL)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logger.error("Не удалось открыть RTSP: %s", RTSP_URL)
        sys.exit(1)

    frame_idx = 0
    saved = 0
    class_stats: dict[str, int] = {}
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            detections = detector.detect(frame)

            for d in detections:
                class_stats[d["class_name"]] = class_stats.get(d["class_name"], 0) + 1

            if frame_idx % SAVE_EVERY == 0:
                if detections:
                    summary = ", ".join(
                        f"{d['class_name']}={d['confidence']:.2f}" for d in detections
                    )
                    logger.info("frame=%d  →  %s", frame_idx, summary)
                else:
                    logger.info("frame=%d  →  (ничего)", frame_idx)

                annotated = _draw(frame, detections)
                out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1

            if MAX_FRAMES > 0 and frame_idx >= MAX_FRAMES:
                break
    except KeyboardInterrupt:
        logger.info("Остановлено")
    finally:
        cap.release()
        _print_stats(class_stats, frame_idx, time.time() - t_start)


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _draw(frame, detections: list):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, label, (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return out


def _print_stats(class_stats: dict, total: int, elapsed: float) -> None:
    logger.info("─" * 60)
    logger.info("Обработано: %d  (%.1f с)", total, elapsed)
    logger.info("Аннотации сохранены → %s", OUTPUT_DIR)
    logger.info("Статистика классов:")
    for cls, cnt in sorted(class_stats.items(), key=lambda x: -x[1]):
        logger.info("  %-25s  %d раз", cls, cnt)
    if not class_stats:
        logger.info("  (ничего не обнаружено)")
        logger.info("")
        logger.info("Совет: снизьте DETECTION_CONF (сейчас %.2f) или обучите", CONF)
        logger.info("       собственную модель – деталей нет в COCO-80.")


if __name__ == "__main__":
    main()
