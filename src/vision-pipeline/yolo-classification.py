"""Обучение YOLO‑классификатора по кропам, полученным от детектора.

Идея:
    1. Есть датасет для классификации в формате папок:

        DATA_DIR/
            chrome_all/
                img1.png
                ...
            black_all/
            black_border_chrome_inside/

       Имя папки = целевой класс.

    2. Для каждого исходного изображения сначала запускаем YOLO‑детектор,
       берём лучший bbox и режем по нему кроп.

    3. Кроп сохраняем в CROP_DIR с той же структурой папок по классам.

    4. Обучаем YOLO‑классификатор на CROP_DIR (а не на исходных целых изображениях).

Переменные окружения:
    INPUT_DIR      – корень исходного датасета (default: test_classification/yolo_cls_dataset)
    CROP_DIR       – корень датасета с кропами (default: test_classification/yolo_cls_crops)
    DET_MODEL_PATH – веса YOLO‑детектора (pt, задача detection)
    YOLO_MODEL_INIT – стартовая модель для классификации (default: yolov8n-cls.pt)
    EPOCHS         – эпохи обучения (default: 60)
    DEVICE         – cpu | cuda | 0 | 1 ... (default: cpu)
"""

import logging
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yolo_classification")


DATA_DIR = Path(os.getenv("INPUT_DIR", "test_classification/yolo_cls_dataset"))
CROP_DIR = Path(os.getenv("CROP_DIR", "test_classification/yolo_cls_crops"))
DET_MODEL_PATH = os.getenv("DET_MODEL_PATH", "/app/models/yolo/part_detector/weights/best.pt")
MODEL_INIT = os.getenv("YOLO_MODEL_INIT", "yolov8n-cls.pt")
EPOCHS = int(os.getenv("EPOCHS", "60"))
DEVICE = os.getenv("DEVICE", "cpu")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_crops() -> int:
    """Проходит по DATA_DIR, запускает детектор и сохраняет кропы в CROP_DIR.

    Возвращает количество успешно созданных кропов.
    """

    if not DATA_DIR.exists():
        logger.error("DATA_DIR не найден: %s", DATA_DIR)
        sys.exit(1)

    logger.info("Загрузка YOLO‑детектора: %s", DET_MODEL_PATH)
    det_model = YOLO(DET_MODEL_PATH)

    total_crops = 0

    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        class_name = class_dir.name
        logger.info("Обработка класса: %s", class_name)

        out_class_dir = CROP_DIR / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Не удалось прочитать %s", img_path)
                continue

            # Запускаем детектор на полном изображении
            results = det_model(img, device=DEVICE, verbose=False)
            if not results:
                logger.warning("%s: детектор не вернул результатов", img_path.name)
                continue

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                logger.info("%s: детектор не нашёл объектов, пропускаем", img_path.name)
                continue

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            best_idx = confs.argmax()
            x1, y1, x2, y2 = boxes_xyxy[best_idx]

            h, w = img.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h - 1))
            y2 = max(0, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                logger.warning("%s: некорректный bbox %s", img_path.name, [x1, y1, x2, y2])
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning("%s: пустой кроп по bbox %s", img_path.name, [x1, y1, x2, y2])
                continue

            out_path = out_class_dir / img_path.name
            cv2.imwrite(str(out_path), crop)
            total_crops += 1

    logger.info("Всего создано кропов: %d", total_crops)
    return total_crops


def main() -> None:
    logger.info("DATA_DIR  : %s", DATA_DIR)
    logger.info("CROP_DIR  : %s", CROP_DIR)
    logger.info("DET_MODEL : %s", DET_MODEL_PATH)
    logger.info("MODEL_INIT: %s", MODEL_INIT)
    logger.info("Epochs    : %d", EPOCHS)

    # 1. Строим датасет кропов по результатам детектора
    n_crops = build_crops()
    if n_crops == 0:
        logger.error("Не удалось создать ни одного кропа, обучение прервано")
        sys.exit(1)

    # 2. Обучаем YOLO‑классификатор на CROP_DIR
    logger.info("Начинаем обучение классификатора на кропах")
    model = YOLO(MODEL_INIT)
    model.train(
        data=str(CROP_DIR),
        epochs=EPOCHS,
        imgsz=224,
        device=DEVICE,
        project="models",
        name="yolo_cls",
    )


if __name__ == "__main__":
    main()