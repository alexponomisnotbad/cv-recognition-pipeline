import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# === НАСТРОЙКИ ===
ANNOTATIONS_XML = "test_classification/annotations.xml"   # путь к XML
IMAGES_DIR      = "test_classification/images"            # где лежат картинки
OUTPUT_DIR      = "test_classification/yolo_cls_dataset"  # куда собрать датасет
VAL_RATIO       = 0.2                                          # доля val

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".png"}

def main():
    random.seed(42)

    ann_path = Path(ANNOTATIONS_XML)
    images_dir = Path(IMAGES_DIR)
    out_dir = Path(OUTPUT_DIR)

    out_train = out_dir / "train"
    out_val   = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(ann_path)
    root = tree.getroot()

    samples = []  # (image_path, class_name)

    for image_elem in root.findall("image"):
        img_name = image_elem.get("name")
        if not img_name:
            continue

        # берём первый <box> как источник класса
        box_elem = image_elem.find("box")
        if box_elem is None:
            print(f"[WARN] image '{img_name}' has no <box>, skipping")
            continue

        class_name = box_elem.get("label")
        if not class_name:
            print(f"[WARN] image '{img_name}' has <box> without label, skipping")
            continue

        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[WARN] image file not found: {img_path}")
            continue

        if img_path.suffix.lower() not in IMAGE_EXTS:
            print(f"[WARN] unsupported image ext: {img_path}")
            continue

        samples.append((img_path, class_name))

    print(f"Всего примеров: {len(samples)}")

    random.shuffle(samples)
    n_val = int(len(samples) * VAL_RATIO)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    def export_split(split_samples, split_root):
        for img_path, class_name in split_samples:
            class_dir = split_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / img_path.name
            shutil.copy2(img_path, dst)

    export_split(train_samples, out_train)
    export_split(val_samples, out_val)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Готовый датасет: {out_dir.resolve()}")

if __name__ == "__main__":
    main()