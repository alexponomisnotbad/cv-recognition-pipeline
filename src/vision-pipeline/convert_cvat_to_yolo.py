# import os
# import shutil
# import random
# import xml.etree.ElementTree as ET
# from pathlib import Path

# # === НАСТРОЙКИ ===
# ANNOTATIONS_XML = "dataset_detection/annotations.xml"   # путь к XML
# IMAGES_DIR      = "dataset_detection/images"            # где лежат картинки
# OUTPUT_DIR      = "dataset_detection/yolo_detection_dataset"  # куда собрать датасет
# VAL_RATIO       = 0.2                                          # долd_dя val

# IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".png"}

# def main():
#     random.seed(42)

#     ann_path = Path(ANNOTATIONS_XML)
#     images_dir = Path(IMAGES_DIR)
#     out_dir = Path(OUTPUT_DIR)

#     out_train = out_dir / "train"
#     out_val   = out_dir / "val"
#     out_train.mkdir(parents=True, exist_ok=True)
#     out_val.mkdir(parents=True, exist_ok=True)

#     tree = ET.parse(ann_path)
#     root = tree.getroot()

#     samples = []  # (image_path, class_name)

#     for image_elem in root.findall("image"):
#         img_name = image_elem.get("name")
#         if not img_name:
#             continue

#         # берём первый <box> как источник класса
#         box_elem = image_elem.find("box")
#         if box_elem is None:
#             print(f"[WARN] image '{img_name}' has no <box>, skipping")
#             continue

#         class_name = box_elem.get("label")
#         if not class_name:
#             print(f"[WARN] image '{img_name}' has <box> without label, skipping")
#             continue

#         img_path = images_dir / img_name
#         if not img_path.exists():
#             print(f"[WARN] image file not found: {img_path}")
#             continue

#         if img_path.suffix.lower() not in IMAGE_EXTS:
#             print(f"[WARN] unsupported image ext: {img_path}")
#             continue

#         samples.append((img_path, class_name))

#     print(f"Всего примеров: {len(samples)}")

#     random.shuffle(samples)
#     n_val = int(len(samples) * VAL_RATIO)
#     val_samples = samples[:n_val]
#     train_samples = samples[n_val:]

#     def export_split(split_samples, split_root):
#         for img_path, class_name in split_samples:
#             class_dir = split_root / class_name
#             class_dir.mkdir(parents=True, exist_ok=True)
#             dst = class_dir / img_path.name
#             shutil.copy2(img_path, dst)

#     export_split(train_samples, out_train)
#     export_split(val_samples, out_val)

#     print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
#     print(f"Готовый датасет: {out_dir.resolve()}")

# if __name__ == "__main__":
#     main()


import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================
# Настройки
# ==========================

ANNOTATIONS_XML = "dataset_detection/annotations.xml"
IMAGES_DIR = "dataset_detection/images"
OUTPUT_DIR = "dataset_detection/yolo_dataset"

VAL_RATIO = 0.2
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


def convert_box(xmin, ymin, xmax, ymax, width, height):
    """
    CVAT -> YOLO
    """

    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height

    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return x_center, y_center, box_width, box_height


def main():

    ann_path = Path(ANNOTATIONS_XML)
    images_dir = Path(IMAGES_DIR)
    out_dir = Path(OUTPUT_DIR)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "images/train").mkdir(parents=True)
    (out_dir / "images/val").mkdir(parents=True)

    (out_dir / "labels/train").mkdir(parents=True)
    (out_dir / "labels/val").mkdir(parents=True)

    tree = ET.parse(ann_path)
    root = tree.getroot()

    # ==========================
    # Собираем список классов
    # ==========================

    class_names = sorted(
        {
            box.get("label")
            for box in root.iter("box")
        }
    )

    class_to_id = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    print("Классы:")

    for k, v in class_to_id.items():
        print(v, k)

    # ==========================
    # Список изображений
    # ==========================

    images = root.findall("image")

    random.shuffle(images)

    n_val = round(len(images) * VAL_RATIO)

    val_images = images[:n_val]
    train_images = images[n_val:]

    def export_split(images_list, split):

        img_out = out_dir / "images" / split
        label_out = out_dir / "labels" / split

        for image in images_list:

            img_name = image.attrib["name"]

            width = float(image.attrib["width"])
            height = float(image.attrib["height"])

            src_img = images_dir / img_name

            if not src_img.exists():
                print("Image not found:", src_img)
                continue

            if src_img.suffix.lower() not in IMAGE_EXTS:
                continue

            shutil.copy2(src_img, img_out / img_name)

            label_path = label_out / (Path(img_name).stem + ".txt")

            with open(label_path, "w") as f:

                for box in image.findall("box"):

                    cls = class_to_id[box.attrib["label"]]

                    xmin = float(box.attrib["xtl"])
                    ymin = float(box.attrib["ytl"])
                    xmax = float(box.attrib["xbr"])
                    ymax = float(box.attrib["ybr"])

                    xc, yc, bw, bh = convert_box(
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        width,
                        height,
                    )

                    f.write(
                        f"{cls} "
                        f"{xc:.6f} "
                        f"{yc:.6f} "
                        f"{bw:.6f} "
                        f"{bh:.6f}\n"
                    )

    export_split(train_images, "train")
    export_split(val_images, "val")

    # ==========================
    # data.yaml
    # ==========================

    with open(out_dir / "data.yaml", "w") as f:

        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")

        f.write("names:\n")

        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print()
    print("===================================")
    print("YOLO dataset created!")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print("===================================")


if __name__ == "__main__":
    main()