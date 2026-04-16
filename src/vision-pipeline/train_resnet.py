"""Обучение ResNet-50 по датасету из CVAT (annotations.xml + images/)

Структура датасета:

    /app/classification/
        annotations.xml
        images/
            Screenshot ....png
            ...

В annotations.xml каждая <image> содержит как минимум один <box> с атрибутами:
    label="chrome_all | black_all | black_border_chrome_inside"
    xtl, ytl, xbr, ybr - координаты bbox (float, в пикселях)

Скрипт:
    - парсит annotations.xml
    - для каждого box обрезает кроп по bbox
    - по кропам обучает ResNet-50 как классификатор 3 классов
    - делит выборку на train/val случайно по VAL_SPLIT

Переменные окружения:
    DATASET_ROOT   - корень датасета (default: /app/classification)
    CKPT_OUT       - путь для сохранения весов (default: /app/models/resnet/resnet50_cls.pt)
    DEVICE         - cpu | cuda | auto (default: auto)
    EPOCHS         - количество эпох (default: 10)
    BATCH_SIZE     - размер батча (default: 16)
    LR             - learning rate (default: 1e-4)
    VAL_SPLIT      - доля валидации (default: 0.2)
"""

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from utils.resnet import build_resnet50

logger = logging.getLogger("train_resnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


CLASS_NAMES: List[str] = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]


@dataclass
class CvatSample:
    img_path: Path
    bbox: Tuple[int, int, int, int]
    label: str


class CvatBBoxDataset(Dataset):
    """Dataset, который читает CVAT annotations.xml и режет кропы по bbox."""

    def __init__(self, root: Path, transform=None) -> None:
        self.root = root
        self.transform = transform
        self.samples: List[CvatSample] = []

        ann_path = root / "annotations.xml"
        images_dir = root / "images"

        if not ann_path.exists():
            raise FileNotFoundError(f"Не найден annotations.xml: {ann_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Не найдена папка images: {images_dir}")

        tree = ET.parse(ann_path)
        root_el = tree.getroot()

        labels_in_xml = set()

        for img_el in root_el.findall("image"):
            img_name = img_el.get("name")
            if not img_name:
                continue
            img_path = images_dir / img_name
            if not img_path.exists():
                logger.warning("Изображение %s из XML не найдено по пути %s", img_name, img_path)
                continue

            width = int(float(img_el.get("width", "0")))
            height = int(float(img_el.get("height", "0")))

            for box_el in img_el.findall("box"):
                label = box_el.get("label")
                if not label:
                    continue
                labels_in_xml.add(label)

                xtl = float(box_el.get("xtl", "0"))
                ytl = float(box_el.get("ytl", "0"))
                xbr = float(box_el.get("xbr", "0"))
                ybr = float(box_el.get("ybr", "0"))

                x1 = max(0, min(int(xtl), width - 1))
                y1 = max(0, min(int(ytl), height - 1))
                x2 = max(0, min(int(xbr), width))
                y2 = max(0, min(int(ybr), height))

                if x2 <= x1 or y2 <= y1:
                    logger.warning("Пропускаем некорректный bbox %s для %s", (xtl, ytl, xbr, ybr), img_name)
                    continue

                self.samples.append(
                    CvatSample(
                        img_path=img_path,
                        bbox=(x1, y1, x2, y2),
                        label=label,
                    )
                )

        if not self.samples:
            raise RuntimeError("В annotations.xml не найдено ни одного валидного bbox")

        logger.info("Всего кропов (samples) в датасете: %d", len(self.samples))
        logger.info("Метки в xml: %s", sorted(labels_in_xml))

        unknown_labels = labels_in_xml.difference(CLASS_NAMES)
        if unknown_labels:
            raise RuntimeError(f"В xml есть метки, не входящие в CLASS_NAMES: {unknown_labels}")

        self.label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]
        img = Image.open(sample.img_path).convert("RGB")
        x1, y1, x2, y2 = sample.bbox
        img_crop = img.crop((x1, y1, x2, y2))

        if self.transform is not None:
            img_crop = self.transform(img_crop)

        target = self.label_to_idx[sample.label]
        return img_crop, target


def train_resnet() -> None:
    dataset_root = Path(os.getenv("DATASET_ROOT", "/app/test_classification/validation_classification"))
    ckpt_out = Path(os.getenv("CKPT_OUT", "/app/models/resnet/resnet50_cls_430.pt"))
    epochs = int(os.getenv("EPOCHS", "50"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    lr = float(os.getenv("LR", "1e-6"))
    val_split = float(os.getenv("VAL_SPLIT", "0.2"))
    device_env = os.getenv("DEVICE", "cpu")

    if device_env == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_env

    logger.info("DATASET_ROOT=%s", dataset_root)

    # Трансформации как в ImageNet для ResNet
    base_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = CvatBBoxDataset(dataset_root, transform=base_tf)

    n_total = len(dataset)
    if n_total < 2 or val_split <= 0.0:
        train_dataset = dataset
        val_dataset = None
        logger.info("Мало данных или VAL_SPLIT=0, обучаемся без валидации (n=%d)", n_total)
    else:
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        if n_train < 1:
            n_train = n_total - 1
            n_val = 1

        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info("Train samples: %d, Val samples: %d", n_train, n_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        val_loader = None

    num_classes = len(CLASS_NAMES)
    logger.info("CLASS_NAMES: %s", CLASS_NAMES)
    logger.info("num_classes=%d", num_classes)

    model, device_t = build_resnet50(num_classes=num_classes, pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device_t)
            targets = targets.to(device_t)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        logger.info("Epoch %d/%d train_loss=%.4f train_acc=%.4f", epoch, epochs, train_loss, train_acc)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device_t)
                    targets = targets.to(device_t)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += preds.eq(targets).sum().item()

            val_loss = val_loss / val_total if val_total > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            logger.info("Epoch %d/%d val_loss=%.4f val_acc=%.4f", epoch, epochs, val_loss, val_acc)

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_out)
    logger.info("Сохранили веса ResNet в %s", ckpt_out)


if __name__ == "__main__":
    train_resnet()
