"""Обучение ResNet-50 с интеграцией MLflow и экспортом в ONNX.

Модуль предназначен для использования с MLflow и Airflow.


Структура датасета (CVAT формат):
    /path/to/dataset/
        annotations.xml
        images/
            image1.png
            ...

Переменные окружения:
    DATASET_ROOT    - корень датасета (default: /app/classification)
    ONNX_OUT        - путь для ONNX модели (default: /app/models/onnx/resnet_classifier.onnx)
    MLFLOW_TRACKING_URI - URI MLflow сервера (default: http://mlflow:5000)
    MLFLOW_EXPERIMENT   - имя эксперимента (default: resnet-classifier)
    DEVICE          - cpu | cuda | auto (default: auto)
    EPOCHS          - количество эпох (default: 60)
    BATCH_SIZE      - размер батча (default: 4)
    LR              - learning rate (default: 1e-5)
    VAL_SPLIT       - доля валидации (default: 0.2)

Пример запуска:
    python train_resnet_mlflow.py

Пример вызова из Airflow:
    from train_resnet_mlflow import train_and_export_onnx

    result = train_and_export_onnx(
        dataset_root="/app/classification",
        onnx_out="/app/models/onnx/resnet_v2.onnx",
        epochs=100,
        batch_size=8,
        lr=1e-4,
    )
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.onnx
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from utils.resnet import build_resnet50

logger = logging.getLogger("train_resnet_mlflow")
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
    """Один сэмпл из CVAT датасета."""
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
                logger.warning("Изображение %s не найдено: %s", img_name, img_path)
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
                    logger.warning("Некорректный bbox %s для %s", (xtl, ytl, xbr, ybr), img_name)
                    continue

                self.samples.append(
                    CvatSample(
                        img_path=img_path,
                        bbox=(x1, y1, x2, y2),
                        label=label,
                    )
                )

        if not self.samples:
            raise RuntimeError("В annotations.xml не найдено валидных bbox")

        logger.info("Всего сэмплов: %d", len(self.samples))
        logger.info("Метки: %s", sorted(labels_in_xml))

        unknown_labels = labels_in_xml.difference(CLASS_NAMES)
        if unknown_labels:
            raise RuntimeError(f"Неизвестные метки: {unknown_labels}")

        self.label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.img_path).convert("RGB")
        x1, y1, x2, y2 = sample.bbox
        img_crop = img.crop((x1, y1, x2, y2))

        if self.transform is not None:
            img_crop = self.transform(img_crop)

        target = self.label_to_idx[sample.label]
        return img_crop, target


def get_transforms() -> transforms.Compose:
    """ImageNet-style transforms для ResNet."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    device: torch.device,
    opset_version: int = 13,
) -> None:
    """Экспортирует PyTorch модель в ONNX формат."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Валидация ONNX
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX модель сохранена и проверена: %s", onnx_path)


def train_and_export_onnx(
    dataset_root: Optional[str] = None,
    onnx_out: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    epochs: int = 60,
    batch_size: int = 4,
    lr: float = 1e-5,
    val_split: float = 0.2,
    device: Optional[str] = None,
    register_model: bool = True,
    model_name: str = "resnet-classifier",
) -> Dict[str, Any]:
    """Основная функция обучения с MLflow трекингом и экспортом в ONNX.

    Эту функцию можно вызывать из Airflow DAG или напрямую.

    Args:
        dataset_root: Путь к датасету с annotations.xml и images/
        onnx_out: Путь для сохранения ONNX модели
        mlflow_tracking_uri: URI MLflow сервера
        mlflow_experiment: Имя эксперимента MLflow
        epochs: Количество эпох обучения
        batch_size: Размер батча
        lr: Learning rate
        val_split: Доля валидационной выборки
        device: Устройство (cpu/cuda/auto)
        register_model: Регистрировать ли модель в MLflow Model Registry
        model_name: Имя модели в Model Registry

    Returns:
        Dict с результатами:
            - run_id: ID MLflow run
            - onnx_path: Путь к ONNX модели
            - best_val_acc: Лучшая валидационная accuracy
            - final_train_acc: Финальная train accuracy
            - model_version: Версия модели в Registry (если register_model=True)
    """
    # Defaults из env переменных
    dataset_root = Path(dataset_root or os.getenv(
        "DATASET_ROOT", "/app/classification"
    ))
    onnx_out = Path(onnx_out or os.getenv(
        "ONNX_OUT", "/app/models/onnx/resnet_classifier.onnx"
    ))
    mlflow_tracking_uri = mlflow_tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow:5000"
    )
    mlflow_experiment = mlflow_experiment or os.getenv(
        "MLFLOW_EXPERIMENT", "resnet-classifier"
    )

    if device is None:
        device_env = os.getenv("DEVICE", "auto")
        if device_env == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_env

    # MLflow setup
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    logger.info("MLflow tracking URI: %s", mlflow_tracking_uri)
    logger.info("MLflow experiment: %s", mlflow_experiment)
    logger.info("Dataset: %s", dataset_root)
    logger.info("Device: %s", device)

    # Dataset
    transform = get_transforms()
    dataset = CvatBBoxDataset(dataset_root, transform=transform)

    n_total = len(dataset)
    if n_total < 2 or val_split <= 0.0:
        train_dataset = dataset
        val_dataset = None
        n_train, n_val = n_total, 0
        logger.info("Обучение без валидации (n=%d)", n_total)
    else:
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info("Train: %d, Val: %d", n_train, n_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    ) if val_dataset else None

    num_classes = len(CLASS_NAMES)
    model, device_t = build_resnet50(
        num_classes=num_classes, pretrained=True, device=device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        # Log parameters
        mlflow.log_params({
            "dataset_root": str(dataset_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "val_split": val_split,
            "device": device,
            "num_classes": num_classes,
            "class_names": ",".join(CLASS_NAMES),
            "n_train": n_train,
            "n_val": n_val,
            "optimizer": "Adam",
            "pretrained": True,
        })

        best_val_acc = 0.0
        final_train_acc = 0.0

        # Training loop
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
            final_train_acc = train_acc

            # Log train metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
            }, step=epoch)

            logger.info(
                "Epoch %d/%d train_loss=%.4f train_acc=%.4f",
                epoch, epochs, train_loss, train_acc
            )

            # Validation
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

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                # Log val metrics
                mlflow.log_metrics({
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, step=epoch)

                logger.info(
                    "Epoch %d/%d val_loss=%.4f val_acc=%.4f",
                    epoch, epochs, val_loss, val_acc
                )

        # Export to ONNX
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        export_to_onnx(model, onnx_out, device_t)

        # Log ONNX artifact
        mlflow.log_artifact(str(onnx_out), artifact_path="model")

        # Log final metrics
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "final_train_acc": final_train_acc,
        })

        # Register model in MLflow Model Registry
        model_version = None
        if register_model:
            try:
                # Создаём временный файл для регистрации ONNX модели
                model_uri = f"runs:/{run_id}/model/{onnx_out.name}"
                result = mlflow.register_model(model_uri, model_name)
                model_version = result.version
                logger.info(
                    "Модель зарегистрирована: %s версия %s",
                    model_name, model_version
                )
            except Exception as e:
                logger.warning("Не удалось зарегистрировать модель: %s", e)

        # Log class names as artifact (для inference)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(CLASS_NAMES))
            class_names_path = f.name
        mlflow.log_artifact(class_names_path, artifact_path="model")
        os.unlink(class_names_path)

    result = {
        "run_id": run_id,
        "onnx_path": str(onnx_out),
        "best_val_acc": best_val_acc,
        "final_train_acc": final_train_acc,
        "model_version": model_version,
    }

    logger.info("Обучение завершено: %s", result)
    return result


def main() -> None:
    """Entry point для запуска из командной строки."""
    # Параметры из env переменных
    epochs = int(os.getenv("EPOCHS", "60"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    lr = float(os.getenv("LR", "1e-5"))
    val_split = float(os.getenv("VAL_SPLIT", "0.2"))

    result = train_and_export_onnx(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
    )

    print(f"\nРезультат обучения:")
    print(f"  MLflow Run ID: {result['run_id']}")
    print(f"  ONNX модель: {result['onnx_path']}")
    print(f"  Best Val Accuracy: {result['best_val_acc']:.4f}")
    print(f"  Final Train Accuracy: {result['final_train_acc']:.4f}")
    if result['model_version']:
        print(f"  Model Registry версия: {result['model_version']}")


if __name__ == "__main__":
    main()
