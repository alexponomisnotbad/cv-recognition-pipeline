"""ONNX-классификатор (ResNet-50) для кропов детали.

Ожидается, что модель принимает вход формата [1, 3, 224, 224] (float32,
ImageNet-нормализация) и возвращает логиты размерности [1, num_classes].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class OnnxPrediction:
    label: str
    score: float
    index: int


class OnnxResnetClassifier:
    """Классификация одного кропа с помощью ONNX ResNet.

    Пример использования:

        clf = OnnxResnetClassifier("/app/models/onnx/classifier-v1.onnx", CLASS_NAMES)
        pred = clf.predict(crop_bgr)
    """

    def __init__(self, model_path: str, class_names: Sequence[str]):
        self.class_names: List[str] = list(class_names)

        logger.info("Загружаем ONNX-классификатор из %s", model_path)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB, resize 224, нормализация под ResNet (ImageNet)."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Пустой кроп передан в классификатор")

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = crop_rgb.astype(np.float32) / 255.0  # [H, W, 3]

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # [3, H, W]
        img = np.expand_dims(img, axis=0)   # [1, 3, H, W]
        return img.astype(np.float32)

    def predict(self, crop_bgr: np.ndarray) -> OnnxPrediction:
        """Вернуть лучший класс и уверенность для одного кропа."""
        inp = self._preprocess(crop_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: inp})[0]

        if outputs.ndim != 2 or outputs.shape[0] != 1:
            raise RuntimeError(f"Ожидался вывод [1, num_classes], получено {outputs.shape}")

        logits: np.ndarray = outputs[0]
        # softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        idx = int(np.argmax(probs))
        score = float(probs[idx])
        label = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        return OnnxPrediction(label=label, score=score, index=idx)
