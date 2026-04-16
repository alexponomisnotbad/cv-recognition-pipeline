"""ResNet-классификатор для изображений (ResNet-50 под твою задачу).

Использует готовую архитектуру ResNet-50 из torchvision и заменяет последний
полносвязный слой под нужное число классов.

Пример использования (в отдельном скрипте):

	from utils.resnet import build_resnet50

	# Например, три класса: chrome_all, black_all, black_border_chrome_inside
	model, device = build_resnet50(num_classes=3, pretrained=True, device="cuda")
	logits = model(batch.to(device))  # batch: [B, 3, H, W]

Модель ожидает вход 3xHxW в диапазоне [0, 1] или [0, 255] с последующей
нормализацией так же, как в ImageNet (см. torchvision.transforms).
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
	"""Обёртка над torchvision.models.resnet50 для классификации.

	- Загружает ResNet-50 .
	- Заменяет последний слой `fc` на Linear(num_features, num_classes).
	"""

	def __init__(self, num_classes: int, pretrained: bool = True) -> None:
		super().__init__()

		# Берём стандартный ResNet-50
		if pretrained:
			try:
				backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
			except AttributeError:
				backbone = models.resnet50(pretrained=True)
		else:
			backbone = models.resnet50(weights=None)

		in_features = backbone.fc.in_features
		backbone.fc = nn.Linear(in_features, num_classes)

		self.backbone = backbone

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)


def build_resnet50(
	num_classes: int,
	pretrained: bool = True,
	device: Optional[str] = None,
) -> tuple[ResNetClassifier, torch.device]:
	"""Создаёт ResNet-50 классификатор и переносит на нужное устройство.

	Возвращает (модель, device), чтобы ты мог легко делать .to(device).
	"""

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	device_obj = torch.device(device)
	model = ResNetClassifier(num_classes=num_classes, pretrained=pretrained)
	model.to(device_obj)
	return model, device_obj

