"""
SegmentorPipeline – SAM1 → SAM2 для сегментации детали на видео.

Цепочка:
    YOLO bbox
        │
        ▼
    SAM1Segmentor  (segment_anything, sam_vit_h.pth)
        │  высококачественная маска якорного кадра
        ▼
    SAM2ImageSegmentor  (sam2, sam2.1_hiera_large.pt)
        │  уточнённая маска: bbox + маска SAM1 как подсказка (mask_input)
        ▼
    итоговая маска → NATSPublisher

Зачем SAM1 перед SAM2?
  SAM1 обучен принимать bbox и генерировать маску.
  Эту маску (перекодированную в логиты 256×256) мы передаём в SAM2
  как mask_input – SAM2 уточняет границы, убирает шум и лучше
  справляется с фоном внутри bbox.

  Без SAM1-чекпоинта пайплайн автоматически работает только через SAM2.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from segment_anything import SamPredictor, build_sam
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


logger = logging.getLogger(__name__)


# ─── Вспомогательная функция ─────────────────────────────────────────────────

def _mask_to_sam2_logit(mask: np.ndarray) -> np.ndarray:
    """
    Конвертируем маску (H, W) в mask_input для SAM2: (1, 256, 256).
    SAM2 ожидает одноканальный тензор логитов (положительные = объект).
    """
    m = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
    logit = m * 20.0 - 10.0        # True → +10 (объект),  False → -10 (фон)
    return logit[np.newaxis, :, :]  # (1, 256, 256)


# ─── SAM1 ─────────────────────────────────────────────────────────────────────

class SAM1Segmentor:
    """
    Обёртка над SamPredictor из пакета segment_anything.

    Веса: sam_vit_h.pth (~2.4 ГБ) или sam_vit_b.pth (~375 МБ).
    Тип модели передаётся через model_type ('vit_h' | 'vit_l' | 'vit_b').
    """

    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str = "cpu"):
        

        logger.info("Загружаем SAM1: %s  (type=%s, device=%s)", checkpoint, model_type, device)
        sam = build_sam(checkpoint=checkpoint)
        sam.to(device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.device = device
        logger.info("SAM1 загружен")

    def segment(self, frame_rgb: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """
        Args:
            frame_rgb – RGB-кадр (H, W, 3).
            bbox      – [x1, y1, x2, y2].
        Returns:
            bool-маска (H, W) или None.
        """
        self.predictor.set_image(frame_rgb)
        input_box = np.array(bbox, dtype=np.float32)

        with torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            return None
        return masks[int(np.argmax(scores))].astype(bool)


# ─── SAM2 ─────────────────────────────────────────────────────────────────────

class SAM2ImageSegmentor:
    """
    Обёртка над SAM2ImagePredictor.
    Принимает bbox и опциональную маску-подсказку от SAM1.
    """

    def __init__(self, config: str, checkpoint: str, device: str = "cpu"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info("Загружаем SAM2: config=%s  checkpoint=%s  device=%s", config, checkpoint, device)

        # build_sam2 ожидает просто имя файла конфига, если мы не используем hydra напрямую
        # По умолчанию он будет искать в своих пакетах.
        sam2 = build_sam2(config_file=config, ckpt_path=checkpoint, device=device)

        self.predictor = SAM2ImagePredictor(sam2)
        self.device = device
        logger.info("SAM2 загружен")

    def segment(
        self,
        frame_rgb: np.ndarray,
        bbox: list,
        hint_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Args:
            frame_rgb  – RGB-кадр (H, W, 3).
            bbox       – [x1, y1, x2, y2].
            hint_mask  – bool-маска от SAM1 (H, W), передаётся как mask_input.
        Returns:
            (mask, score) – bool-маска (H, W) и float уверенность.
        """
        mask_input = _mask_to_sam2_logit(hint_mask) if hint_mask is not None else None

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(self.device, dtype=dtype):
            self.predictor.set_image(frame_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(bbox, dtype=np.float32).reshape(1, 4),
                mask_input=mask_input,
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            return None, 0.0

        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


# ─── Единая точка входа ───────────────────────────────────────────────────────

class SegmentorPipeline:
    """
    Использование:
        pipeline = SegmentorPipeline(
            sam1_checkpoint = "/app/models/sam1/sam_vit_h.pth",   # можно None
            sam1_model_type = "vit_h",
            sam2_config     = "/app/models/sam2/sam2_hiera_large.yaml",
            sam2_checkpoint = "/app/models/sam2/sam2.1_hiera_large.pt",
            device          = "cpu",
        )
        mask, score = pipeline.segment(frame_bgr, bbox)
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        sam1_checkpoint: Optional[str] = None,
        sam1_model_type: str = "vit_h",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # SAM1 – опциональный; если чекпоинт не указан или файл не найден – пропускаем
        self.sam1: Optional[SAM1Segmentor] = None
        if sam1_checkpoint and os.path.exists(sam1_checkpoint):
            self.sam1 = SAM1Segmentor(sam1_checkpoint, sam1_model_type, device)
        else:
            if sam1_checkpoint:
                logger.warning(
                    "SAM1-чекпоинт не найден (%s). Пайплайн работает только через SAM2.",
                    sam1_checkpoint,
                )
            else:
                logger.info("SAM1_CHECKPOINT не задан. Работаем только через SAM2.")

        # SAM2 – обязательный
        self.sam2 = SAM2ImageSegmentor(sam2_config, sam2_checkpoint, device)

    def segment(
        self,
        frame: np.ndarray,       # BGR
        bbox: list,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Полный пайплайн: SAM1 (если есть) → SAM2.

        Args:
            frame – BGR-кадр.
            bbox  – [x1, y1, x2, y2].
        Returns:
            (mask, score) – bool-маска (H, W) и score SAM2.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Шаг 1: SAM1 → маска якорного кадра
        sam1_mask: Optional[np.ndarray] = None
        if self.sam1 is not None:
            sam1_mask = self.sam1.segment(rgb, bbox)
            if sam1_mask is None:
                logger.debug("SAM1 не вернул маску, используем только SAM2")

        # Шаг 2: SAM2 уточняет маску (с подсказкой от SAM1 или без)
        mask, score = self.sam2.segment(rgb, bbox, hint_mask=sam1_mask)
        return mask, score



class SAM2Segmentor:
    """
    Инициализируется один раз, переиспользуется для каждого кадра.

    Аргументы:
        config     – имя yaml-конфига SAM2 (например 'sam2_hiera_large.yaml').
                     Конфиги лежат внутри пакета sam2.
        checkpoint – путь к .pt-файлу с весами.
        device     – 'cuda' | 'cpu' | 'mps'.
    """

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(
            "Загружаем SAM2: config=%s  checkpoint=%s  device=%s",
            config, checkpoint, device,
        )

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(config_file=config, ckpt_path=checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 успешно загружен")

    # ------------------------------------------------------------------
    def segment(
        self,
        frame: np.ndarray,
        bbox: list,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Сегментирует объект внутри bbox.

        Args:
            frame  – BGR-кадр (H, W, 3).
            bbox   – [x1, y1, x2, y2] в пикселях.

        Returns:
            mask   – bool-маска (H, W)  или None при неудаче.
            score  – float уверенность SAM2.
        """
        # SAM2 ожидает RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(rgb)

            input_box = np.array(bbox, dtype=np.float32).reshape(1, 4)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            logger.debug("SAM2 не вернул маску для bbox=%s", bbox)
            return None, 0.0

        best_idx = int(np.argmax(scores))
        mask: np.ndarray = masks[best_idx].astype(bool)   # (H, W)
        score: float = float(scores[best_idx])
        return mask, score
