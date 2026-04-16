"""
MultiColorClassifier – определяет набор цветов внутри маски.

Вместо поиска одного доминирующего цвета, этот модуль проверяет наличие
нескольких предопределённых цветов (например, "хром" и "черный") и
возвращает метку, описывающую их комбинацию.
"""
import cv2
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)

# ── 1. Определяем диапазоны цветов в HSV ───────────────────────────────────────

# Точность 79 процентов, но черный как серый не распознается 
# COLOR_RANGES_HSV = {
#     "black": {
#         "lower": np.array([0, 0, 0]),
#         "upper": np.array([180, 255, 110]),  # V < 60    
#     },
#     "chrome": {
#         "lower": np.array([0, 0, 133]), # V > 130, S < 50
#         "upper": np.array([180, 50, 255]),
#     },
# }


#  точность 78 процентов, черный как серый распознается, но при этом распознается и хром
# COLOR_RANGES_HSV = {
    # "black": {
        # "lower": np.array([0, 0, 0]),
        # "upper": np.array([180, 255, 130]),  # V < 60    
    # },
    # "chrome": {
        # "lower": np.array([0, 0, 133]), # V > 130, S < 50
        # "upper": np.array([180, 50, 255]),
    # },
# }

# Точность 90%
# COLOR_RANGES_HSV = {
#     "black": {
#         "lower": np.array([0, 0, 0]),
#         "upper": np.array([180, 255, 130]),  # V < 60    
#     },
#     "chrome": {
#         "lower": np.array([0, 0, 150]), # V > 130, S < 50
#         "upper": np.array([180, 50, 255]),
#     },
# }


# 94% точность
COLOR_RANGES_HSV = {
    "black": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 130]),  # V < 60    
    },
    "chrome": {
        "lower": np.array([0, 0, 160]), # V > 130, S < 50
        "upper": np.array([180, 50, 255]),
    },
}


# Минимальный процент пикселей от общей площади маски, чтобы цвет считался значимым
MIN_AREA_PERCENT = 0.10 


class MultiColorClassifier:
    def classify(self, frame: np.ndarray, mask: np.ndarray) -> str:
        """
        Анализирует пиксели под маской и возвращает цветовую метку.

        Args:
            frame: BGR-кадр, из которого вырезается область.
            mask: Булева маска (True, где находится объект).

        Returns:
            "chrome", "black", "mixed" или "unknown".
        """
        total_mask_pixels = np.count_nonzero(mask)
        if total_mask_pixels == 0:
            return "unknown"

        # Конвертируем в HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_colors = set()

        for color_name, ranges in COLOR_RANGES_HSV.items():
            # Создаём бинарную маску для текущего цветового диапазона
            color_mask = cv2.inRange(hsv_frame, ranges["lower"], ranges["upper"])

            # Находим пересечение маски объекта и маски цвета
            intersection = cv2.bitwise_and(color_mask, color_mask, mask=mask.astype(np.uint8))
            
            # Считаем, сколько пикселей нужного цвета попало в маску
            color_pixel_count = np.count_nonzero(intersection)
            
            # Проверяем, является ли площадь значимой
            area_percentage = color_pixel_count / total_mask_pixels
            if area_percentage >= MIN_AREA_PERCENT:
                detected_colors.add(color_name)
        logger.info("detected_colors=%s", detected_colors)

        # ── 2. Применяем логику для присвоения метки ───────────────────────────
        if "chrome" in detected_colors:
           
            return "chrome"
        
        if "black" in detected_colors:
            
            return "black"
        
        return "unknown"
    
    def classify_border_and_inside(self, frame: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
        """
        Возвращает (цвет по контуру, цвет внутри).
        """
        # маска -> uint8 (0/255)
        mask_u8 = mask.astype(np.uint8) * 255

        # эрозия, чтобы убрать край
        kernel = np.ones((8, 8), np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)

        # контур = то, что было в маске, но исчезло после эрозии
        border = cv2.subtract(mask_u8, eroded)

        # внутренняя часть = эродированная маска
        inner = eroded

        # переводим обратно в bool для существующего метода
        border_bool = border > 0
        inner_bool = inner > 0

        border_label = self.classify(frame, border_bool)
        logger.info("цвет контура=%s", border_label)
        inner_label = self.classify(frame, inner_bool)
        logger.info("цвет внутри=%s", inner_label)

        return border_label, inner_label
    
    def classify_colors(self, frame: np.ndarray, mask: np.ndarray) -> str:
        border_label, inner_label = self.classify_border_and_inside(frame, mask)

        if border_label == "black" and inner_label == "chrome":
            return "black_border_chrome_inside"
        if border_label == "chrome" or inner_label == "chrome":
            return "chrome_all"
        if inner_label == "black" or border_label== "black":
            return "black_all"

        return "zalupupa"
