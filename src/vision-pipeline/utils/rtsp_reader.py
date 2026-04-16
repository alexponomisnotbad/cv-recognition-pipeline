"""
RTSPReader – асинхронный генератор кадров из RTSP-потока.
При обрыве соединения автоматически переподключается.
"""

import asyncio
import logging
from typing import AsyncIterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RTSPReader:
    def __init__(
        self,
        url: str,
        reconnect_delay: float = 3.0,
        max_retries: int = -1,  # -1 = бесконечно
    ):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Асинхронный генератор BGR-кадров."""
        attempts = 0
        while self.max_retries == -1 or attempts <= self.max_retries:
            cap = cv2.VideoCapture(self.url)

            if not cap.isOpened():
                attempts += 1
                logger.warning(
                    "Не удалось открыть RTSP-поток %s. "
                    "Повтор через %.1f с (попытка %d)...",
                    self.url, self.reconnect_delay, attempts,
                )
                await asyncio.sleep(self.reconnect_delay)
                continue

            logger.info("Подключились к RTSP-потоку: %s", self.url)
            attempts = 0  # сбрасываем счётчик при успешном подключении

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Потерян кадр – переподключаемся...")
                    cap.release()
                    break
                yield frame
                # Отдаём управление event-loop'у, чтобы не блокировать его
                await asyncio.sleep(0)

            await asyncio.sleep(self.reconnect_delay)
