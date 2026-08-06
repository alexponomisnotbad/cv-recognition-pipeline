import json
import logging
from typing import Optional
import nats

# Добавляем логгер
logger = logging.getLogger("nats_publisher")

class NATSPublisher:
    def __init__(self, url: str, subject: str):
        self.url = url
        self.subject = subject
        self._nc: Optional[nats.NATS] = None

    async def connect(self):
        """Устанавливает соединение с NATS."""
        try:
            self._nc = await nats.connect(
                self.url,
                reconnect_time_wait=2,
                max_reconnect_attempts=-1
            )
            logger.info("Подключено к NATS для публикации")
        except Exception as e:
            logger.error("Ошибка подключения к NATS: %s", e)
            raise

    async def publish(
        self,
        flag: bool,
        det_class: str,
        db_class74: str,
        db_class76: str,
        seq_74: str | None = None,
        seq_76: str | None = None,
        frame_id: int | None = None ,
    ):
        if self._nc is None:
            logger.error("NATS клиент не инициализирован, публикация невозможна")
            return

        payload = {
            "flag": flag,
            "det_class": det_class,
            "db_class74": db_class74,
            "db_class76": db_class76,
            "seq_74": seq_74,
            "seq_76": seq_76,
            "frame_id": frame_id,
        }
        try:
            await self._nc.publish(
                self.subject,
                json.dumps(payload).encode("utf-8"),
            )
            logger.debug("Опубликовано: %s", payload)
        except Exception:
            logger.exception("Ошибка при публикации в NATS")