"""Comparator service
=====================

Получает по NATS:
  1) сообщения от vision-pipeline (детекция + классификация детали),
  2) сообщения от внешнего сервера с информацией о "текущей реальной детали".

Сервер пока не реализован, поэтому приём его сообщений сделан как заглушка:
мы просто подписываемся на отдельный subject и логируем всё, что туда придёт.

Переменные окружения:
    NATS_URL        – адрес NATS (default: nats://nats:4222)
    VISION_SUBJECT  – subject с сообщениями vision-pipeline
                      (default: значение CMP_SUB_SUBJECT или vision.segmentation)
    SERVER_SUBJECT  – subject с сообщениями сервера (default: vision.current_part)
    DB_URL          – строка подключения к БД (пока не используется)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import nats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("comparator")


NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
VISION_SUBJECT = os.getenv("VISION_SUBJECT", os.getenv("CMP_SUB_SUBJECT", "vision.segmentation"))
SERVER_SUBJECT = os.getenv("SERVER_SUBJECT", "vision.current_part")
DB_URL = os.getenv("DB_URL", "postgresql://cv_user:cv_pass@db:5432/cv_db")


async def main() -> None:
    logger.info("Comparator: подключаемся к NATS %s", NATS_URL)
    nc = await nats.connect(NATS_URL)

    state: Dict[str, Optional[Dict[str, Any]]] = {"current_part": None}

    async def handle_vision(msg: nats.aio.msg.Msg) -> None:  # type: ignore[name-defined]
        try:
            payload = json.loads(msg.data.decode("utf-8"))
        except Exception:
            logger.exception("Ошибка парсинга JSON из vision-pipeline")
            return

        frame_id = payload.get("frame_id")
        det = payload.get("detection", {}) or {}
        cls = payload.get("classification", {}) or {}

        logger.info(
            "VISION: frame=%s det_class=%s det_conf=%.3f cls_label=%s cls_conf=%.3f",
            frame_id,
            det.get("class"),
            float(det.get("confidence", 0.0) or 0.0),
            cls.get("label"),
            float(cls.get("confidence", 0.0) or 0.0),
        )

        if state["current_part"] is not None:
            logger.info("Текущая реальная деталь (snapshot): %s", state["current_part"])
        else:
            logger.info("Текущая реальная деталь ещё не получена (ждём сообщение от сервера)")

        # TODO: здесь позже можно добавить сравнение с эталоном и запись в БД по DB_URL

    async def handle_server(msg: nats.aio.msg.Msg) -> None:  # type: ignore[name-defined]
        """Заглушка приёма сообщения от сервера.

        Ожидаем, что сервер в будущем будет слать JSON с описанием
        "настоящей текущей детали". Пока просто сохраняем последнюю
        версию и логируем.
        """
        try:
            payload = json.loads(msg.data.decode("utf-8"))
        except Exception:
            logger.exception("Ошибка парсинга JSON от сервера")
            return

        state["current_part"] = payload
        logger.info("SERVER: обновлена текущая реальная деталь: %s", payload)

    await nc.subscribe(VISION_SUBJECT, cb=handle_vision)
    await nc.subscribe(SERVER_SUBJECT, cb=handle_server)

    logger.info(
        "Comparator запущен. Подписки:\n  vision: %s\n  server: %s",
        VISION_SUBJECT,
        SERVER_SUBJECT,
    )

    try:
        await asyncio.Future()  # run forever
    finally:
        await nc.drain()
        await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
