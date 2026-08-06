import asyncio
import json
import logging
import os
from utils.read_repl import main_loop
import nats
from utils.nats_publisher import NATSPublisher
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")




# def make_message_handler(data_queue, publisher):
#     """
#     Возвращает асинхронную функцию-обработчик для NATS-сообщений.
#     Внутри неё происходит чтение из очереди данных БД, сравнение и публикация.
#     """
#     async def message_handler(msg):
#         try:

#             # 1. Читаем сообщение от детектора
#             payload = json.loads(msg.data.decode())
#             det_class = payload.get("det_class", "")
#             logger.info("Получено сообщение: det_class=%s", det_class)


#             # 2. Забираем последние данные из БД (из очереди)
#             try:
#                 db_class, color = data_queue.get_nowait()
#             except asyncio.QueueEmpty:
#                 logger.warning("Нет данных из БД (очередь пуста), пропускаем")
#                 return

#             # 3. Сравниваем классы
#             if det_class is not None or det_class != "":
#                 flag = (det_class == db_class)
#                 logger.info("Сравнение: det=%s, db=%s => flag=%s", det_class, db_class, flag)
#             else:
#                 flag = True
#                 logger.info("Сравнения нет, т.к. нет детектированной детали")

#             # 4. Публикуем результат
#             await publisher.publish(flag, det_class, db_class)

#         except Exception as e:
#             logger.exception("Ошибка в обработчике сообщения: %s", e)

#     return message_handler

IDLE_TIMEOUT = int(50)  # секунды без детекции

def peek_queue_item(queue: asyncio.Queue):
    try:
        return queue._queue[0]
    except IndexError:
        raise asyncio.QueueEmpty

async def publish_on_idle(data_queue_74, data_queue_76, publisher, event):
    """
    Периодически проверяет, были ли сообщения от детектора.
    Если в течение IDLE_TIMEOUT секунд событие не установлено – публикует heartbeat.
    """
    while True:
        try:
            # Ждём событие (приход сообщения) с таймаутом
            await asyncio.wait_for(event.wait(), timeout=IDLE_TIMEOUT)
            # Событие произошло – очищаем и продолжаем
            event.clear()
        except asyncio.TimeoutError:
            # Таймаут – сообщений не было, публикуем данные из БД
            try:
                db_class_74, color_74, seq_74 = peek_queue_item(data_queue_74)
                db_class_76, color_76, seq_76 = peek_queue_item(data_queue_76)
                
                if db_class_74 is not None:
                    await publisher.publish(
                        flag=True,
                        det_class="Это heartbeat",
                        db_class74=db_class_74,
                        db_class76=db_class_76,
                        seq_74=seq_74,
                        seq_76=seq_76,
                    )
                    logger.info("Idle heartbeat: опубликовано (db_class74=%s, db_class76=%s, seq_74=%s, seq_76=%s)", db_class_74, db_class_76, seq_74, seq_76)
                else:
                    logger.debug("Idle: очередь пуста или db_class=None")
            except asyncio.QueueEmpty:
                logger.debug("Idle: очередь пуста")
            except Exception as e:
                logger.exception("Ошибка в idle-публикации: %s", e)
            # Событие уже сброшено (после таймаута оно не установлено),
            # но на всякий случай очищаем
            event.clear()

def make_message_handler(data_queue_74, data_queue_76, publisher, event):
    async def message_handler(msg):
        try:
            payload = json.loads(msg.data.decode())
            # новая переменная
            frame_id = payload.get("frame_id", "")
        
            det_class = payload.get("classification", {}).get("label", "")
            side = payload.get("side", "")
            logger.info("Получено сообщение: det_class=%s side=%s", det_class, side)

            # Сигнализируем, что сообщение пришло
            event.set()

            # Забираем данные из очереди
            try:
                db_class_74, color_74, seq_74 = peek_queue_item(data_queue_74)
                db_class_76, color_76, seq_76 = peek_queue_item(data_queue_76)
            except asyncio.QueueEmpty:
                logger.warning("Нет данных из БД, пропускаем")
                return

            if side == "left":
                # Сравниваем
                if det_class:
                    flag = (det_class == db_class_76)
                    logger.info("Сравнение: det=%s, db=%s => flag=%s", det_class, db_class_76, flag)
                else:
                    flag = True
                    logger.info("Нет детектированного класса слева, flag=True")

                await publisher.publish(flag, det_class, None, db_class_76, seq_74=seq_74, seq_76=seq_76, frame_id=frame_id)

            if side == "right":
                # Сравниваем
                if det_class:
                    flag = (det_class == db_class_74)
                    logger.info("Сравнение: det=%s, db=%s => flag=%s", det_class, db_class_74, flag)
                else:
                    flag = True
                    logger.info("Нет детектированного класса справа, flag=True")

                await publisher.publish(flag, det_class, db_class_74, None, seq_74=seq_74, seq_76=seq_76)
                

        except Exception as e:
            logger.exception("Ошибка в обработчике сообщения: %s", e)

    return message_handler

async def main():
    nats_url = os.getenv("NATS_URL", "nats://nats:8006")
    vision_subject = os.getenv("NATS_SUBJECT_VISION", "cv.vision-pipeline")
    db_subject = os.getenv("NATS_SUBJECT_DB", "cv.db")
    output_subject = os.getenv("NATS_SUBJECT_OUTPUT", "cv.output")
   

        
    logging.info(f"Connecting to NATS at {nats_url}")
    nc = await nats.connect(nats_url, reconnect_time_wait=2, max_reconnect_attempts=-1) # бесконечное переключение 
    logger.info("Подключено к NATS")
    
    
    
    data_queue_74 = asyncio.Queue(maxsize=2)
    data_queue_76 = asyncio.Queue(maxsize=2)

    asyncio.create_task(main_loop(data_queue_74, data_queue_76))
    
    # Subscribe to subject and publish on output 

    publisher = NATSPublisher(nats_url, output_subject)     
    await publisher.connect() 
    event = asyncio.Event()
    
    asyncio.create_task(publish_on_idle(data_queue_74, data_queue_76, publisher, event))

    # message_handler = make_message_handler(data_queue, publisher)
    message_handler = make_message_handler(data_queue_74, data_queue_76, publisher, event)

    
    sub_vision = await nc.subscribe(vision_subject, cb=message_handler)
    

    logging.info(f"✓ Comparator started. Listening on {vision_subject} and publish on {output_subject}")


    # Держим приложение активным
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Завершение работы")
    finally:
        await nc.close()
        if publisher._nc:
            await publisher._nc.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
        sys.exit(0)
