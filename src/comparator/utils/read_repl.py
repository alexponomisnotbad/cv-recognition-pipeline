import asyncio
import json
import logging
import os
import sys
import pandas as pd
from sqlalchemy import create_engine,text
from urllib.parse import quote_plus
import time 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("comparator")

# # ─── Конфигурация и подключение к БД ───────────────────────────────────────────────
CLASS_NAMES = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]

DB_SERVER_KNG   = os.getenv("DB_SERVER")
DB_PORT_KNG     = os.getenv("DB_PORT")
DB_NAME_KNG     = os.getenv("DB_NAME")
DB_USERNAME_KNG = os.getenv("DB_USERNAME")
DB_PASSWORD_KNG = os.getenv("DB_PASSWORD")
DB_DRIVER_KNG   = os.getenv("DB_DRIVER")
DB_NAME_tbd     = os.getenv("DB_NAME_TDB")
POLL_INTERVAL   = int(os.getenv("POLL_INTERVAL"))

# # ─── Функция подключение к БД ─────────────────────────────────────────────────────────────
def create_connection_bd(database_name):

    odbc_str = (
        f"DRIVER={DB_DRIVER_KNG};"                    
        f"SERVER={DB_SERVER_KNG};"
        f"PORT={DB_PORT_KNG};"
        f"DATABASE={database_name};"
        f"UID={DB_USERNAME_KNG};"
        f"PWD={DB_PASSWORD_KNG};"
        "TrustServerCertificate=yes;"
        "LoginTimeout=60;"
    )

    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}")


# # ─── Подключение к БД ──────────────────────────────────────────

engine_kng = create_connection_bd(DB_NAME_KNG)
engine_tbd = create_connection_bd(DB_NAME_tbd)

logger.info("Подключились к KNG")
logger.info("Подключились к TDB")

# # ──── Просмотр всех таблиц в БД ─────────────────────────────

# try:
#     with engine_kng.connect() as conn:
#         result = conn.execute(text("""SELECT
#             TABLE_SCHEMA,
#             TABLE_NAME
#             FROM INFORMATION_SCHEMA.TABLES
#             WHERE TABLE_TYPE = 'BASE TABLE'
#             ORDER BY TABLE_SCHEMA, TABLE_NAME"""))
#         print(f"Успешный зАпРоС к KnG! Результат: {result.fetchone()}")
# except Exception as e:
#     print(f"Ошибка: {e}")

# try:
#     with engine_tdb.connect() as conn:
#         result = conn.execute(text("""SELECT
#             TABLE_SCHEMA,
#             TABLE_NAME
#             FROM INFORMATION_SCHEMA.TABLES
#             WHERE TABLE_TYPE = 'BASE TABLE'
#             ORDER BY TABLE_SCHEMA, TABLE_NAME"""))
#         print(f"Успешный зАпРоС к tDb! Результат: {result.fetchone()}")
# except Exception as e:
#     print(f"Ошибка: {e}")

# # ─── Функция итерации по извлечению вида ручки ─────────────────────────────────────────────
def process_iteration():

    logger.info("="*50)
    logger.info("Новая итерация")

    query_log_74 = text("""
    SELECT TOP 1
        [Function],
        [Message]
    FROM DiagnosticLog_Simple
    WHERE [Function] = 'OnEnterWaitForBatchStartConfirmSwitch' AND [StationIdentifier] = 'KITT0074'
    ORDER BY [Event] DESC
    """)

    query_log_76 = text("""
    SELECT TOP 1
        [Function],
        [Message]
    FROM DiagnosticLog_Simple
    WHERE [Function] = 'OnEnterWaitForBatchStartConfirmSwitch' AND [StationIdentifier] = 'KITT0076'
    ORDER BY [Event] DESC
    """)
    try:
    # # ─── Чтение логов из KNG из таблицы DiagnosticLog_Simple и извлечение Trim_Sequence_Number по 74 станции ─────────────────────────────────────────────
        df_log_74 = pd.read_sql(query_log_74, engine_kng)

        if df_log_74.empty:
            logger.warning("Не найдено ни одной записи в логах 76")
            return


        message_74 = df_log_74.iloc[0]["Message"]

        logger.info("Сообщение: %s", message_74)

        Trim_Sequence_Number_74 = pd.Series([message_74]).str.extract(
            r"starts with item '([^']+)'",
            expand=False
        ).iloc[0]


        if pd.isna(Trim_Sequence_Number_74):
             logger.warning("Не удалось извлечь Trim_Sequence_Number_74")
             return

        Trim_Sequence_Number_74 = str(Trim_Sequence_Number_74).strip()

        logger.info("Извлечен Trim_Sequence_Number_74 = %s", Trim_Sequence_Number_74)

    # # ─── Чтение логов из KNG из таблицы DiagnosticLog_Simple и извлечение Trim_Sequence_Number по 76 станции ─────────────────────────────────────────────
        df_log_76 = pd.read_sql(query_log_76, engine_kng)

        if df_log_76.empty:
            logger.warning("Не найдено ни одной записи в логах 76")
            return


        message_76 = df_log_76.iloc[0]["Message"]

        logger.info("Сообщение: %s", message_76)

        Trim_Sequence_Number_76 = pd.Series([message_76]).str.extract(
            r"starts with item '([^']+)'",
            expand=False
        ).iloc[0]


        if pd.isna(Trim_Sequence_Number_76):
             logger.warning("Не удалось извлечь Trim_Sequence_Number_76")
             return

        Trim_Sequence_Number_76 = str(Trim_Sequence_Number_76).strip()

        logger.info("Извлечен Trim_Sequence_Number_76 = %s", Trim_Sequence_Number_76)

    # #  ─── Извлечение Model_Variant из TDB по Trim_Sequence_Number_74 ─────────────────────────────────────────────
        query_vehicle_74 = text(f"""
        SELECT TOP 1
            [Trim_Sequence_Number],
            [Model_Variant]
        FROM tbd_Vehicle
        WHERE [Trim_Sequence_Number] = {Trim_Sequence_Number_74}
        """)

        try:
            with engine_tbd.connect() as conn:
                result_74 = conn.execute(query_vehicle_74)
                # print(f"Полученный Model_Variant: {result.fetchone()}")
                Model_Variant_74 = str(result_74.fetchone()[1]).strip()
                print(f"Полученный Model_Variant_74: {Model_Variant_74}")
        except Exception as e:
            print(f"Ошибка: {e}")
    # #  ─── Извлечение Model_Variant из TDB по Trim_Sequence_Number_76 ─────────────────────────────────────────────
        query_vehicle_76 = text(f"""
        SELECT TOP 1
            [Trim_Sequence_Number],
            [Model_Variant]
        FROM tbd_Vehicle
        WHERE [Trim_Sequence_Number] = {Trim_Sequence_Number_76}
        """)

        try:
            with engine_tbd.connect() as conn:
                result_76 = conn.execute(query_vehicle_76)
                # print(f"Полученный Model_Variant: {result.fetchone()}")
                Model_Variant_76 = str(result_76.fetchone()[1]).strip()
                print(f"Полученный Model_Variant_76: {Model_Variant_76}")
        except Exception as e:
            print(f"Ошибка: {e}")
            

    # # ─── Чтение таблицы с ручками и сранение с Model_Variant_74 и Model_Variant_76 ──────────────────────────────────────────────────────────────────────────────

        df_hand = pd.read_excel("utils/hand.xlsx")
        df_hand.columns = df_hand.iloc[0]
        df_hand = df_hand[1:].reset_index(drop=True)
        try:
            matching_74 = df_hand[df_hand['Model Variant'].astype(str).str.strip() == Model_Variant_74]
        except Exception as e:
            logger.warning("Ошибка %s", e)

        if len(matching_74) > 0:
            interior_74 = matching_74['Interior'].values[0]
            color_74 = matching_74['Color'].values[0]
            logger.info("Model Variant_74 ─ %s , Color ─ %s", Model_Variant_74, color_74)
        else:
            logger.info("Model Variant_74 %s не найден в столбце 'Model Variant'", Model_Variant_74)

        if interior_74 == "Ручка - хром, кантик - черный":
            classification_74 = CLASS_NAMES[2] # "black_border_chrome_inside"
        elif interior_74 == "Ручка и кантик - хром":
            classification_74 = CLASS_NAMES[0] # "chrome_all"
        elif interior_74 == "Полностью черная":
            classification_74 = CLASS_NAMES[1] # "black_all"

        logger.info("Найденный из DataBase 74 класс ─ %s, цвет ─ %s", classification_74, color_74)
        
        try:
            matching_76 = df_hand[df_hand['Model Variant'].astype(str).str.strip() == Model_Variant_76]
        except Exception as e:
            logger.warning("Ошибка %s", e)

        if len(matching_76) > 0:
            interior_76 = matching_76['Interior'].values[0]
            color_76 = matching_76['Color'].values[0]
            logger.info("Model Variant_76 ─ %s , Color ─ %s", Model_Variant_76, color_76)
        else:
            logger.info("Model Variant_76 %s не найден в столбце 'Model Variant'", Model_Variant_76)

        if interior_76 == "Ручка - хром, кантик - черный":
            classification_76 = CLASS_NAMES[2] # "black_border_chrome_inside"
        elif interior_76 == "Ручка и кантик - хром":
            classification_76 = CLASS_NAMES[0] # "chrome_all"
        elif interior_76 == "Полностью черная":
            classification_76 = CLASS_NAMES[1] # "black_all"

        logger.info("Найденный из DataBase 76 класс ─ %s, цвет ─ %s", classification_76, color_76)
        return Trim_Sequence_Number_74, classification_74, color_74, Trim_Sequence_Number_76, classification_76, color_76
        
    except Exception as e:
        logger.info("Ошибка в итерации: %s", e)
        return (None, None, None, None, None, None)
        
async def main_loop(data_queue_74: asyncio.Queue, data_queue_76: asyncio.Queue):
    last_seq_74 = None
    last_seq_76 = None

    def current_seq(queue: asyncio.Queue):
        try:
            return queue._queue[0][2]
        except (IndexError, AttributeError):
            return None

    while True:
        # Получаем все данные (номера, классы, цвета)
        (trim_74, class_74, color_74,
         trim_76, class_76, color_76) = process_iteration()

        # ---- Обработка для 74 ----
        if trim_74 is not None and class_74 is not None and color_74 is not None:
            # Обновляем очередь по правилу: при новом seq удаляем только первый элемент,
            # второй (если есть) станет первым, а новый добавляется вторым.
            current = current_seq(data_queue_74)
            if trim_74 != last_seq_74 and trim_74 != current:
                qsize = data_queue_74.qsize()
                if qsize == 0:
                    # пустая очередь — кладём как первый
                    await data_queue_74.put((class_74, color_74, trim_74))
                elif qsize == 1:
                    # есть один элемент — добавляем новый как второй
                    await data_queue_74.put((class_74, color_74, trim_74))
                else:
                    # 2 и более — удаляем только первый, затем добавляем новый как второй
                    try:
                        data_queue_74.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await data_queue_74.put((class_74, color_74, trim_74))
                last_seq_74 = trim_74
                logger.debug("Обновлена очередь 74: seq=%s, class=%s, color=%s, qsize_before=%s", trim_74, class_74, color_74, qsize)
        else:
            logger.debug("Данные для 74 не получены")

        # ====== Обработка для 76 ===========
        if trim_76 is not None and class_76 is not None and color_76 is not None:
            # Обновляем очередь по правилу: при новом seq удаляем только первый элемент,
            # второй (если есть) станет первым, а новый добавляется вторым.
            current = current_seq(data_queue_76)
            if trim_76 != last_seq_76 and trim_76 != current:
                qsize = data_queue_76.qsize()
                if qsize == 0:
                    await data_queue_76.put((class_76, color_76, trim_76))
                elif qsize == 1:
                    await data_queue_76.put((class_76, color_76, trim_76))
                else:
                    try:
                        data_queue_76.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await data_queue_76.put((class_76, color_76, trim_76))
                last_seq_76 = trim_76
                logger.debug("Обновлена очередь 76: seq=%s, class=%s, color=%s, qsize_before=%s", trim_76, class_76, color_76, qsize)
        else:
            logger.debug("Данные для 76 не получены")

        await asyncio.sleep(POLL_INTERVAL)

