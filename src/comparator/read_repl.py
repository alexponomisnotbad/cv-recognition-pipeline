import asyncio
import json
import logging
import os
import sys
import pandas as pd
from sqlalchemy import create_engine,text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("comparator")

#  ─── Конфигурация и подключение к БД ───────────────────────────────────────────────
CLASS_NAMES = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]

DB_SERVER_KNG        = os.getenv("DB_SERVER")
DB_PORT_KNG        = os.getenv("DB_PORT")
DB_NAME_KNG    = os.getenv("DB_NAME")
DB_USERNAME_KNG = os.getenv("DB_USERNAME")
DB_PASSWORD_KNG = os.getenv("DB_PASSWORD")
DB_DRIVER_KNG = os.getenv("DB_DRIVER")


DB_NAME_tdb = os.getenv("DB_NAME_TDB")

#  ─── Подключение к БД ─────────────────────────────────────────────────────────────
connection_KNG = f"mssql+pyodbc://{DB_USERNAME_KNG}:{DB_PASSWORD_KNG}@{DB_SERVER_KNG}:{DB_PORT_KNG}/{DB_NAME_KNG}?driver={DB_DRIVER_KNG}"
engine_kng = create_engine(connection_KNG)


connection_tdb = f"mssql+pyodbc://{DB_USERNAME_KNG}:{DB_PASSWORD_KNG}@{DB_SERVER_KNG}:{DB_PORT_KNG}/{DB_NAME_tdb}?driver={DB_DRIVER_KNG}"
engine_tdb = create_engine(connection_tdb)

#  ─── Чтение данных о ручках из Excel ───────────────────────────────────────────────────────
# df_hand = pd.read_excel("hand.xlsx")

# df_hand.columns = df_hand.iloc[0]
# df_hand = df_hand[1:].reset_index(drop=True)

logger.info("Подключились к KNG")
logger.info("Подключились к TDB")

df = pd.read_sql("""
SELECT
    TABLE_SCHEMA,
    TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_SCHEMA, TABLE_NAME
""", engine_kng)

print(df)

#  ─── Чтение логов из KNG и извлечение item_number ─────────────────────────────────────────────
# query_log = text(f"""
# SELECT TOP 1
#     [Function],
#     [Message]
# FROM {DB_NAME_KNG}
# WHERE [Function] = 'OnEnterWaitForBatchStartConfirmSwitch'
# ORDER BY [Timestamp] DESC
# """)

# df_log = pd.read_sql(query_log, engine_kng)

# if df_log.empty:
#     raise ValueError("Не найдено ни одной записи в логах")

# message = df_log.iloc[0]["Message"]

# logger.info("Сообщение: %s", message)

# item_number = pd.Series([message]).str.extract(
#     r"starts with item '([^']+)'",
#     expand=False
# ).iloc[0]

# if pd.isna(item_number):
#     raise ValueError("Не удалось извлечь item_number")

# item_number = str(item_number).strip()

# logger.info("item_number = %s", item_number)

# #  ─── Извлечение Model_Variant из TDB по item_number ─────────────────────────────────────────────
# query_vehicle = text("""
# SELECT TOP 1
#     Trim_Sequence_Number,
#     Model_Variant
# FROM {DB_NAME_tdb}
# WHERE Trim_Sequence_Number = :item_number
# """)

# with engine_tdb.connect() as conn:


