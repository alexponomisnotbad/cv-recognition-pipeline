import pandas as pd
import sqlite3
import logging
from typing import Optional
import os
import time
import json
import sqlalchemy
import psycopg2
import pymysql
import oracledb
import re

CLASS_NAMES = [
    "chrome_all",
    "black_all",
    "black_border_chrome_inside",
]
#  ─── Вспомогательная функция для подключения к БД ────────────────────────────────
def connect_to_db(db_url: str):
    """Подключаемся к БД по URL и возвращаем SQLAlchemy engine."""
    try:
        engine = sqlalchemy.create_engine(db_url)
        # Проверяем соединение
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logging.info("Успешно подключились к БД")
        return engine
    except Exception as e:
        logging.error("Ошибка подключения к БД: %s", e)
        raise
# Таблица как датафрейм (сейчас просто анализ таблицы Excel, но в будущем SQL)

df_log = pd.read_excel("KNG.xlsx") 

df_tbd_Vehicle = pd.read_excel("tbd_Vehicle.xlsx") 

df_hand = pd.read_excel("hand.xlsx") 

# Используем первую строку df_hand как имена столбцов
df_hand.columns = df_hand.iloc[0]
df_hand = df_hand[1:].reset_index(drop=True)

# Фильтруем по функции OnEnterWaitForBatchStartConfirmSwitch
filtered_df = df_log[df_log["Function"] == "OnEnterWaitForBatchStartConfirmSwitch"]

# Извлекаем номер из сообщения типа:
# "Waiting on switch to confirm the current rack starts with item 'XXX' and ends with 'XXX'."
#

extract_item_number = filtered_df['item_number'] = filtered_df["Message"].str.extract(
    r"starts with item '([^']+)'",
    expand=False
)

# Обращаемся к первому элементу 

item_number = extract_item_number.iloc[0]
print(f"Извлечённый номер детали: {item_number}")

# По этому элементу обращаемся к другой таблице и извлекаем информацию о детали в столбце Trim_Sequence_Number и находим в этой строке Model_Variant

if pd.notna(item_number):
    # Отладка: проверяем типы данных
    print(f"Тип item_number: {type(item_number)}, Значение: {repr(item_number)}")
    print(f"Тип Trim_Sequence_Number: {df_tbd_Vehicle['Trim_Sequence_Number'].dtype}")
    
    # Конвертируем в строку для сравнения
    item_number_str = str(item_number).strip()
    
    # Ищем строку в df_tbd_Vehicle где Trim_Sequence_Number совпадает с item_number
    matching_rows = df_tbd_Vehicle[df_tbd_Vehicle["Trim_Sequence_Number"].astype(str).str.strip() == item_number_str]
    
    if len(matching_rows) > 0:
        model_variant = matching_rows["Model_Variant"].values[0]
        print(f"Найден Model_Variant: {model_variant}")
    else:
        print(f"Деталь с номером {item_number_str} не найдена в tbd_Vehicle")


if pd.notna(model_variant):
    model_variant_clean = str(model_variant).strip()
    
    # Ищем model_variant в столбце 'Model Variant'

    matching = df_hand[df_hand['Model Variant'].astype(str).str.strip() == model_variant_clean]
    
    if len(matching) > 0:
        interior = matching['Interior'].values[0]
        color = matching['Color'].values[0]
        print(f"✓ Найдено совпадение в столбце 'Model Variant'")
        print(f"  Interior: {interior}")
        print(f"  Color: {color}")
    else:
        print(f"Model Variant {model_variant_clean} не найден в столбце 'Model Variant'")

if interior == "Ручка - хром, кантик - черный":
    classification = CLASS_NAMES[2] # "black_border_chrome_inside"
elif interior == "Ручка и кантик - хром":
    classification = CLASS_NAMES[0] # "chrome_all"
elif interior == "Полностью черная":
    classification = CLASS_NAMES[1] # "black_all"



