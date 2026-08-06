import pandas as pd
import logging
import time
from typing import Callable, Optional, Union
import sqlalchemy
from sqlalchemy import text


# ─── Конфигурация логирования ────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ─── Мониторинг логов из БД или Excel ────────────────────────────────

class LogMonitor:
    """Мониторит логи из БД или Excel файла в реальном времени."""
    
    def __init__(self, source: Union[str], interval: int = 5, source_type: str = "auto"):
        """
        Args:
            source: Путь к Excel файлу или DB URL 
                   (postgresql://user:pass@localhost/db, 
                    mysql://user:pass@localhost/db,
                    oracle://user:pass@localhost:1521/db)
            interval: Интервал проверки в секундах (default: 5)
            source_type: "excel", "db", или "auto" (автоопределение)
        """
        self.source = source
        self.interval = interval
        self.last_processed_id = 0  # Для БД используем последний ID вместо индекса
        self.source_type = self._detect_source_type(source_type)
        self.engine: Optional[sqlalchemy.engine.Engine] = None
        self.on_item_found: Optional[Callable] = None
        
        # Инициализируем подключение к БД если нужно
        if self.source_type == "db":
            self._init_db_connection()
    
    def _detect_source_type(self, source_type: str) -> str:
        """Определяет тип источника данных."""
        if source_type == "auto":
            if source_type.startswith(("postgresql", "mysql", "oracle")):
                return "db"
            elif source_type.endswith(".xlsx") or source_type.endswith(".xls"):
                return "excel"
            else:
                return "excel"  # По умолчанию Excel
        return source_type
    
    def _init_db_connection(self):
        """Инициализирует подключение к БД."""
        try:
            self.engine = sqlalchemy.create_engine(self.source)
            # Проверяем соединение
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logging.info("✓ Успешно подключились к БД: %s", self.source)
        except Exception as e:
            logging.error("✗ Ошибка подключения к БД: %s", e)
            raise
    
    def set_callback(self, callback: Callable):
        """
        Устанавливает callback функцию для обработки найденных номеров.
        
        Args:
            callback: Функция вида: callback(item_number: str, row: dict)
        """
        self.on_item_found = callback
    
    def _process_from_excel(self) -> int:
        """Обрабатывает логи из Excel файла."""
        try:
            # Загружаем файл с логами
            df_log = pd.read_excel(self.source)
            
            # Фильтруем по функции OnEnterWaitForBatchStartConfirmSwitch
            filtered_df = df_log[df_log["Function"] == "OnEnterWaitForBatchStartConfirmSwitch"].copy()
            
            # Обрабатываем только новые строки (после last_processed_id)
            new_rows = filtered_df.iloc[self.last_processed_id:]
            
            if len(new_rows) > 0:
                # Извлекаем номер детали из Message
                new_rows['item_number'] = new_rows["Message"].str.extract(
                    r"starts with item '([^']+)'",
                    expand=False
                )
                
                # Обрабатываем каждую новую строку
                for idx, row in new_rows.iterrows():
                    item_number = row['item_number']
                    if pd.notna(item_number):
                        logging.info("✓ Найден номер детали: %s | Функция: %s", 
                                   item_number, row['Function'])
                        
                        # Вызываем callback если он установлен
                        if self.on_item_found:
                            self.on_item_found(str(item_number), row.to_dict())
                
                # Обновляем индекс последней обработанной строки
                self.last_processed_id = len(filtered_df)
            
            return self.last_processed_id
            
        except FileNotFoundError:
            logging.error("✗ Файл %s не найден", self.source)
            return self.last_processed_id
        except Exception as e:
            logging.error("✗ Ошибка обработки Excel: %s", e)
            return self.last_processed_id
    
    def _process_from_db(self) -> int:
        """Обрабатывает логи из БД."""
        try:
            with self.engine.connect() as conn:
                # SQL запрос для получения новых логов
                # Адаптируйте названия таблицы и столбцов под вашу БД
                query = text("""
                    SELECT id, "Function", "Message", timestamp
                    FROM logs
                    WHERE "Function" = 'OnEnterWaitForBatchStartConfirmSwitch'
                    AND id > :last_id
                    ORDER BY id ASC
                """)
                
                result = conn.execute(query, {"last_id": self.last_processed_id})
                rows = result.fetchall()
                
                if rows:
                    for row in rows:
                        # Извлекаем номер из Message
                        message = row[2]
                        import re
                        match = re.search(r"starts with item '([^']+)'", message)
                        item_number = match.group(1) if match else None
                        
                        if item_number:
                            logging.info("✓ Найден номер детали: %s | Функция: %s", 
                                       item_number, row[1])
                            
                            # Вызываем callback если он установлен
                            if self.on_item_found:
                                row_dict = {
                                    "id": row[0],
                                    "Function": row[1],
                                    "Message": row[2],
                                    "timestamp": row[3],
                                    "item_number": item_number
                                }
                                self.on_item_found(item_number, row_dict)
                        
                        # Обновляем последний обработанный ID
                        self.last_processed_id = row[0]
            
            return self.last_processed_id
            
        except Exception as e:
            logging.error("✗ Ошибка обработки БД: %s", e)
            return self.last_processed_id
    
    def process_logs(self) -> int:
        """Обрабатывает логи в зависимости от типа источника."""
        if self.source_type == "db":
            return self._process_from_db()
        else:
            return self._process_from_excel()
    
    def start(self):
        """Запускает цикл мониторинга (блокирует выполнение)."""
        logging.info("🚀 Запущен мониторинг из %s: %s (интервал: %ds)", 
                    self.source_type.upper(), self.source, self.interval)
        
        try:
            while True:
                self.process_logs()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logging.info("⛔ Мониторинг остановлен пользователем")


# ─── Примеры использования ────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # Пример 1: Мониторинг Excel файла
    print("=" * 60)
    print("Пример 1: Мониторинг Excel файла")
    print("=" * 60)
    
    monitor_excel = LogMonitor("KNG.xlsx", interval=5)
    
    def on_item_excel(item_number: str, row: dict):
        print(f"📦 Excel | Номер: {item_number}")
    
    monitor_excel.set_callback(on_item_excel)
    # monitor_excel.start()  # Раскомментируйте для запуска
    
    # Пример 2: Мониторинг PostgreSQL БД
    print("\n" + "=" * 60)
    print("Пример 2: Мониторинг PostgreSQL БД")
    print("=" * 60)
    
    monitor_pg = LogMonitor(
        "postgresql://cv_user:cv_pass@localhost:5432/cv_db",
        interval=5,
        source_type="db"
    )
    
    def on_item_pg(item_number: str, row: dict):
        print(f"📦 PostgreSQL | Номер: {item_number} | ID: {row['id']}")
    
    monitor_pg.set_callback(on_item_pg)
    # monitor_pg.start()  # Раскомментируйте для запуска
    
    # Пример 3: Мониторинг MySQL БД
    print("\n" + "=" * 60)
    print("Пример 3: Мониторинг MySQL БД")
    print("=" * 60)
    
    monitor_mysql = LogMonitor(
        "mysql+pymysql://root:password@localhost:3306/cv_logs",
        interval=5,
        source_type="db"
    )
    
    def on_item_mysql(item_number: str, row: dict):
        print(f"📦 MySQL | Номер: {item_number}")
    
    monitor_mysql.set_callback(on_item_mysql)
    # monitor_mysql.start()  # Раскомментируйте для запуска
    
    print("\n✓ Примеры сконфигурированы. Раскомментируйте нужный .start() для запуска")
