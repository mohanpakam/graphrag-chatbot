import sqlite3
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def create_sales_report_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_report (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Month TEXT,
                Electronics TEXT,
                Clothing TEXT,
                Food TEXT,
                Books TEXT,
                Total TEXT
            )
            ''')
            conn.commit()
        logger.info("Sales report table created or already exists")

    def insert_sales_report_data(self, data: List[Dict[str, Any]]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for row in data:
                cursor.execute('''
                INSERT INTO sales_report (Month, Electronics, Clothing, Food, Books, Total)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('Month', ''),
                    row.get('Electronics', ''),
                    row.get('Clothing', ''),
                    row.get('Food', ''),
                    row.get('Books', ''),
                    row.get('Total', '')
                ))
            conn.commit()
        logger.info(f"Inserted {len(data)} rows into sales_report table")

    def get_all_sales_report_data(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sales_report')
            data = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Retrieved {len(data)} rows from sales_report table")
        return data

    def execute_query(self, query, params=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()

    def get_schema_info(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema_info[table_name] = [column[1] for column in columns]
        
        return schema_info

    def get_column_names(self, query):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [description[0] for description in cursor.description]