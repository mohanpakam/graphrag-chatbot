import sqlite3
from sqlite3 import Connection
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def init_database(self):
        with self.connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            ''')
            
            # Create an index on the filename for faster lookups
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)
            ''')

            # Create an index on the chunk_index for faster lookups
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chunk_index ON documents(chunk_index)
            ''')

            print("Database initialized with required tables and indices.")

    def clear_documents(self):
        with self.connect() as conn:
            conn.execute('DELETE FROM documents')
        print("All documents cleared from the database.")