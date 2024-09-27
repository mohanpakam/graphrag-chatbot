import sqlite3
import json

class LLMGraphTransformer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS graph_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB,
                metadata TEXT
            )
            ''')
            conn.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc TEXT,
                source_index INTEGER,
                target_doc TEXT,
                target_index INTEGER,
                relationship_type TEXT
            )
            ''')

    def store_data(self, data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            INSERT INTO graph_documents (filename, chunk_index, content, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (data['filename'], data['chunk_index'], data['content'], 
                  data['embedding'].tobytes(), json.dumps(data['metadata'])))

    def store_relationship(self, data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            INSERT INTO relationships (source_doc, source_index, target_doc, target_index, relationship_type)
            VALUES (?, ?, ?, ?, ?)
            ''', (data['source_doc'], data['source_index'], data['target_doc'], 
                  data['target_index'], data['relationship_type']))