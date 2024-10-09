import sqlite3
import json
import numpy as np

class LLMGraphTransformer:
    def __init__(self, db_path):
        self.db_path = db_path

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
                source_doc INTEGER,
                target_doc INTEGER,
                relationship_type TEXT,
                FOREIGN KEY (source_doc) REFERENCES graph_documents (id),
                FOREIGN KEY (target_doc) REFERENCES graph_documents (id)
            )
            ''')

    def store_data(self, data):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO graph_documents (filename, chunk_index, content, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (data['filename'], data['chunk_index'], data['content'], 
                  data['embedding'].tobytes(), json.dumps(data['metadata'])))
            return cursor.lastrowid

    def store_relationship(self, source_id, target_id, relationship_type):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            INSERT INTO relationships (source_doc, target_doc, relationship_type)
            VALUES (?, ?, ?)
            ''', (source_id, target_id, relationship_type))

    def get_document_data(self, doc_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, filename, chunk_index, content, embedding, metadata
            FROM graph_documents
            WHERE id = ?
            ''', (doc_id,))
            result = cursor.fetchone()
            if result:
                id, filename, chunk_index, content, embedding_blob, metadata_json = result
                return {
                    'id': id,
                    'filename': filename,
                    'chunk_index': chunk_index,
                    'content': content,
                    'embedding': np.frombuffer(embedding_blob, dtype=np.float32),
                    'metadata': json.loads(metadata_json)
                }
            return None

    def get_relationships(self, doc_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, source_doc, target_doc, relationship_type
            FROM relationships
            WHERE source_doc = ? OR target_doc = ?
            ''', (doc_id, doc_id))
            return cursor.fetchall()