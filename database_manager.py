import sqlite3
from sqlite3 import Connection
from contextlib import contextmanager
import json
import numpy as np

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
                    embedding BLOB  -- Add this line to store embeddings
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    properties TEXT,
                    document_filename TEXT NOT NULL,
                    document_index INTEGER NOT NULL,
                    FOREIGN KEY (document_filename, document_index) 
                    REFERENCES documents (filename, chunk_index)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    properties TEXT,
                    document_filename TEXT NOT NULL,
                    document_index INTEGER NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_id) REFERENCES nodes (id),
                    FOREIGN KEY (document_filename, document_index) 
                    REFERENCES documents (filename, chunk_index)
                )
            ''')
            
            # Create indices for faster lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chunk_index ON documents(chunk_index)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_node_document ON nodes(document_filename, document_index)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relationship_document ON relationships(document_filename, document_index)')

        print("Database initialized with required tables and indices.")

    def store_graph_document(self, doc_data):
        with self.connect() as conn:
            conn.execute('''
                INSERT INTO documents (filename, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?)
            ''', (doc_data['filename'], doc_data['chunk_index'], doc_data['content'], 
                  np.array(doc_data['embedding']).tobytes()))

    def store_node(self, node_data):
        with self.connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO nodes (id, type, properties, document_filename, document_index)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                node_data['id'],
                node_data['type'],
                json.dumps(node_data['properties']),
                node_data['document_filename'],
                node_data['document_index']
            ))

    def store_relationship(self, relationship_data):
        with self.connect() as conn:
            conn.execute('''
                INSERT INTO relationships (source_id, target_id, type, properties, document_filename, document_index)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                relationship_data['source_id'],
                relationship_data['target_id'],
                relationship_data['type'],
                json.dumps(relationship_data['properties']),
                relationship_data['document_filename'],
                relationship_data['document_index']
            ))

    def clear_documents(self):
        with self.connect() as conn:
            conn.execute('DELETE FROM documents')
            conn.execute('DELETE FROM nodes')
            conn.execute('DELETE FROM relationships')
        print("All documents, nodes, and relationships cleared from the database.")

    def find_similar_documents(self, query_embedding, top_k):
        query_embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, chunk_index
                FROM documents
                ORDER BY cosine_similarity(embedding, ?) DESC
                LIMIT ?
            ''', (query_embedding_bytes, top_k))
            return cursor.fetchall()

    def get_subgraph_for_documents(self, doc_ids):
        with self.connect() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in doc_ids)
            cursor.execute(f'''
                SELECT * FROM nodes
                WHERE (document_filename, document_index) IN ({placeholders})
            ''', doc_ids)
            nodes = cursor.fetchall()

            cursor.execute(f'''
                SELECT * FROM relationships
                WHERE (document_filename, document_index) IN ({placeholders})
            ''', doc_ids)
            relationships = cursor.fetchall()

        return {'nodes': nodes, 'relationships': relationships}

    def get_document_content(self, filename, chunk_index):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content FROM documents
                WHERE filename = ? AND chunk_index = ?
            ''', (filename, chunk_index))
            result = cursor.fetchone()
            return result[0] if result else None