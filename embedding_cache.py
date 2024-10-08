import numpy as np
from typing import List, Tuple, Dict
import faiss
import json
import sqlite3
from logger_config import LoggerConfig

logger = LoggerConfig.setup_logger(__name__)
config = LoggerConfig.load_config()

class EmbeddingCache:
    def __init__(self, index_file: str, db_path: str):
        self.index_file = index_file
        self.db_path = db_path
        self.faiss_index = None
        self.load_index()
        self.init_db()

    def load_index(self):
        logger.info("Loading FAISS index...")
        self.faiss_index = faiss.read_index(self.index_file)
        logger.info(f"Loaded {self.faiss_index.ntotal} chunk embeddings.")

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    filename TEXT,
                    content TEXT,
                    metadata TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER,
                    chunk_index INTEGER,
                    content TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_chunk ON chunks(document_id, chunk_index)')
            conn.commit()

    def store_document(self, filename: str, content: str, metadata: dict, chunks: List[str]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents (filename, content, metadata)
                VALUES (?, ?, ?)
            ''', (filename, content, json.dumps(metadata)))
            doc_id = cursor.lastrowid

            for i, chunk in enumerate(chunks):
                cursor.execute('''
                    INSERT INTO chunks (document_id, chunk_index, content)
                    VALUES (?, ?, ?)
                ''', (doc_id, i, chunk))

            return doc_id

    def find_similar_chunks(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        if self.faiss_index is None:
            return []

        # Search in chunk-level index
        distances, indices = self.faiss_index.search(np.array([query_embedding], dtype=np.float32), k)

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS uses -1 for empty slots
                    cursor.execute('''
                        SELECT c.document_id, c.chunk_index, c.content, d.filename, d.metadata
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.id = ?
                    ''', (int(idx),))
                    row = cursor.fetchone()
                    if row:
                        doc_id, chunk_index, content, filename, metadata_str = row
                        metadata = json.loads(metadata_str)
                        results.append({
                            'distance': float(distances[0][i]),
                            'document_id': doc_id,
                            'chunk_index': chunk_index,
                            'content': content,
                            'filename': filename,
                            'metadata': metadata
                        })

        return results

    def get_all_chunks_for_document(self, document_id: int) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT chunk_index, content
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
            ''', (document_id,))
            chunks = cursor.fetchall()
            return [{'chunk_index': idx, 'content': content} for idx, content in chunks]

    def close(self):
        # No need to close anything for FAISS or SQLite (connection is closed after each operation)
        pass

# Example usage
if __name__ == "__main__":
    index_file = config['faiss_index_file']
    db_path = config['sqlite_db_path']
    cache = EmbeddingCache(index_file, db_path)
    
    # Example search
    query_embedding = [0.1] * cache.faiss_index.d  # Replace with actual query embedding
    similar_docs = cache.find_similar_chunks(query_embedding, k=5)
    print("Similar documents:")
    for dist, metadata in similar_docs:
        print(f"Distance: {dist}, File: {metadata['filename']}, Chunk: {metadata['chunk_index']}")
        print(f"Content preview: {metadata['content'][:100]}...")