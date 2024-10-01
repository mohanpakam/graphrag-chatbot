import logging
import sqlite3
import numpy as np
from typing import List
from langchain.schema import Document
from langchain_ai_service import get_langchain_ai_service
from base_langchain_service import BaseLangChainAIService
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.db_path = "./vector_store.db"
        self.ai_service: BaseLangChainAIService = get_langchain_ai_service(config.get('ai_service', 'ollama'))
        self.embedding_dim = self.ai_service.get_embedding_dim()
        self.init_database()
        logger.info(f"Initialized VectorStoreManager with {self.document_count()} documents")

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()

    def add_document(self, document: Document):
        try:
            embedding = self.ai_service.get_embedding(document.page_content)
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}")
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO documents (content, embedding, metadata)
                    VALUES (?, ?, ?)
                ''', (document.page_content, embedding_blob, str(document.metadata)))
                conn.commit()
            
            logger.info(f"Added document with embedding. New count: {self.document_count()}")
        except Exception as e:
            logger.error(f"Error adding document: {e}")

    def retrieve_relevant_documents(self, query: str) -> List[Document]:
        k = config.get('top_k', 5)
        query_embedding = self.ai_service.get_embedding(query)
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {len(query_embedding)}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, embedding, metadata FROM documents")
            results = cursor.fetchall()
        
        if not results:
            return []
        
        similarities = []
        for doc_id, content, embedding_blob, metadata in results:
            doc_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, doc_id, content, metadata))
        
        similarities.sort(reverse=True)
        top_k_results = similarities[:k]
        
        documents = []
        for _, _, content, metadata in top_k_results:
            documents.append(Document(page_content=content, metadata=eval(metadata)))
        
        return documents

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def document_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]