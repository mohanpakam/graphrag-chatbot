import sqlite3
import numpy as np
from typing import List
from langchain.schema import Document
from langchain_ai_service import get_langchain_ai_service
from base_langchain_service import BaseLangChainAIService
from embedding_cache import EmbeddingCache
from graph_manager import GraphManager
import yaml
import logging
import json

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.sqlite_db_path = config['database_path']
        self.ai_service: BaseLangChainAIService = get_langchain_ai_service(config.get('ai_service', 'ollama'))
        self.embedding_dim = self.ai_service.get_embedding_dim()
        self.init_database()
        self.embedding_cache = EmbeddingCache(self.sqlite_db_path)
        self.graph_manager = GraphManager(self.sqlite_db_path)
        self.embedding_cache.cache_embeddings()
        logger.info(f"Initialized VectorStoreManager with {self.document_count()} documents")

    def init_database(self):
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB
                )
            ''')
            conn.commit()

    def add_document(self, document: Document):
        try:
            embedding = self.ai_service.get_embedding(document.page_content)
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}")
            
            filename = document.metadata.get('filename', 'unknown')
            chunk_index = document.metadata.get('chunk_index', 0)
            
            embedding_array = np.array(embedding, dtype=np.float32)
            
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO documents (filename, chunk_index, content, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?)
                ''', (filename, chunk_index, document.page_content, json.dumps(document.metadata), embedding_array.tobytes()))
                doc_id = cursor.lastrowid
                conn.commit()
            
            document.metadata['sqlite_id'] = doc_id
            
            # Update the FAISS index in the embedding cache
            self.embedding_cache.add_embedding(doc_id, embedding_array)
            
            # Store graph data
            self.graph_manager.store_data(document, embedding)
            
            logger.info(f"Added document with embedding and graph data. New count: {self.document_count()}")
        except Exception as e:
            logger.error(f"Error adding document: {e}")

    def retrieve_relevant_documents(self, query: str) -> List[Document]:
        k = config.get('top_k', 5)
        query_embedding = self.ai_service.get_embedding(query)
        
        # Use the embedding cache to find similar documents
        similar_docs = self.embedding_cache.find_similar(query_embedding, k)
        
        relevant_docs = []
        for distance, doc_id in similar_docs:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename, chunk_index, content, metadata 
                    FROM documents 
                    WHERE id = ?
                """, (int(doc_id),))
                doc = cursor.fetchone()
                if doc:
                    filename, chunk_index, content, metadata = doc
                    doc_metadata = json.loads(metadata)
                    doc_metadata['sqlite_id'] = doc_id
                    doc_metadata['similarity_score'] = float(distance)  # Add similarity score to metadata
                    relevant_doc = Document(
                        page_content=content,
                        metadata=doc_metadata
                    )
                    relevant_docs.append(relevant_doc)

        # Fetch graph data for relevant documents
        graph_data = self.graph_manager.get_subgraph_for_documents(relevant_docs)
        
        # Attach graph data to relevant documents
        for doc in relevant_docs:
            doc_id = doc.metadata['sqlite_id']
            if doc_id in graph_data:
                doc.metadata['graph_data'] = graph_data[doc_id]

        return relevant_docs

    def document_count(self) -> int:
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

    def get_documents_by_ids(self, doc_ids: List[int]) -> List[Document]:
        relevant_docs = []
        logger.debug(f"Retrieving documents for IDs: {doc_ids}")
        with sqlite3.connect(self.sqlite_db_path) as conn:
            cursor = conn.cursor()
            id_string = ','.join(map(str, doc_ids))
            query = f"""
                SELECT id, filename, chunk_index, content, metadata 
                FROM documents 
                WHERE id IN ({id_string})
            """
            logger.debug(f"Executing query: {query}")
            cursor.execute(query)
            for row in cursor.fetchall():
                logger.debug(f"Retrieved Row: {row}")
                doc_id, filename, chunk_index, content, metadata = row
                doc_metadata = json.loads(metadata)
                doc_metadata['sqlite_id'] = doc_id
                relevant_doc = Document(
                    page_content=content,
                    metadata=doc_metadata
                )
                relevant_docs.append(relevant_doc)
        
        logger.debug(f"Retrieved {len(relevant_docs)} documents")
        return relevant_docs