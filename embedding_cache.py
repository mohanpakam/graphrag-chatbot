import sqlite3
import numpy as np
from typing import List, Tuple
import faiss
from database_manager import DatabaseManager
from ai_service import get_ai_service
from logger_config import LoggerConfig

logger = LoggerConfig.setup_logger(__name__)


config = LoggerConfig.load_config()

class EmbeddingCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        ai_service = get_ai_service()
        self.embedding_dim = ai_service.get_embedding_dim()
        self.faiss_index = None

    def cache_embeddings(self):
        logger.info("Caching embeddings from SQLite to FAISS...")
        base_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index = faiss.IndexIDMap(base_index)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM documents")
            embeddings = []
            ids = []
            for row in cursor.fetchall():
                doc_id, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                if len(embedding) != self.embedding_dim:
                    logger.warning(f"Embedding dimension mismatch for document {doc_id}. Expected {self.embedding_dim}, got {len(embedding)}. Skipping.")
                    continue
                embeddings.append(embedding)
                ids.append(doc_id)
            
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.faiss_index.add_with_ids(embeddings_array, np.array(ids))
        
        logger.info(f"Cached {self.faiss_index.ntotal} embeddings in FAISS index.")

    def add_embedding(self, doc_id: int, embedding: np.ndarray):
        if self.faiss_index is None:
            logger.error("add_embedding - FAISS index not initialized. Call cache_embeddings() first.")
            return
        if len(embedding) != self.embedding_dim:
            logger.error(f"Embedding dimension mismatch for document {doc_id}. Expected {self.embedding_dim}, got {len(embedding)}. Skipping.")
            return
        self.faiss_index.add_with_ids(embedding.reshape(1, -1), np.array([doc_id]))

    def find_similar(self, query_embedding: List[float], top_k: int) -> List[Tuple[float, int]]:
        if self.faiss_index is None:
            logger.warning("find_similar - FAISS index not initialized. Call cache_embeddings() first.")
            self.cache_embeddings()

        query_embedding_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if query_embedding_array.shape[1] != self.embedding_dim:
            logger.error(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {query_embedding_array.shape[1]}.")
            return []

        distances, indices = self.faiss_index.search(query_embedding_array, top_k)
        
        return list(zip(distances[0], indices[0]))

    def close(self):
        # No need to close anything for FAISS
        pass

# Example usage
if __name__ == "__main__":
    db_path = config['database_path']
    cache = EmbeddingCache(db_path)
    
    # Cache embeddings from the documents table
    cache.cache_embeddings()
    
    # Example search
    query_embedding = [0.1] * cache.embedding_dim  # Replace with actual query embedding
    similar_docs = cache.find_similar(query_embedding, top_k=5)
    print("Similar documents:", similar_docs)