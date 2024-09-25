import sqlite3
import sqlite_vec
from typing import List, Tuple
import struct
from database_manager import DatabaseManager
from ai_service import get_ai_service
from logger_config import LoggerConfig

logger = LoggerConfig.setup_logger(__name__)

class EmbeddingCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        ai_service = get_ai_service()
        self.embedding_dim = ai_service.get_embedding_dim()
        self.memory_conn = None

    def get_memory_db_connection(self):
        if self.memory_conn is None:
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self.memory_conn = conn
        return self.memory_conn

    def serialize_f32(self, vector: List[float]) -> bytes:
        """Serializes a list of floats into a compact "raw bytes" format"""
        return struct.pack("%sf" % len(vector), *vector)

    def deserialize_f32(self, raw_bytes: bytes) -> List[float]:
        """Deserializes a compact "raw bytes" format into a list of floats"""
        return list(struct.unpack("%sf" % (len(raw_bytes) // 4), raw_bytes))

    def setup_cache(self):
        conn = self.get_memory_db_connection()
        conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS embedding_vectors 
        USING vec0(embedding float[{self.embedding_dim}])
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            chunk_index INTEGER,
            content TEXT
        )
        """)
        logger.info("In-memory embedding cache tables created.")

    def cache_embeddings(self):
        # Ensure connection is established and tables are created
        self.get_memory_db_connection()
        self.setup_cache()

        # Clear existing cache
        #self.memory_conn.execute("DELETE FROM embedding_vectors")
        #self.memory_conn.execute("DELETE FROM embedding_metadata")
        #logger.info("Existing in-memory embedding cache cleared.")

        # Fetch all documents from the disk database
        with self.db_manager.connect() as disk_conn:
            documents = disk_conn.execute("""
            SELECT id, filename, chunk_index, content, embedding 
            FROM documents
            """).fetchall()

        # Insert into in-memory embedding cache
        for doc in documents:
            doc_id, filename, chunk_index, content, embedding = doc
            serialized_embedding = self.serialize_f32(self.deserialize_f32(embedding))
            self.memory_conn.execute("""
            INSERT INTO embedding_vectors (rowid, embedding) VALUES (?, ?)
            """, (doc_id, serialized_embedding))
            self.memory_conn.execute("""
            INSERT INTO embedding_metadata (id, filename, chunk_index, content)
            VALUES (?, ?, ?, ?)
            """, (doc_id, filename, chunk_index, content))

        self.memory_conn.commit()
        logger.info(f"Cached {len(documents)} embeddings in memory.")

    def find_similar(self, query_embedding: List[float], top_k: int) -> List[Tuple[float, str, int, str]]:
        serialized_query = self.serialize_f32(query_embedding)
        
        results = self.memory_conn.execute("""
        SELECT
            v.distance,
            m.filename,
            m.chunk_index,
            m.content
        FROM embedding_vectors v
        JOIN embedding_metadata m ON v.rowid = m.id
        WHERE v.embedding MATCH ? 
        AND k = ?
        ORDER BY v.distance
        """, (serialized_query, top_k)).fetchall()

        return results

    def get_cached_embedding(self, document_id: int) -> List[float]:
        result = self.memory_conn.execute("""
        SELECT embedding FROM embedding_vectors WHERE rowid = ?
        """, (document_id,)).fetchone()
        
        if result:
            return self.deserialize_f32(result[0])
        return None

    def close(self):
        if self.memory_conn:
            self.memory_conn.close()
            self.memory_conn = None
            logger.info("In-memory database connection closed.")

# Example usage
if __name__ == "__main__":
    db_path = "vectordb.sqlite"
    cache = EmbeddingCache(db_path)
    
    # Cache embeddings from the documents table
    cache.cache_embeddings()
    
    # Example search
    query_embedding = [0.1] * cache.embedding_dim
    similar_docs = cache.find_similar(query_embedding, top_k=5)
    print("Similar documents:", similar_docs)

    # Close the in-memory database connection
    cache.close()