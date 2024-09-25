from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
from database_manager import DatabaseManager
from embedding_cache import EmbeddingCache
import yaml
from ai_service import get_ai_service, AIService
from logger_config import LoggerConfig
import time
import uvicorn

app = FastAPI()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
db_manager = DatabaseManager(config['database_path'])
embedding_cache = EmbeddingCache(config['database_path'])

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    embedding_cache.cache_embeddings()

@app.post("/chat")
async def chat(request: ChatRequest, ai_service: AIService = Depends(get_ai_service)):
    start_time = time.time()
    user_input = request.message
    
    logger.info(f"Received chat request: {user_input[:50]}...")

    embedding_start = time.time()
    user_embedding = ai_service.get_embedding(user_input)
    embedding_time = time.time() - embedding_start
    logger.info(f"Embedding generation took {embedding_time:.2f} seconds")

    search_start = time.time()
    similar_chunks = embedding_cache.find_similar(user_embedding, config['top_k'])
    search_time = time.time() - search_start
    logger.info(f"Similarity search took {search_time:.2f} seconds")

    context = "\n".join([f"From {filename} (chunk {chunk_index}):\n{content}" 
                         for distance, filename, chunk_index, content in similar_chunks])
    
    response_start = time.time()
    response = ai_service.generate_response(user_input, context)
    response_time = time.time() - response_start
    logger.info(f"Response generation took {response_time:.2f} seconds")

    sources = [{"filename": filename, "chunk_index": chunk_index} 
               for _, filename, chunk_index, _ in similar_chunks]
    
    total_time = time.time() - start_time
    logger.info(f"Total request processing time: {total_time:.2f} seconds")

    return {
        "response": response,
        "sources": sources
    }

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    embedding_cache.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)