from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import numpy as np
from database_manager import DatabaseManager
from embedding_cache import EmbeddingCache
import yaml
from langchain_ai_service import get_langchain_ai_service
from logger_config import LoggerConfig
import time
import uvicorn
import os
from graph_rag import GraphRAG

app = FastAPI()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
db_manager = DatabaseManager(config['database_path'])
embedding_cache = EmbeddingCache(config['database_path'])
graph_rag = GraphRAG()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

class ChatRequest(BaseModel):
    message: str
    reset: bool = False
    vertex_ai_token: str = None

ai_service = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    embedding_cache.cache_embeddings()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/chat")
async def chat(request: ChatRequest):
    global ai_service

    if request.vertex_ai_token:
        os.environ['VERTEX_AI_TOKEN'] = request.vertex_ai_token
        ai_service = get_langchain_ai_service(config['ai_service'])
        logger.info("Updated Vertex AI token and reinitialized AI service")

    if ai_service is None:
        ai_service = get_langchain_ai_service(config['ai_service'])

    if request.reset:
        if hasattr(ai_service, 'clear_conversation_history'):
            ai_service.clear_conversation_history()
        logger.info("Chat history has been reset")
        return {"response": "Chat history has been cleared."}

    # Get query embedding
    query_embedding = ai_service.get_embedding(request.message)

    # Find similar documents
    similar_chunks = embedding_cache.find_similar(query_embedding, top_k=3)

    # Prepare context from similar documents
    context = "\n".join([f"From {filename} (chunk {chunk_index}):\n{content}" 
                         for distance, filename, chunk_index, content in similar_chunks])

    # Generate response using the AI service
    response = ai_service.generate_response(request.message, context)
    
    return {"response": response}

@app.post("/rag-chat")
async def rag_chat(request: ChatRequest):
    if request.reset:
        graph_rag.memory_manager.clear_chat_history()
        logger.info("RAG chat history has been reset")
        return {"response": "RAG chat history has been cleared."}

    response = graph_rag.process_query(request.message)
    
    return {"response": response}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    embedding_cache.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)