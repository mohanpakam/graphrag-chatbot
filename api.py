from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import numpy as np
from utils.database_manager import DatabaseManager
from utils.embedding_cache import EmbeddingCache
from graphrag import GraphRAG
import yaml
from ai_services import get_langchain_ai_service
import time
import uvicorn
import os
from config import LoggerConfig

app = FastAPI()

config = LoggerConfig.load_config()
db_manager = DatabaseManager(config['database_path'])
embedding_cache = EmbeddingCache(config['faiss_index_file'], config['database_path'])
graph_rag = GraphRAG()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

class ChatRequest(BaseModel):
    message: str
    reset: bool = False

class ProductionIssueRequest(BaseModel):
    query: str

ai_service = None
schema_info = None

@app.on_event("startup")
async def startup_event():
    global ai_service, schema_info
    logger.info("Starting up the application")
    ai_service = get_langchain_ai_service(config['ai_service'])
    schema_info = db_manager.get_schema_info()
    logger.info("AI service initialized and schema info fetched")

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
    if request.reset:
        graph_rag.reset_conversation()
        logger.info("Chat history has been reset")
        return {"response": "Chat history has been cleared."}

    response = graph_rag.process_query(request.message)
    return {"response": response}

@app.post("/production_issue_query")
async def production_issue_query(request: ProductionIssueRequest):
    query_embedding = ai_service.get_embedding(request.query)
    similar_chunks = embedding_cache.find_similar_chunks(query_embedding, k=5)
    
    # Use GraphRAG to process the query with the retrieved chunks
    response = graph_rag.process_query(request.query, similar_chunks)
    
    return {"response": response}

@app.post("/production_issue_analysis")
async def production_issue_analysis(request: ProductionIssueRequest):
    query_embedding = ai_service.get_embedding(request.query)
    similar_chunks = embedding_cache.find_similar_chunks(query_embedding, k=5)
    
    # Use GraphRAG to process the analysis query
    analysis_summary = graph_rag.process_analysis_query(request.query, similar_chunks)
    
    return {"analysis_summary": analysis_summary}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    embedding_cache.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)