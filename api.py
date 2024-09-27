from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
from database_manager import DatabaseManager
from embedding_cache import EmbeddingCache
import yaml
from langchain_ai_service import get_langchain_ai_service
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
    reset: bool = False

ai_service = get_langchain_ai_service(config['ai_service'])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    embedding_cache.cache_embeddings()

# Existing chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    if request.reset:
        if hasattr(ai_service, 'clear_conversation_history'):
            ai_service.clear_conversation_history()
        logger.info("Chat history has been reset")
        return {"response": "Chat history has been cleared."}

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

# New RAG chat endpoint
def retrieve_relevant_subgraph(query: str, top_k: int = 5):
    query_embedding = ai_service.get_embedding(query)
    similar_docs = db_manager.find_similar_documents(query_embedding, top_k)
    subgraph = db_manager.get_subgraph_for_documents(similar_docs)
    return subgraph, similar_docs

def subgraph_to_context(subgraph, similar_docs):
    context = "Relevant information:\n"
    for node in subgraph['nodes']:
        context += f"- {node[2]}: {node[1]}\n"
    for rel in subgraph['relationships']:
        context += f"- {rel[1]} {rel[3]} {rel[2]}\n"
    
    context += "\nRelevant document excerpts:\n"
    for doc in similar_docs:
        content = db_manager.get_document_content(doc[1], doc[2])
        context += f"- {content[:200]}...\n"
    
    return context

@app.post("/rag-chat")
async def rag_chat(request: ChatRequest):
    if request.reset:
        ai_service.clear_conversation_history()
        logger.info("RAG chat history has been reset")
        return {"response": "RAG chat history has been cleared."}

    start_time = time.time()
    user_input = request.message
    
    logger.info(f"Received RAG chat request: {user_input[:50]}...")

    subgraph_start = time.time()
    subgraph, similar_docs = retrieve_relevant_subgraph(user_input, config['top_k'])
    subgraph_time = time.time() - subgraph_start
    logger.info(f"Subgraph retrieval took {subgraph_time:.2f} seconds")

    context_start = time.time()
    context = subgraph_to_context(subgraph, similar_docs)
    context_time = time.time() - context_start
    logger.info(f"Context generation took {context_time:.2f} seconds")
    
    response_start = time.time()
    response = ai_service.generate_response(user_input, context)
    response_time = time.time() - response_start
    logger.info(f"Response generation took {response_time:.2f} seconds")

    sources = [{"filename": doc[1], "chunk_index": doc[2]} for doc in similar_docs]
    
    total_time = time.time() - start_time
    logger.info(f"Total RAG request processing time: {total_time:.2f} seconds")

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