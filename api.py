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
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

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

class ProductionIssueRequest(BaseModel):
    query: str

ai_service = None
schema_info = None

@app.on_event("startup")
async def startup_event():
    global ai_service, schema_info
    logger.info("Starting up the application")
    embedding_cache.cache_embeddings()
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

    if request.vertex_ai_token:
        os.environ['VERTEX_AI_TOKEN'] = request.vertex_ai_token
        ai_service = get_langchain_ai_service(config['ai_service'])
        logger.info("Updated Vertex AI token and reinitialized AI service")

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

@app.post("/reset_production_chat")
async def reset_production_chat():
    global ai_service
    if hasattr(ai_service, 'clear_conversation_history'):
        ai_service.clear_conversation_history()
    logger.info("Production support chat history has been reset")
    return {"message": "Production support chat history has been cleared."}

@app.post("/production_issue_query")
async def production_issue_query(request: ProductionIssueRequest):
    global ai_service, schema_info

    # Generate SQL query from natural language
    sql_query = ai_service.generate_sql_query(request.query, schema_info)

    # Execute the query
    result = db_manager.execute_query(sql_query)

    return {"sql_query": sql_query, "result": result}

@app.post("/production_issue_analysis")
async def production_issue_analysis(request: ProductionIssueRequest):
    global ai_service, schema_info

    # Generate SQL query from natural language
    sql_query = ai_service.generate_sql_query(request.query, schema_info)

    # Execute the query
    result = db_manager.execute_query(sql_query)

    # Generate analysis summary
    analysis_summary = ai_service.generate_analysis_summary(request.query, result)

    return {"analysis_summary": analysis_summary}

@app.post("/production_issue_trend")
async def production_issue_trend(request: ProductionIssueRequest):
    global ai_service, schema_info

    # Generate SQL query from natural language
    sql_query = ai_service.generate_sql_query(request.query, schema_info)

    # Execute the query
    result = db_manager.execute_query(sql_query)

    # Convert result to DataFrame
    df = pd.DataFrame(result, columns=[col[0] for col in db_manager.get_column_names(sql_query)])

    # Ask AI to determine appropriate x and y axes
    axes_selection = ai_service.determine_trend_axes(request.query, df.columns.tolist())

    # Generate trend graph
    plt.figure(figsize=(10, 6))
    plt.plot(df[axes_selection['x']], df[axes_selection['y']])
    plt.title(request.query)
    plt.xlabel(axes_selection['x'])
    plt.ylabel(axes_selection['y'])
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the bytes buffer to base64
    graph_base64 = base64.b64encode(buf.getvalue()).decode()

    return {"graph": graph_base64, "x_axis": axes_selection['x'], "y_axis": axes_selection['y']}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    embedding_cache.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)