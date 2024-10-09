from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from utils.database_manager import DatabaseManager
from utils.embedding_cache import EmbeddingCache
from graphrag import GraphRAG
from structured.data_storage import StructuredDataStorage
from ai_services import get_langchain_ai_service
from services import ProductionIssueService
from utils.query_classifier import QueryClassifier
import time
import uvicorn
from config import LoggerConfig

app = FastAPI()

config = LoggerConfig.load_config()
db_manager = DatabaseManager(config['database_path'])
embedding_cache = EmbeddingCache(config['faiss_index_file'], config['database_path'])
graph_rag = GraphRAG()
structured_data_storage = StructuredDataStorage(config['database_path'])
production_issue_service = ProductionIssueService()
query_classifier = QueryClassifier(graph_rag.get_ai_service())

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

class ChatRequest(BaseModel):
    message: str
    reset: bool = False

class ProductionIssueRequest(BaseModel):
    query: str

ai_service = None

@app.on_event("startup")
async def startup_event():
    global ai_service
    logger.info("Starting up the application")
    ai_service = get_langchain_ai_service(config['ai_service'])
    logger.info("AI service initialized")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.reset:
        graph_rag.reset_conversation()
        logger.info("Chat history has been reset")
        return {"response": "Chat history has been cleared."}

    response = graph_rag.process_query(request.message)
    return {"response": response}

@app.post("/production_issue_query")
async def production_issue_query(request: ProductionIssueRequest):
    #query_type = query_classifier.classify_query(request.query)
    query_type ='specific_issue'
    logger.info(f"Query type: {query_type}")

    if query_type == "specific_issue":
        return await specific_issue_analysis(request.query)
    elif query_type == "trend_analysis":
        return await trend_analysis(request.query)
    else:  # general_question
        return await general_question_analysis(request.query)

async def specific_issue_analysis(query: str):
    try:
        analysis_result = production_issue_service.analyze_production_issue(query)
        return {
            "query_type": "specific_issue",
            "sql_query": query,
            "analysis_summary": analysis_result["analysis_summary"],
            "extracted_data": {
                "nodes": analysis_result["extracted_data"]["nodes"],
                "relationships": analysis_result["extracted_data"]["relationships"],
                "keywords": analysis_result["extracted_data"]["keywords"]
            },
            "similar_issues": [
                {
                    "id": issue["id"],
                    "summary": issue["summary"] if "summary" in issue else None
                } for issue in analysis_result["similar_issues"]
            ],
            "query_results_count": len(analysis_result["query_results"])
        }
    except Exception as e:
        logger.error(f"Error in specific issue analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in specific issue analysis")

async def trend_analysis(query: str):
    try:
        trend_result = production_issue_service.analyze_trend(query)
        return {
            "query_type": "trend_analysis",
            "query": query,
            "trend_summary": trend_result["trend_summary"],
            "trend_data": trend_result["trend_data"],
            "trend_axes": trend_result["trend_axes"]
        }
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in trend analysis")

async def general_question_analysis(query: str):
    try:
        general_result = production_issue_service.analyze_general_question(query)
        return {
            "query_type": "general_question",
            "query": query,
            "analysis_summary": general_result["analysis_summary"],
            "sql_query": general_result["sql_query"],
            "query_results": general_result["query_results"]
        }
    except Exception as e:
        logger.error(f"Error in general question analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in general question analysis")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    embedding_cache.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)