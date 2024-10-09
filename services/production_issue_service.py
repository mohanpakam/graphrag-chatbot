import json
from utils.database_manager import DatabaseManager
from ai_services import get_langchain_ai_service
from config import LoggerConfig
from typing import List, Dict
import faiss
import numpy as np
from structured.data_storage import StructuredDataStorage

logger = LoggerConfig.setup_logger(__name__)
config = LoggerConfig.load_config()

class ProductionIssueService:
    def __init__(self):
        self.db_manager = DatabaseManager(config['database_path'])
        self.ai_service = get_langchain_ai_service(config['ai_service'])
        self.data_storage = StructuredDataStorage(config['database_path'])
        self.schema_info = self.data_storage.get_schema_info()
        self.faiss_index = faiss.read_index(config['faiss_index_file'])

    def analyze_production_issue(self, query: str) -> Dict:
        # Generate SQL query using LLM
        sql_query = self.ai_service.generate_sql_query(query)
        logger.info(f"Generated SQL query: {sql_query}")

        # Execute the SQL query
        try:
            query_results = self.db_manager.execute_query(sql_query)
            column_names = self.db_manager.get_column_names(sql_query)
            formatted_results = [dict(zip(column_names, row)) for row in query_results]
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise Exception("Error executing SQL query")

        # Generate analysis summary using LLM
        analysis_summary = self.ai_service.generate_analysis_summary(query, json.dumps(formatted_results, indent=2))

        # Determine trend axes
        trend_axes = self.ai_service.determine_trend_axes(query, column_names)

        # Extract nodes and relationships from the query results
        extracted_data = self.extract_nodes_and_relationships(formatted_results)

        # Perform similarity search using FAISS
        #similar_issues = self.find_similar_issues(query, k=5)

        return {
            "analysis_summary": analysis_summary,
            "query_results": formatted_results,
            "trend_axes": trend_axes,
            "extracted_data": extracted_data
          #  "similar_issues": similar_issues
        }

    def extract_nodes_and_relationships(self, results: List[Dict]) -> Dict:
        extracted_data = {
            "nodes": [],
            "relationships": [],
            "keywords": set()
        }
        for result in results:
            text = json.dumps(result)
            data = self.ai_service.extract_nodes_and_relationships(text)
            extracted_data["nodes"].extend(data["nodes"])
            extracted_data["relationships"].extend(data["relationships"])
            extracted_data["keywords"].update(data["keywords"])

        # Remove duplicates
        extracted_data["nodes"] = list({node["id"]: node for node in extracted_data["nodes"]}.values())
        extracted_data["relationships"] = list({json.dumps(rel): rel for rel in extracted_data["relationships"]}.values())
        extracted_data["keywords"] = list(extracted_data["keywords"])

        return extracted_data

    def find_similar_issues(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self.ai_service.get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')

        _, indices = self.faiss_index.search(query_embedding, k)
        similar_issues = []

        for idx in indices[0]:
            issue_data = self.db_manager.get_document_by_chunk_id(int(idx))
            if issue_data:
                similar_issues.append(issue_data)

        return similar_issues