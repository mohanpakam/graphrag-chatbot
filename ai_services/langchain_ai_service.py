from typing import List, Dict
from langchain.memory import ConversationBufferMemory
from ai_services import BaseLangChainAIService
from config import LoggerConfig
from langchain.callbacks.manager import CallbackManager
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings

from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from langchain_community.llms.ollama import Ollama
import os
import json
from structured.data_storage import StructuredDataStorage
from ai_services.sql_query_chain import SQLQueryChain
from ai_services.analysis_summary_chain import AnalysisSummaryChain
from ai_services.trend_axes_chain import TrendAxesChain
from ai_services.node_relationship_extraction import NodeRelationshipExtraction
from ai_services.response_generator import ResponseGenerator

config = LoggerConfig.load_config()
logger = LoggerConfig.setup_logger(__name__)

class LangChainAIService(BaseLangChainAIService):
    def __init__(self, llm, embeddings, callback_manager=None):
        super().__init__(llm, embeddings, callback_manager)
        self.memory = ConversationBufferMemory()
        self.data_storage = StructuredDataStorage(config['database_path'])
        self.sql_query_chain = SQLQueryChain(llm)
        self.analysis_summary_chain = AnalysisSummaryChain(llm)
        self.trend_axes_chain = TrendAxesChain(llm)
        self.node_relationship_extraction = NodeRelationshipExtraction(llm)
        self.response_generator = ResponseGenerator(llm, embeddings, callback_manager)
        logger.info(f"Initialized {self.__class__.__name__}")

    def generate_sql_query(self, natural_language_query: str) -> str:
        """Generate an SQL query based on the natural language query and schema information."""
        schema_info = self.data_storage.get_schema_info()
        sql_query_response = self.sql_query_chain.generate_sql_query(natural_language_query)
        
        # Verify the generated SQL query against the schema
        verified_query = self.verify_sql_query(sql_query_response, schema_info)
        return verified_query.strip()

    def verify_sql_query(self, sql_query: str, schema_info: str) -> str:
        """Verify the generated SQL query against the database schema."""
        # Implement logic to verify the SQL query against the schema information
        # and return the verified query
        return sql_query

    def generate_analysis_summary(self, query: str, sql_query_response: str) -> str:
        """Generate an analysis summary based on the query and SQL query response."""
        response = self.analysis_summary_chain.generate_analysis_summary(query, sql_query_response)
        return response.strip()

    def determine_trend_axes(self, query: str, available_columns: List[str]) -> Dict[str, str]:
        """Determine the appropriate axes for trend visualization."""
        return self.trend_axes_chain.determine_trend_axes(query, available_columns)

    def extract_nodes_and_relationships(self, text: str) -> Dict:
        """Extract nodes, relationships, keywords, and embedding text from the given text."""
        return self.node_relationship_extraction.extract_nodes_and_relationships(text)

    def generate_response(self, prompt: str, context: str) -> str:
        """Generate a response given a prompt and context."""
        return self.response_generator.generate_response(prompt, context)

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        self.response_generator.clear_conversation_history()
        logger.info("LangChain conversation history has been cleared.")

class AzureOpenAILangChainAIService(LangChainAIService):
    def __init__(self, callback_manager=None):
        logger.info("Initializing AzureOpenAILangChainAIService")
        llm = AzureOpenAI(
            deployment_name=config['azure_openai_completion_deployment'],
            model_name="gpt-35-turbo",
            openai_api_key=config['azure_openai_api_key'],
            azure_endpoint=config['azure_openai_endpoint'],
            api_version="2023-05-15",
            callback_manager=callback_manager
        )
        super().__init__(llm, None)


class OllamaLangChainAIService(LangChainAIService):
    def __init__(self, callback_manager=None):
        logger.info("Initializing OllamaLangChainAIService")
        llm = Ollama(model=config['ollama_model'])
        embeddings = OllamaEmbeddings(model=config['ollama_model'])
        super().__init__(llm, embeddings)

class OpenAILangChainAIService(BaseLangChainAIService):
    def __init__(self, callback_manager=None):
        logger.info("Initializing OpenAILangChainAIService")
        llm = OpenAI(api_key=config['openai_api_key'], temperature=0.7)
        embeddings = OpenAIEmbeddings(api_key=config['openai_api_key'])
        super().__init__(llm, embeddings)

class VertexGeminiLangChainAIService(LangChainAIService):
    def __init__(self, callback_manager=None):
        logger.info("Initializing VertexGeminiLangChainAIService")
        
        # Use the token from environment variable
        token = os.environ.get('VERTEX_AI_TOKEN')
        if not token:
            raise ValueError("Vertex AI token not found. Please provide a valid token.")
        
        # Create credentials using the token
        credentials = Credentials(token=token)
        
        # Refresh the token if it's expired
        if credentials.expired:
            credentials.refresh(Request())
        
        # Initialize Vertex AI LLM
        llm = VertexAI(
            model_name="gemini-pro",
            project=config['vertex_ai_project'],
            location=config['vertex_ai_location'],
            credentials=credentials,
            max_output_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
        )
        
        # Initialize Vertex AI Embeddings
        embeddings = VertexAIEmbeddings(
            model_name="textembedding-gecko",
            project=config['vertex_ai_project'],
            location=config['vertex_ai_location'],
            credentials=credentials,
        )
        
        super().__init__(llm, embeddings, callback_manager)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            callback_manager=callback_manager
        )

    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        embedding = self.embeddings.embed_query(text)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            logger.info(f"Set embedding dimension to {self.embedding_dim}")
        return embedding

    def generate_response(self, prompt: str, context: str) -> str:
        """Generate a response given a prompt and context."""
        full_prompt = f"Context: {context}\n\nHuman: {prompt}"
        logger.debug(f"Generating response for prompt: {full_prompt}")
        response = self.conversation.predict(input=full_prompt)
        
        logger.info(f"Vertex Gemini generated response: {response}")
        return response

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension, generating a dummy embedding if necessary."""
        if self.embedding_dim is None:
            logger.debug("Generating dummy embedding to get dimension")
            self.get_embedding("dummy text")
        return self.embedding_dim
def get_langchain_ai_service(service_type: str, callback_manager: CallbackManager = None) -> BaseLangChainAIService:
    """Get the appropriate LangChain AI service based on the service type."""
    logger.info(f"Getting LangChain AI service for type: {service_type}")
    if service_type == 'azure_openai':
        return AzureOpenAILangChainAIService(callback_manager)
    elif service_type == 'ollama':
        return OllamaLangChainAIService(callback_manager)
    elif service_type == 'openai':
        return OpenAILangChainAIService(callback_manager)
    elif service_type == 'vertex_gemini':
        return VertexGeminiLangChainAIService(callback_manager)
    else:
        error_msg = f"Unknown LangChain AI service type: {service_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)