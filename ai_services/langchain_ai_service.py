from typing import List

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from ai_services import BaseLangChainAIService
from config import LoggerConfig
from langchain.callbacks.manager import CallbackManager
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import os

config = LoggerConfig.load_config()
logger = LoggerConfig.setup_logger(__name__)

class LangChainAIService(BaseLangChainAIService):
    def __init__(self, llm, embeddings, callback_manager=None):
        super().__init__(llm, embeddings, callback_manager)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            callback_manager=callback_manager
        )
        logger.info(f"Initialized {self.__class__.__name__}")

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
        
        logger.info(f"LangChain generated response: {response}")
        return response

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension, generating a dummy embedding if necessary."""
        if self.embedding_dim is None:
            logger.debug("Generating dummy embedding to get dimension")
            self.get_embedding("dummy text")
        return self.embedding_dim

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        logger.info("LangChain conversation history has been cleared.")

    def generate_sql_query(self, natural_language_query, schema_info):
        prompt = f"""
        Given the following database schema:
        {schema_info}

        Generate an SQL query for the following question:
        {natural_language_query}

        SQL Query:
        """
        response = self.llm(prompt)
        return response.strip()

    def generate_analysis_summary(self, query, result):
        prompt = f"""
        Given the following query and its result:
        Query: {query}
        Result: {result}

        Provide a brief analysis summary of the data:
        """
        response = self.llm(prompt)
        return response.strip()

    def determine_trend_axes(self, query: str, available_columns: List[str]) -> dict[str, str]:
        prompt = f"""
        Given the following query and available columns:
        Query: {query}
        Available columns: {', '.join(available_columns)}

        Determine the most appropriate columns for the x and y axes of a trend graph.
        Return your answer in the format:
        x: <column_name>
        y: <column_name>
        """
        response = self.llm(prompt)
        axes = {}
        for line in response.strip().split('\n'):
            axis, column = line.split(': ')
            axes[axis.strip()] = column.strip()
        return axes

class AzureOpenAILangChainAIService(BaseLangChainAIService):
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
        embeddings = AzureOpenAIEmbeddings(
            deployment=config['azure_openai_embedding_deployment'],
            model="text-embedding-ada-002",
            openai_api_key=config['azure_openai_api_key'],
            azure_endpoint=config['azure_openai_endpoint'],
            api_version="2023-05-15"
        )
        super().__init__(llm, embeddings)

class OllamaLangChainAIService(BaseLangChainAIService):
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

class VertexGeminiLangChainAIService(BaseLangChainAIService):
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

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        logger.info("Vertex Gemini conversation history has been cleared.")

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