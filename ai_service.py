import requests
from typing import List
import numpy as np
from openai import AzureOpenAI
import ollama
import yaml
import time
from logger_config import LoggerConfig
from base_ai_service import AIService
from langchain_ai_service import get_langchain_ai_service
from langchain_community.embeddings import OllamaEmbeddings

def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

config = load_config()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

class OllamaService(AIService):
    def __init__(self, model: str = None):
        self.model = model or config.get('ollama_model', 'llama3.1')
        self.embeddings = OllamaEmbeddings(model=self.model)

    def get_embedding(self, text: str) -> List[float]:
        start_time = time.time()
        embedding = self.embeddings.embed_query(text)
        duration = time.time() - start_time
        logger.info(f"Ollama embedding generation took {duration:.2f} seconds")
        return embedding

    def generate_response(self, prompt: str, context: str) -> str:
        start_time = time.time()
        full_prompt = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"
        logger.info(f"Request sent to Ollama Generate API: {full_prompt}")
        response = ollama.generate(model=self.model, prompt=full_prompt)
        logger.info(f"Response received from Ollama Generate API: {response['response']}")
        duration = time.time() - start_time
        logger.info(f"Ollama response generation took {duration:.2f} seconds")
        return response['response'].strip()

    def get_embedding_dim(self) -> int:
        # You might need to have a mapping of Ollama models to their embedding dimensions
        # or generate a dummy embedding to get the dimension
        dummy_embedding = self.get_embedding("dummy text")
        return len(dummy_embedding)

    def clear_conversation_history(self):
        # Ollama doesn't maintain conversation history by default
        pass

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(documents)

class AzureOpenAIService(AIService):
    def __init__(self, api_key: str = None, endpoint: str = None, 
                 embedding_deployment: str = None, completion_deployment: str = None):
        self.api_key = api_key or config.get('azure_openai_api_key')
        self.endpoint = endpoint or config.get('azure_openai_endpoint')
        self.embedding_deployment = embedding_deployment or config.get('azure_openai_embedding_deployment')
        self.completion_deployment = completion_deployment or config.get('azure_openai_completion_deployment')
        
        if not all([self.api_key, self.endpoint, self.embedding_deployment, self.completion_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration.")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2023-05-15",
            azure_endpoint=self.endpoint
        )

    def get_embedding(self, text: str) -> List[float]:
        start_time = time.time()
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        duration = time.time() - start_time
        logger.info(f"Azure OpenAI embedding generation took {duration:.2f} seconds")
        return response.data[0].embedding

    def generate_response(self, prompt: str, context: str) -> str:
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nHuman: {prompt}"}
        ]
        logger.info(f"Request sent to Azure OpenAI Generate API: {messages}")
        response = self.client.chat.completions.create(
            model=self.completion_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        response_content = response.choices[0].message.content.strip()
        logger.info(f"Response received from Azure OpenAI Generate API: {response_content}")
        duration = time.time() - start_time
        logger.info(f"Azure OpenAI response generation took {duration:.2f} seconds")
        return response_content

    def get_embedding_dim(self) -> int:
        # You might need to have a mapping of Azure OpenAI models to their embedding dimensions
        azure_embedding_dims = {
            "text-embedding-ada-002": 1536,
            # Add other models as needed
        }
        return azure_embedding_dims.get(self.embedding_deployment, 1536)  # Default to 1536 if unknown

    def clear_conversation_history(self):
        # Azure OpenAI doesn't maintain conversation history by default
        pass

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=documents,
            model=self.embedding_deployment
        )
        return [item.embedding for item in response.data]

def get_ai_service() -> AIService:
    service_type = config.get('ai_service', 'ollama')
    use_langchain = config.get('use_langchain', False)
    
    if use_langchain:
        return get_langchain_ai_service(service_type)
    elif service_type == 'ollama':
        return OllamaService()
    elif service_type == 'azure_openai':
        return AzureOpenAIService()
    else:
        raise ValueError(f"Unknown AI service: {service_type}")