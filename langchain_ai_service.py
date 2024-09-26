from typing import List
from langchain_community.llms import AzureOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from base_ai_service import AIService
import yaml
from logger_config import LoggerConfig

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
logger = LoggerConfig.setup_logger(__name__)

class BaseLangChainAIService(AIService):
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.embedding_dim = None  # Will be set after first embedding
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.embeddings.embed_query(text)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        return embedding

    def generate_response(self, prompt: str, context: str) -> str:
        full_prompt = f"Context: {context}\n\nHuman: {prompt}"
        response = self.conversation.predict(input=full_prompt)
        
        logger.info(f"LangChain generated response: {response}")
        return response

    def get_embedding_dim(self) -> int:
        if self.embedding_dim is None:
            # Generate a dummy embedding to get the dimension
            self.get_embedding("dummy text")
        return self.embedding_dim

    def clear_conversation_history(self):
        self.memory.clear()
        logger.info("LangChain conversation history has been cleared.")

class AzureOpenAILangChainAIService(BaseLangChainAIService):
    def __init__(self):
        llm = AzureOpenAI(
            deployment_name=config['azure_openai_completion_deployment'],
            model_name="gpt-35-turbo",
            openai_api_key=config['azure_openai_api_key'],
            azure_endpoint=config['azure_openai_endpoint'],
            api_version="2023-05-15"
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
    def __init__(self):
        llm = Ollama(model=config['ollama_model'])
        embeddings = OllamaEmbeddings(model=config['ollama_model'])
        super().__init__(llm, embeddings)

class OpenAILangChainAIService(BaseLangChainAIService):
    def __init__(self):
        llm = OpenAI(api_key=config['openai_api_key'], temperature=0.7)
        embeddings = OpenAIEmbeddings(api_key=config['openai_api_key'])
        super().__init__(llm, embeddings)

def get_langchain_ai_service(service_type: str) -> BaseLangChainAIService:
    if service_type == 'azure_openai':
        return AzureOpenAILangChainAIService()
    elif service_type == 'ollama':
        return OllamaLangChainAIService()
    elif service_type == 'openai':
        return OpenAILangChainAIService()
    else:
        raise ValueError(f"Unknown LangChain AI service type: {service_type}")