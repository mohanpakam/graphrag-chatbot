from langchain_community.llms import AzureOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from services.base_langchain_service import BaseLangChainAIService
from utils.logger_config import LoggerConfig
from config import config

logger = LoggerConfig.setup_logger(__name__)

class AzureOpenAILangChainAIService(BaseLangChainAIService):
    # ... (existing AzureOpenAILangChainAIService code) ...