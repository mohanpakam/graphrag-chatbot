from typing import List
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from services.base_langchain_service import BaseLangChainAIService
from utils.logger_config import LoggerConfig

logger = LoggerConfig.setup_logger(__name__)

class LangChainAIService(BaseLangChainAIService):
    # ... (existing LangChainAIService code) ...

    def get_supported_services(self) -> List[str]:
        return [
            "azure_openai",
            "ollama",
            "openai",
            "vertex_gemini",
            "mock",
        ]

    def get_service(self, service_type: str) -> BaseLangChainAIService:
        if service_type == "azure_openai":
            return AzureOpenAILangChainAIService()
        elif service_type == "ollama":
            return OllamaLangChainAIService()
        elif service_type == "openai":
            return OpenAILangChainAIService()
        elif service_type == "vertex_gemini":
            return VertexGeminiLangChainAIService()
        elif service_type == "mock":
            return MockLangChainAIService()
        else:
            raise ValueError(f"Unsupported service type: {service_type}")