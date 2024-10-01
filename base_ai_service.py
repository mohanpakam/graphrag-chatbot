from abc import ABC, abstractmethod
from typing import List

class AIService(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def generate_response(self, prompt: str, context: str) -> str:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

    @abstractmethod
    def clear_conversation_history(self):
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        pass