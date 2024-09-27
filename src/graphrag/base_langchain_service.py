from typing import List
from langchain.callbacks.manager import CallbackManager

class BaseLangChainAIService:
    def __init__(self, llm, embeddings, callback_manager: CallbackManager = None):
        self.llm = llm
        self.embeddings = embeddings
        self.embedding_dim = None
        self.callback_manager = callback_manager

    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    def generate_response(self, prompt: str, context: str) -> str:
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        raise NotImplementedError

    def clear_conversation_history(self):
        raise NotImplementedError

    def convert_to_graph_documents(self, chunks):
        raise NotImplementedError