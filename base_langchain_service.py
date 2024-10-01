from typing import List
from langchain.callbacks.manager import CallbackManager
from langchain.schema import Document

class BaseLangChainAIService:
    def __init__(self, llm, embeddings, callback_manager: CallbackManager = None):
        self.llm = llm
        self.embeddings = embeddings
        self.embedding_dim = None
        self.callback_manager = callback_manager

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.embeddings.embed_query(text)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        return embedding

    def generate_response(self, prompt: str, context: str) -> str:
        full_prompt = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"
        return self.llm.invoke(full_prompt)  # Changed from predict to invoke

    def get_embedding_dim(self) -> int:
        if self.embedding_dim is None:
            # Generate a dummy embedding to get the dimension
            self.get_embedding("dummy text")
        return self.embedding_dim

    def clear_conversation_history(self):
        # If the LLM has a conversation history, clear it here
        # This is a placeholder and might need to be implemented in subclasses
        pass

    def convert_to_graph_documents(self, chunks: List[Document]) -> List[dict]:
        graph_documents = []
        for i, chunk in enumerate(chunks):
            graph_documents.append({
                'filename': chunk.metadata.get('filename', f'chunk_{i}'),
                'chunk_index': i,
                'content': chunk.page_content,
                'embedding': self.get_embedding(chunk.page_content),
                'metadata': chunk.metadata
            })
        return graph_documents

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(documents)