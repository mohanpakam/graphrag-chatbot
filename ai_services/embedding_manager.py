from typing import List
from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings, OllamaEmbeddings

class EmbeddingManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embedding_dim = None

    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        embedding = self.embeddings.embed_query(text)
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
        return embedding

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension, generating a dummy embedding if necessary."""
        if self.embedding_dim is None:
            self.get_embedding("dummy text")
        return self.embedding_dim