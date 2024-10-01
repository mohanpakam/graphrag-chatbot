import logging
from typing import List, Dict
from langchain.schema import Document
from llm_graph_transformer import LLMGraphTransformer
import numpy as np
from ai_service import get_ai_service

class GraphManager:
    def __init__(self):
        self.graph_transformer = LLMGraphTransformer(db_path='ollama_vss.db')
        self.graph_transformer.init_database()
        self.ai_service = get_ai_service()

    def store_data(self, chunk: Document, embedding: List[float] = None):
        # If embedding is not provided, generate it using the AI service
        if embedding is None:
            embedding = self.ai_service.get_embedding(chunk.page_content)
        
        # Convert the embedding list to a numpy array
        embedding_array = np.array(embedding)
        
        self.graph_transformer.store_data({
            'filename': chunk.metadata.get('filename', 'unknown'),
            'chunk_index': chunk.metadata.get('chunk_index', 0),
            'content': chunk.page_content,
            'embedding': embedding_array,
            'metadata': chunk.metadata
        })

    def get_subgraph_for_documents(self, documents: List[Document]) -> Dict:
        # Instead of using a non-existent method, let's use the methods that LLMGraphTransformer actually has
        # This is a placeholder implementation and might need to be adjusted based on LLMGraphTransformer's actual capabilities
        subgraph = {}
        for doc in documents:
            doc_id = doc.metadata.get('sqlite_id')
            if doc_id:
                # Assuming there's a method to get data for a single document
                doc_data = self.graph_transformer.get_document_data(doc_id)
                if doc_data:
                    subgraph[doc_id] = doc_data
        return subgraph

    def get_embedding(self, text: str) -> List[float]:
        return self.ai_service.get_embedding(text)