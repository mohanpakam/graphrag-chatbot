import logging
from typing import List, Dict
from langchain.schema import Document
from llm_graph_transformer import LLMGraphTransformer
import numpy as np
from ai_service import get_ai_service

class GraphManager:
    def __init__(self, db_path: str):
        self.graph_transformer = LLMGraphTransformer(db_path)
        self.graph_transformer.init_database()
        self.ai_service = get_ai_service()

    def store_data(self, chunk: Document, embedding: List[float]):
        embedding_array = np.array(embedding, dtype=np.float32)
        
        doc_id = self.graph_transformer.store_data({
            'filename': chunk.metadata.get('filename', 'unknown'),
            'chunk_index': chunk.metadata.get('chunk_index', 0),
            'content': chunk.page_content,
            'embedding': embedding_array,
            'metadata': chunk.metadata
        })
        
        # Store the doc_id in the chunk's metadata for future reference
        chunk.metadata['graph_id'] = doc_id

    def get_subgraph_for_documents(self, documents: List[Document]) -> Dict:
        subgraph = {}
        for doc in documents:
            doc_id = doc.metadata.get('sqlite_id')
            if doc_id:
                doc_data = self.graph_transformer.get_document_data(doc_id)
                if doc_data:
                    subgraph[doc_id] = {
                        'document': doc_data,
                        'relationships': self.graph_transformer.get_relationships(doc_id)
                    }
        return subgraph

    def get_embedding(self, text: str) -> List[float]:
        return self.ai_service.get_embedding(text)