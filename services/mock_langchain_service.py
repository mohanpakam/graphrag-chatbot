import random
from typing import List, Dict, Any
from services.base_langchain_service import BaseLangChainAIService
from langchain.schema import Document
from logger_config import LoggerConfig
import spacy
import networkx as nx
import numpy as np

logger = LoggerConfig.setup_logger(__name__)

# Load configuration
import yaml
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

class MockLLM:
    def invoke(self, prompt: str) -> str:
        return f"Mock response to: {prompt}"

class SpacyEmbeddings:
    def __init__(self, nlp):
        self.nlp = nlp

    def embed_query(self, text: str) -> List[float]:
        doc = self.nlp(text)
        return doc.vector.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.embed_query(doc) for doc in documents]

class MockLangChainAIService(BaseLangChainAIService):
    def __init__(self, callback_manager=None):
        logger.info("Initializing MockLangChainAIService")
        llm = MockLLM()
        self.nlp = spacy.load("en_core_web_sm")
        embeddings = SpacyEmbeddings(self.nlp)
        super().__init__(llm, embeddings, callback_manager)
        self.embedding_dim = self.nlp.vocab.vectors_length
        self.max_relationships = config.get('graph_transformer', {}).get('max_relationships', 5)

    def generate_response(self, prompt: str, context: str) -> str:
        full_prompt = f"Context: {context}\n\nHuman: {prompt}"
        response = f"Mock response to: {full_prompt}"
        logger.info(f"Mock generated response: {response}")
        return response

    def convert_to_graph_documents(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        graph_documents = []
        for i, chunk in enumerate(chunks):
            doc = self.nlp(chunk.page_content)
            graph_document = {
                'filename': chunk.metadata.get('filename', f'chunk_{i}'),
                'chunk_index': i,
                'content': chunk.page_content,
                'embedding': self.embeddings.embed_query(chunk.page_content),
                'metadata': chunk.metadata,
                'nodes': self._generate_nodes(doc),
                'relationships': self._generate_relationships(doc)
            }
            graph_documents.append(graph_document)
        return graph_documents

    def _generate_nodes(self, doc) -> List[Dict[str, Any]]:
        nodes = []
        for ent in doc.ents:
            nodes.append({
                'id': f'node_{len(nodes)}',
                'type': ent.label_,
                'properties': {'name': ent.text, 'start': ent.start_char, 'end': ent.end_char}
            })
        return nodes

    def _generate_relationships(self, doc) -> List[Dict[str, Any]]:
        relationships = []
        graph = nx.Graph()
        
        for token in doc:
            if token.dep_ != 'punct':
                graph.add_edge(token.head.text, token.text, type=token.dep_)
        
        for edge in list(graph.edges(data=True))[:self.max_relationships]:
            relationships.append({
                'source': f'node_{hash(edge[0]) % 1000}',  # Use hash to create consistent node IDs
                'target': f'node_{hash(edge[1]) % 1000}',
                'type': edge[2]['type'],
                'properties': {}
            })
        
        return relationships

    def clear_conversation_history(self):
        logger.info("Mock conversation history cleared (no-op in mock service).")