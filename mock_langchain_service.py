from typing import List, Dict, Any
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from base_langchain_service import BaseLangChainAIService

class MockLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        return f"Mocked response for: {prompt[:30]}..."

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def _identifying_params(self):
        return {"name": "MockLLM"}

class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

class MockNode:
    def __init__(self, id: str, type: str, properties: Dict[str, Any]):
        self.id = id
        self.type = type
        self.properties = properties

class MockRelationship:
    def __init__(self, source: MockNode, target: MockNode, type: str):
        self.source = source
        self.target = target
        self.type = type

class MockGraph:
    def __init__(self):
        self.nodes = []
        self.relationships = []
        self.source = None

    def add_node(self, node: MockNode):
        self.nodes.append(node)

    def add_relationship(self, relationship: MockRelationship):
        self.relationships.append(relationship)

class MockLangChainAIService(BaseLangChainAIService):
    def __init__(self):
        super().__init__(MockLLM(), MockEmbeddings())
        self.embedding_dim = 3

    def get_embedding(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def generate_response(self, prompt: str, context: str) -> str:
        return self.llm(f"Context: {context}\n\nHuman: {prompt}")

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def clear_conversation_history(self):
        pass

    def convert_to_graph_documents(self, chunks: List[Document]) -> List[MockGraph]:
        graph_documents = []
        for chunk in chunks:
            graph = MockGraph()
            node1 = MockNode(id="1", type="Entity", properties={"name": "Entity1"})
            node2 = MockNode(id="2", type="Entity", properties={"name": "Entity2"})
            rel = MockRelationship(source=node1, target=node2, type="RELATED_TO")
            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_relationship(rel)
            graph.source = chunk
            graph_documents.append(graph)
        return graph_documents