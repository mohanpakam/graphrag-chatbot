from typing import List, Dict, Any, TypedDict
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from vector_store_manager import VectorStoreManager
from memory_manager import MemoryManager
from graph_manager import GraphManager
from langchain_ai_service import get_langchain_ai_service
from embedding_cache import EmbeddingCache
import yaml
import logging

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
logger = logging.getLogger(__name__)

class State(TypedDict):
    query: str
    context: List[Document]
    graph_data: Dict
    chat_history: List[Dict[str, str]]
    response: str

class GraphRAG:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.ai_service = get_langchain_ai_service(config['ai_service'])
        self.memory_manager = MemoryManager()
        self.graph_manager = GraphManager(config['database_path'])
        self.embedding_cache = EmbeddingCache(config['database_path'])
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def retrieve_context_and_graph(self, query: str) -> Dict[str, Any]:
        # Generate embedding for the query
        query_embedding = self.ai_service.get_embedding(query)
        
        logger.debug(f"query embeddings size {len(query_embedding)}")
        
        # Use the embedding cache to find similar documents
        similar_docs = self.embedding_cache.find_similar(query_embedding, config.get('top_k', 5))
        
        logger.debug(f"similar_docs size {len(similar_docs)} and similar docs are {similar_docs}")
        
        # Retrieve full documents and graph data
        relevant_docs = self.vector_store_manager.get_documents_by_ids([doc_id for _, doc_id in similar_docs])
        logger.debug(f"Relevant docs are {len(relevant_docs)} and relevant_docs are {relevant_docs}")
        graph_data = self.graph_manager.get_subgraph_for_documents(relevant_docs)
        
        return {
            "context": relevant_docs,
            "graph_data": graph_data
        }

    def retrieve_context(self, state: State) -> State:
        query = state['query']
        result = self.retrieve_context_and_graph(query)
        logging.debug(f'Context and Graph Generated are {result}')
        state['context'] = result['context']
        state['graph_data'] = result['graph_data']
        return state

    def generate_response(self, state: State) -> State:
        query = state['query']
        context = state['context']
        graph_data = state['graph_data']
        chat_history = state['chat_history']
        
        prompt = PromptTemplate.from_template(
            "Answer the following question based on the given context, graph data, and chat history:\n\n"
            "Context: {context}\n"
            "Graph Data: {graph_data}\n"
            "Chat History: {chat_history}\n"
            "Question: {question}\n"
            "Answer: "
        )
        
        formatted_prompt = prompt.format(
            context=context,
            graph_data=graph_data,
            chat_history=chat_history,
            question=query
        )
        
        logging.debug(f'Formatted Prompt is  {formatted_prompt}')
        
        response = self.ai_service.generate_response(formatted_prompt, "")
        
        state['response'] = response
        state['chat_history'] = self.memory_manager.add_interaction(query, response)
        return state

    def process_query(self, query: str):
        logging.info(f"Processing query: '{query}'")
        
        initial_state: State = {
            "query": query,
            "context": [],
            "graph_data": {},
            "chat_history": self.memory_manager.get_chat_history(),
            "response": ""
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        logging.info("Query processed successfully.")
        return final_state['response']

def subgraph_to_context(subgraph, similar_docs):
    context = "Relevant information:\n"
    for node in subgraph['nodes']:
        context += f"- {node[2]}: {node[1]}\n"
    for rel in subgraph['relationships']:
        context += f"- {rel[1]} {rel[3]} {rel[2]}\n"
    
    context += "\nRelevant document excerpts:\n"
    for doc in similar_docs:
        content = doc.page_content[:200]  # Truncate to first 200 characters
        context += f"- {content}...\n"
    
    return context