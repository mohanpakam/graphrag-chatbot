from typing import List, Dict, Any, TypedDict
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from vector_store_manager import VectorStoreManager
from memory_manager import MemoryManager
from graph_manager import GraphManager
from langchain_ai_service import get_langchain_ai_service
from embedding_cache import EmbeddingCache
from query_classifier import QueryClassifier
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
        self.embedding_cache = EmbeddingCache(config['faiss_index_file'], config['database_path'])
        self.query_classifier = QueryClassifier(self.ai_service)
        self.workflow = self._create_workflow()
        self.conversation_context = []
        self.current_issue_context = None

    def _create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def retrieve_context_and_graph(self, query: str, query_type: str) -> Dict[str, Any]:
        query_embedding = self.ai_service.get_embedding(query)
        
        if query_type == "specific_issue" or self.current_issue_context is None:
            similar_chunks = self.embedding_cache.find_similar_chunks(query_embedding, config.get('top_k', 5))
            
            logger.info(f"Retrieved {len(similar_chunks)} similar chunks from embeddings")
            for chunk in similar_chunks:
                logger.info(f"Chunk from file: {chunk['filename']}, chunk #{chunk['chunk_index']}, distance: {chunk['distance']:.4f}")
            
            if similar_chunks:
                most_similar_doc_id = similar_chunks[0]['document_id']
                all_doc_chunks = self.embedding_cache.get_all_chunks_for_document(most_similar_doc_id)
                logger.info(f"Retrieved all {len(all_doc_chunks)} chunks for document ID {most_similar_doc_id}")
                
                graph_data = self.graph_manager.get_subgraph_for_documents([most_similar_doc_id])
                logger.info(f"Retrieved graph data: {len(graph_data['nodes'])} nodes and {len(graph_data['relationships'])} relationships")
                
                return {
                    "chunks": all_doc_chunks,
                    "graph_data": graph_data
                }
        else:
            # For follow-up questions, use the context from the current issue
            logger.info("Using existing context for follow-up question")
            return self.current_issue_context

        # For broader queries or when no specific context is set
        doc_ids = list(set(chunk['document_id'] for chunk in similar_chunks))
        graph_data = self.graph_manager.get_subgraph_for_documents(doc_ids)
        expanded_graph = self.expand_graph_for_broad_query(graph_data)
        logger.info(f"Retrieved expanded graph data: {len(expanded_graph['nodes'])} nodes and {len(expanded_graph['relationships'])} relationships")
        return {
            "chunks": similar_chunks,
            "graph_data": expanded_graph
        }

    def expand_graph_for_broad_query(self, initial_graph: Dict) -> Dict:
        # Implement logic to expand the graph based on relationships
        # This could involve traversing the graph to find related nodes
        # For simplicity, let's assume we're just returning the initial graph for now
        return initial_graph

    def retrieve_context(self, state: State) -> State:
        query = state['query']
        result = self.retrieve_context_and_graph(query)
        logger.debug(f'Context and Graph Generated are {result}')
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
        
        logger.debug(f'Formatted Prompt is  {formatted_prompt}')
        
        response = self.ai_service.generate_response(formatted_prompt, "")
        
        state['response'] = response
        state['chat_history'] = self.memory_manager.add_interaction(query, response)
        return state

    def process_query(self, query: str):
        query_type = self.query_classifier.classify_query(query, self.conversation_context)
        logger.info(f"Query classified as: {query_type}")
        
        # Add the current query to the conversation context
        self.conversation_context.append(query)
        self.conversation_context = self.conversation_context[-5:]  # Keep last 5 exchanges
        
        result = self.retrieve_context_and_graph(query, query_type)
        chunks = result["chunks"]
        graph_data = result["graph_data"]
        
        context = self.combine_context(chunks, graph_data, query_type)
        
        if query_type == "specific_issue":
            response = self.process_specific_issue_query(query, context, graph_data)
            self.current_issue_context = {"chunks": chunks, "graph_data": graph_data}
        elif query_type == "trend_analysis":
            response = self.process_trend_analysis_query(query, context, graph_data)
        else:
            response = self.process_general_question(query, context, graph_data)
        
        return response

    def combine_context(self, chunks, graph_data, query_type):
        context = "Relevant information:\n"
        for chunk in chunks:
            context += f"- From {chunk['filename']} (Chunk {chunk['chunk_index']}): {chunk['content'][:200]}...\n"
        
        context += "\nRelevant entities and relationships:\n"
        for node in graph_data['nodes']:
            context += f"- {node['type']}: {node['properties']}\n"
        for rel in graph_data['relationships']:
            context += f"- {rel['source']} {rel['type']} {rel['target']}\n"
        
        if query_type == "trend_analysis":
            context += "\nTrend Analysis Context:\n"
            # Add any specific trend-related information here
        
        logger.info(f"Combined context: {len(context.split())} words, {len(chunks)} chunks, {len(graph_data['nodes'])} nodes, {len(graph_data['relationships'])} relationships")
        return context

    def process_specific_issue_query(self, query, context, graph_data):
        prompt = f"""
        Given the following specific issue query and context, provide a detailed response:
        Query: {query}
        
        Context:
        {context}
        
        Focus on the specific issue mentioned in the query and use the provided context and graph data to give a comprehensive answer.
        If the query is a follow-up question, relate it to the previously discussed issue.
        """
        logger.info(f"Processing specific issue query with {len(context.split())} words of context")
        return self.ai_service.generate_response(prompt, "")

    def process_trend_analysis_query(self, query, context, graph_data):
        prompt = f"""
        Analyze the following trend-related query using the provided context and graph data:
        Query: {query}
        
        Context:
        {context}
        
        Identify any trends, patterns, or recurring issues in the data. Provide statistical insights if possible.
        """
        logger.info(f"Processing trend analysis query with {len(context.split())} words of context")
        return self.ai_service.generate_response(prompt, "")

    def process_general_question(self, query, context, graph_data):
        prompt = f"""
        Answer the following general question about production issues using the provided context and graph data:
        Query: {query}
        
        Context:
        {context}
        
        Provide a comprehensive answer based on the available information.
        If this question relates to a previously discussed specific issue, make connections where relevant.
        """
        logger.info(f"Processing general question with {len(context.split())} words of context")
        return self.ai_service.generate_response(prompt, "")

    def reset_conversation(self):
        self.conversation_context = []
        self.current_issue_context = None
        logger.info("Conversation context and current issue context have been reset")

# ... (rest of the code remains the same)