import logging
from typing import List, Dict, Any, TypedDict
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from text_processor import TextProcessor
from vector_store_manager import VectorStoreManager
from memory_manager import MemoryManager
from graph_manager import GraphManager
from langchain_ai_service import get_langchain_ai_service
from base_langchain_service import BaseLangChainAIService
import yaml

# Load config
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the state schema
class State(TypedDict):
    query: str
    context: List[Document]
    graph_data: Dict
    chat_history: List[Dict[str, str]]
    response: str

class LocalLlamaProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.ai_service: BaseLangChainAIService = get_langchain_ai_service(config.get('ai_service', 'ollama'))
        self.memory_manager = MemoryManager()
        self.graph_manager = GraphManager()
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def retrieve_context(self, state: State) -> State:
        query = state['query']
        context = self.vector_store_manager.retrieve_relevant_documents(query)
        graph_data = self.graph_manager.get_subgraph_for_documents(context)
        state['context'] = context
        state['graph_data'] = graph_data
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
        
        response = self.ai_service.generate_response(formatted_prompt, "")
        
        state['response'] = response
        state['chat_history'] = self.memory_manager.add_interaction(query, response)
        return state

    def process_text(self, text: str):
        logging.info("Processing text...")
        chunks = self.text_processor.chunk_text(text)
        for chunk in chunks:
            embedding = self.ai_service.get_embedding(chunk.page_content)
            self.graph_manager.store_data(chunk, embedding)
            self.vector_store_manager.add_document(chunk)
        logging.info(f"Text processing completed. Total documents: {self.vector_store_manager.document_count()}")

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

def main():
    logging.info("Starting the program...")
    
    processor = LocalLlamaProcessor()
    
    logging.info("Processing sample text...")
    sample_text = """
    NVIDIA (NASDAQ: NVDA) today reported revenue for the first quarter ended April 28, 2024, of $26.0 billion, up 18% from the previous quarter and up 262% from a year ago.

    For the quarter, GAAP earnings per diluted share was $5.98, up 21% from the previous quarter and up 629% from a year ago. Non-GAAP earnings per diluted share was $6.12, up 19% from the previous quarter and up 461% from a year ago.

    "The next industrial revolution has begun — companies and countries are partnering with NVIDIA to shift the trillion-dollar traditional data centers to accelerated computing and build a new type of data center — AI factories — to produce a new commodity: artificial intelligence," said Jensen Huang, founder and CEO of NVIDIA. "AI will bring significant productivity gains to nearly every industry and help companies be more cost- and energy-efficient, while expanding revenue opportunities.

    "Our data center growth was fueled by strong and accelerating demand for generative AI training and inference on the Hopper platform. Beyond cloud service providers, generative AI has expanded to consumer internet companies, and enterprise, sovereign AI, automotive and healthcare customers, creating multiple multibillion-dollar vertical markets.
    """
    processor.process_text(sample_text)
    
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        response = processor.process_query(query)
        
        logging.info(f"Query: {query}")
        logging.info(f"Response: {response}")
        print(f"\nResponse: {response}\n")
    
    logging.info("Program completed.")

if __name__ == "__main__":
    main()