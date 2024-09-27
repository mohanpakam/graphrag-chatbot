from database_manager import DatabaseManager
from langchain_ai_service import get_langchain_ai_service
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
db_manager = DatabaseManager(config['database_path'])
ai_service = get_langchain_ai_service(config['ai_service'])

def retrieve_relevant_subgraph(query: str, top_k: int = 5):
    query_embedding = ai_service.get_embedding(query)
    similar_docs = db_manager.find_similar_documents(query_embedding, top_k)
    subgraph = db_manager.get_subgraph_for_documents(similar_docs)
    return subgraph, similar_docs

def subgraph_to_context(subgraph, similar_docs):
    context = "Relevant information:\n"
    for node in subgraph['nodes']:
        context += f"- {node[2]}: {node[1]}\n"
    for rel in subgraph['relationships']:
        context += f"- {rel[1]} {rel[3]} {rel[2]}\n"
    
    context += "\nRelevant document excerpts:\n"
    for doc in similar_docs:
        content = db_manager.get_document_content(doc[1], doc[2])
        context += f"- {content[:200]}...\n"
    
    return context

def generate_response(query: str):
    subgraph, similar_docs = retrieve_relevant_subgraph(query, config['top_k'])
    context = subgraph_to_context(subgraph, similar_docs)
    response = ai_service.generate_response(query, context)
    return response