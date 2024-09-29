import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import yaml
import logging
from logger_config import LoggerConfig
from langchain_ai_service import VertexGeminiLangChainAIService

# Load configuration
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

# Initialize Vertex AI service
vertex_ai_service = VertexGeminiLangChainAIService()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize vector store
vectorstore = Chroma(embedding_function=vertex_ai_service.embeddings, persist_directory="./chroma_db")

def chunk_text(text: str) -> List[Document]:
    logger.info("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
    )
    chunks = text_splitter.create_documents([text])
    logger.info(f"Text chunked into {len(chunks)} documents.")
    return chunks

def process_text(text: str, filename: str):
    logger.info(f"Processing text from file: {filename}")
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)}...")
        # Add document to vector store
        vectorstore.add_texts([chunk.page_content], metadatas=[{"source": filename, "chunk": i}])
    
    vectorstore.persist()
    logger.info(f"Text processing completed for file: {filename}")

def process_text_files(folder_path: str):
    logger.info(f"Processing text files from folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            process_text(text, filename)
    logger.info("All text files processed.")

def generate_response(query: str):
    logger.info(f"Generating response for query: '{query}'")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=vertex_ai_service.llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )
    
    result = qa_chain({"question": query})
    response = result['answer']
    
    logger.info("Response generated successfully.")
    return response

if __name__ == "__main__":
    logger.info("Starting the Vertex AI File Chat program...")
    
    text_folder = config['text_folder']
    process_text_files(text_folder)
    
    print("Welcome to the Vertex AI File Chat!")
    print("You can start asking questions about the processed documents.")
    print("Type 'quit' to exit the program.")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'quit':
            break
        
        response = generate_response(query)
        
        print(f"\nResponse: {response}\n")
    
    vectorstore.persist()
    logger.info("Vertex AI File Chat program completed.")