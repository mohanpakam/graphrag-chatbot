import logging
import os
from typing import List
from langchain.schema import Document
from text_processor import TextProcessor
from vector_store_manager import VectorStoreManager
from graph_rag import GraphRAG
import yaml

# Load config
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalLlamaProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.graph_rag = GraphRAG()

    def process_text(self, text: str, filename: str = "unknown"):
        logger.info(f"Processing text from {filename}...")
        chunks = self.text_processor.chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk.metadata['filename'] = filename
            chunk.metadata['chunk_index'] = i
            self.vector_store_manager.add_document(chunk)
        logger.info(f"Text processing completed for {filename}. Total documents: {self.vector_store_manager.document_count()}")

    def process_files_from_folder(self, folder_path: str = 'texts'):
        logger.info(f"Processing files from folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                self.process_text(text, filename)
        logger.info("Finished processing all files from the folder.")

def main():
    logger.info("Starting the Local Llama Processor...")
    
    processor = LocalLlamaProcessor()
    
    # Process files from the 'texts' folder
    processor.process_files_from_folder()
   
    response = processor.graph_rag.process_query("exercise")
        
    logger.info(f"Response: {response}")
    
    logger.info("Local Llama Processor completed.")

if __name__ == "__main__":
    main()