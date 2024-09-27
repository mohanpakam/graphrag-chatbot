import os
from tests.text_processor import process_text, init_database, load_config
from src.common.logger_config import LoggerConfig
from src.common.database_manager import DatabaseManager
from src.graphrag.langchain_ai_service import get_langchain_ai_service

config = load_config()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

# Initialize database manager and AI service
db_manager = DatabaseManager(config['database_path'])
ai_service = get_langchain_ai_service(config['ai_service'])

def process_text_files(folder_path):
    db_manager.init_database()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Process text and get chunks
            chunks = process_text(content, filename)
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                embedding = ai_service.get_embedding(chunk)
                
                # Store the document with embedding
                doc_data = {
                    'filename': filename,
                    'chunk_index': i,
                    'content': chunk,
                    'embedding': embedding
                }
                db_manager.store_graph_document(doc_data)
                
                logger.info(f"Processed and stored chunk {i} from {filename}")

    logger.info("Text processing, chunking, and embedding generation complete.")

if __name__ == "__main__":
    text_folder = config['text_folder']
    process_text_files(text_folder)