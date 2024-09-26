import os
from text_processor import process_text, init_database, load_config
from logger_config import LoggerConfig

config = load_config()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

def process_text_files(folder_path):
    init_database()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            process_text(content, filename)

    logger.info("Text processing, chunking, and embedding generation complete.")

if __name__ == "__main__":
    text_folder = config['text_folder']
    process_text_files(text_folder)