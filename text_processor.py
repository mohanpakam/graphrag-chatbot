import spacy
import numpy as np
from database_manager import DatabaseManager
from ai_service import get_ai_service
import yaml
from logger_config import LoggerConfig

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Initialize the AI service
ai_service = get_ai_service()

db_manager = DatabaseManager(config['database_path'])

def chunk_text(text, chunk_size, chunk_overlap):
    doc = nlp(text)
    sentences = list(doc.sents)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_text = sentence.text.strip()
        sentence_length = len(sentence_text)

        if current_chunk_size + sentence_length <= chunk_size:
            current_chunk.append(sentence_text)
            current_chunk_size += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence_text]
            current_chunk_size = sentence_length

        # Check for overlap
        while current_chunk_size > chunk_overlap and len(current_chunk) > 1:
            removed_sentence = current_chunk.pop(0)
            current_chunk_size -= len(removed_sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Handle case where a single sentence is longer than chunk_size
    if not chunks and sentences:
        chunks = [sentences[0].text.strip()]

    return chunks

def process_text(text: str, filename: str) -> List[str]:
    chunks = chunk_text(text)
    return chunks

def init_database():
    # Initialize the database and create tables
    db_manager.init_database()
    
    # Clear existing documents (optional, comment out if you want to keep existing documents)
    db_manager.clear_documents()