import os
import numpy as np
from database_manager import DatabaseManager
from ai_service import get_ai_service
import yaml
import spacy
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

def process_text_files(folder_path):
    # Initialize the database and create tables
    db_manager.init_database()
    
    # Clear existing documents (optional, comment out if you want to keep existing documents)
    db_manager.clear_documents()

    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Chunk the content using spaCy
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            logger.info(f"Processing {filename}: {len(chunks)} chunks created")
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for each chunk
                embedding = ai_service.get_embedding(chunk)
                
                # Print the dimension of the embedding
                embedding_dim = len(embedding)
                logger.info(f"Embedding dimension for chunk {i}: {embedding_dim}")
                
                # Convert the embedding to a binary format
                embedding_binary = np.array(embedding, dtype=np.float32).tobytes()
                
                # Store the chunk and its embedding in the database
                with db_manager.connect() as conn:
                    conn.execute('''
                        INSERT INTO documents (filename, chunk_index, content, embedding)
                        VALUES (?, ?, ?, ?)
                    ''', (filename, i, chunk, embedding_binary))
                
                # Verify that the stored embedding matches the original
                with db_manager.connect() as conn:
                    stored_embedding = conn.execute('''
                        SELECT embedding FROM documents
                        WHERE filename = ? AND chunk_index = ?
                    ''', (filename, i)).fetchone()[0]
                
                stored_embedding_array = np.frombuffer(stored_embedding, dtype=np.float32)
                
                if not np.array_equal(np.array(embedding, dtype=np.float32), stored_embedding_array):
                    logger.error(f"Mismatch in stored embedding for {filename}, chunk {i}")
                else:
                    logger.info(f"Embedding for {filename}, chunk {i} stored correctly")

    logger.info("Text processing, chunking, and embedding generation complete.")

if __name__ == "__main__":
    text_folder = config['text_folder']
    process_text_files(text_folder)