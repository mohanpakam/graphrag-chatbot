import os
import time
import yaml
import sqlite3
import requests
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from database_manager import DatabaseManager

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Separate configuration for local Ollama API
local_ollama_config = {
    'api_url': config.get('local_ollama_embeddings_api_url', 'http://localhost:11434/api/embeddings'),
    'model': config.get('local_ollama_model', 'llama3.1')
}

# Initialize API call counter and total response time
embedding_api_calls = 0
total_response_time = 0

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        length_function=len
    )
    return text_splitter.split_text(text)

def get_embeddings(texts):
    global embedding_api_calls, total_response_time
    embeddings = []
    for text in texts:
        start_time = time.time()
        response = requests.post(
            local_ollama_config['api_url'],
            json={"model": local_ollama_config['model'], "prompt": text}
        )
        end_time = time.time()
        response_time = end_time - start_time
        total_response_time += response_time
        embedding_api_calls += 1
        
        if response.status_code == 200:
            embedding = response.json()['embedding']
            embeddings.append(embedding)
            print(f"Processed chunk {embedding_api_calls}. Response time: {response_time:.2f} seconds")
            print(f"Embedding (first 5 values): {embedding[:5]}")
        else:
            print(f"Error processing chunk {embedding_api_calls}: {response.text}")
    
    return embeddings

def process_texts():
    global embedding_api_calls, total_response_time
    text_folder = config['text_folder']
    db_path = config['database_path']

    db_manager = DatabaseManager(db_path)

    with db_manager.connect() as conn:
        
        # sqlite_version, vec_version = conn.execute("select sqlite_version(), vec_version()" ).fetchone()
        # print(f"sqlite_version={sqlite_version}, vec_version={vec_version}")
        # Create table for storing document chunks and vectors
        conn.execute('''CREATE TABLE IF NOT EXISTS documents 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         content TEXT,
                         embedding BLOB)''')

        for filename in os.listdir(text_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(text_folder, filename)
                print(f"Processing {filename}...")
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                chunks = create_chunks(text)
                embeddings = get_embeddings(chunks)

                # Store chunks and embeddings in the database
                for chunk, embedding in zip(chunks, embeddings):
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    conn.execute('INSERT INTO documents (content, embedding) VALUES (?, ?)',
                                 (chunk, embedding_bytes))

        conn.commit()

    print(f"\nProcessing complete.")
    print(f"Total embedding API calls: {embedding_api_calls}")
    print(f"Total response time: {total_response_time:.2f} seconds")
    print(f"Average response time: {(total_response_time / embedding_api_calls):.2f} seconds per call")

if __name__ == "__main__":
    process_texts()