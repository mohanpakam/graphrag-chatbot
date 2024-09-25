import os
import time
import yaml
import sqlite3
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

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
            config['ollama_api_url'],
            json={"model": config['ollama_model'], "prompt": text}
        )
        end_time = time.time()
        response_time = end_time - start_time
        total_response_time += response_time
        embedding_api_calls += 1
        
        if response.status_code == 200:
            embedding = response.json()['embedding']
            embeddings.append(embedding)
            print(f"Processed chunk {embedding_api_calls}. Response time: {response_time:.2f} seconds")
        else:
            print(f"Error processing chunk {embedding_api_calls}: {response.text}")
    
    return embeddings

def process_texts():
    global embedding_api_calls, total_response_time
    text_folder = config['text_folder']
    db_path = config['database_path']

    # Initialize SQLite database with vector search capability
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension("sqlite_vss")

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
                conn.execute('INSERT INTO documents (content, embedding) VALUES (?, ?)',
                             (chunk, sqlite3.Binary(bytes(embedding))))

    conn.commit()
    conn.close()

    print(f"\nProcessing complete.")
    print(f"Total embedding API calls: {embedding_api_calls}")
    print(f"Total response time: {total_response_time:.2f} seconds")
    print(f"Average response time: {(total_response_time / embedding_api_calls):.2f} seconds per call")

if __name__ == "__main__":
    process_texts()