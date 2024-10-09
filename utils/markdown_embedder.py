import os
from typing import List, Dict
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from utils import DatabaseManager
import json
import hashlib

class MarkdownEmbedder:
    def __init__(self, db_path: str):
        self.db_manager = DatabaseManager(db_path)
        self.embeddings = VertexAIEmbeddings()
        self.text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_markdown_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = self.text_splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk)
            
            # Create a unique hash for the chunk
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

            doc_data = {
                'filename': os.path.basename(file_path),
                'chunk_index': i,
                'content': chunk,
                'embedding': json.dumps(embedding),
                'chunk_hash': chunk_hash
            }

            self.store_embedding(doc_data)

    def store_embedding(self, doc_data: Dict):
        with self.db_manager.connect() as conn:
            cursor = conn.execute('''
                INSERT INTO MIM_pdf_documents 
                (filename, chunk_index, content, embedding, chunk_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                doc_data['filename'],
                doc_data['chunk_index'],
                doc_data['content'],
                doc_data['embedding'],
                doc_data['chunk_hash']
            ))
            return cursor.lastrowid

    def process_directory(self, directory_path: str):
        for filename in os.listdir(directory_path):
            if filename.endswith('.md'):
                file_path = os.path.join(directory_path, filename)
                self.process_markdown_file(file_path)

    def init_database(self):
        with self.db_manager.connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS MIM_pdf_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL UNIQUE
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_mim_pdf_chunk_hash ON MIM_pdf_documents(chunk_hash)')

        print("MIM_pdf_documents table initialized with required columns and index.")

    def clear_embeddings(self):
        with self.db_manager.connect() as conn:
            conn.execute('DELETE FROM MIM_pdf_documents')
        print("All embeddings cleared from the MIM_pdf_documents table.")

if __name__ == "__main__":
    db_path = "your_database.db"
    markdown_dir = "path/to/your/markdown/files"

    embedder = MarkdownEmbedder(db_path)
    embedder.init_database()
    embedder.clear_embeddings()  # Optional: clear existing embeddings before processing
    embedder.process_directory(markdown_dir)