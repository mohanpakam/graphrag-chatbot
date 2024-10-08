import os
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
import faiss
import numpy as np
from dataclasses import dataclass
import json
from embedding_cache import EmbeddingCache
from langchain.schema import Document
from graph_manager import GraphManager

class EmbedFileToFaiss:
    def __init__(self, project_id, location, credentials_path, index_file, db_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        aiplatform.init(project=project_id, location=location)
        self.embeddings = VertexAIEmbeddings()
        self.index = None
        self.embedding_cache = EmbeddingCache(index_file, db_path)
        self.graph_manager = GraphManager(db_path)  # Initialize GraphManager

    def get_file_embedding(self, file_path):
        if file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        document = loader.load()[0]  # Assuming single document per file
        
        # Use embed_documents method for consistency, even though it's a single document
        document_embedding = self.embeddings.embed_documents([document.page_content])[0]
        
        filename = os.path.basename(file_path)
        return document, document_embedding, filename

    def add_to_index(self, document, document_embedding, filename):
        embedding_array = np.array([document_embedding]).astype('float32')
        if self.index is None:
            dimension = len(document_embedding)
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embedding_array)
        metadata = {
            "filename": filename,
            "metadata": document.metadata
        }
        doc_id = self.embedding_cache.store_document(filename, document.page_content, metadata)
        # Ensure the FAISS index ID matches the SQLite row ID
        self.index.reconstruct(self.index.ntotal - 1, embedding_array[0])  # This updates the last added vector

        return doc_id

    def process_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.txt', '.md')):
                file_path = os.path.join(folder_path, filename)
                document, embedding, filename = self.get_file_embedding(file_path)
                doc_id = self.add_to_index(document, embedding, filename)
                
                # Process the entire document with GraphManager
                self.graph_manager.process_document(document, doc_id)
                
                print(f"Processed {filename}")

    def save_index(self, index_file):
        faiss.write_index(self.index, index_file)
        print(f"Index saved to {index_file}")

if __name__ == "__main__":
    project_id = "your-project-id"
    location = "your-location"
    credentials_path = "path/to/your/google-cloud-credentials.json"
    index_file = "embeddings_index.faiss"
    db_path = "embeddings.db"
    
    embedder = EmbedFileToFaiss(project_id, location, credentials_path, index_file, db_path)
    
    folder_path = "path/to/your/text/files/folder"
    embedder.process_folder(folder_path)
    
    embedder.save_index(index_file)
    
    # Example search
    query_embedding = embedder.embeddings.embed_query("Your query text here")
    similar_docs = embedder.embedding_cache.find_similar(query_embedding)
    
    print("Similar documents:")
    for result in similar_docs:
        print(result)
        print("---")
