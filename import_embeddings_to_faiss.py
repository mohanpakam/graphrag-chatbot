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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class EmbedFileToFaiss:
    def __init__(self, project_id, location, credentials_path, index_file, db_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        aiplatform.init(project=project_id, location=location)
        self.embeddings = VertexAIEmbeddings()
        self.index = None
        self.embedding_cache = EmbeddingCache(index_file, db_path)
        self.graph_manager = GraphManager(db_path)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def get_file_embedding(self, file_path):
        if file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        document = loader.load()[0]  # Assuming single document per file
        chunks = self.text_splitter.split_text(document.page_content)
        
        # Generate embeddings for chunks
        chunk_embeddings = self.embeddings.embed_documents(chunks)
        
        filename = os.path.basename(file_path)
        return document, chunks, chunk_embeddings, filename

    def add_to_index(self, document, chunks, chunk_embeddings, filename):
        embedding_array = np.array(chunk_embeddings).astype('float32')
        if self.index is None:
            dimension = len(chunk_embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
        
        metadata = {
            "filename": filename,
            "metadata": document.metadata
        }
        doc_id, chunk_ids = self.embedding_cache.store_document(filename, document.page_content, metadata, chunks)
        
        # Add embeddings to FAISS index and ensure IDs match
        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embedding_array)):
            self.index.add_with_ids(embedding.reshape(1, -1), np.array([chunk_id]))

        logger.info(f"Added document {filename} with {len(chunks)} chunks to index and database")
        return doc_id

    def process_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.txt', '.md')):
                file_path = os.path.join(folder_path, filename)
                document, chunks, chunk_embeddings, filename = self.get_file_embedding(file_path)
                doc_id = self.add_to_index(document, chunks, chunk_embeddings, filename)
                
                # Process the entire document with GraphManager
                self.graph_manager.process_document(document, doc_id)
                
                logger.info(f"Processed {filename}")

    def save_index(self, index_file):
        faiss.write_index(self.index, index_file)
        logger.info(f"Index saved to {index_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
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
    similar_chunks = embedder.embedding_cache.find_similar_chunks(query_embedding)
    
    logger.info("Similar chunks:")
    for result in similar_chunks:
        logger.info(f"Document ID: {result['document_id']}, Chunk Index: {result['chunk_index']}, Distance: {result['distance']}")
        logger.info(f"Content preview: {result['content'][:100]}...")
        logger.info("---")
