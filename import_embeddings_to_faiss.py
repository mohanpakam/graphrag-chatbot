import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from google.cloud import aiplatform
import faiss
import numpy as np
from dataclasses import dataclass
import json
from utils import EmbeddingCache
from langchain.schema import Document
from graphrag import GraphManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ai_services import get_langchain_ai_service
from config import LoggerConfig


config = LoggerConfig.load_config()
logger = LoggerConfig.setup_logger(__name__)

class EmbedFileToFaiss:
    def __init__(self, index_file, db_path):
       
        self.ai_service = get_langchain_ai_service(config['ai_service'])
        self.index = None
        self.embedding_cache = EmbeddingCache(index_file, db_path)
        self.graph_manager = GraphManager(db_path, self.ai_service)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def get_file_embedding(self, file_path):
        if file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        loaded_documents = loader.load()
        if not loaded_documents:
            raise ValueError(f"No content loaded from file: {file_path}")
        
        document = loaded_documents[0]
        if not isinstance(document, Document):
            raise TypeError(f"Loaded content is not a Document object: {type(document)}")
        
        if not hasattr(document, 'page_content') or not document.page_content:
            raise ValueError(f"Loaded document has no page_content: {file_path}")
        
        # Set metadata
        filename = os.path.basename(file_path)
        document.metadata = {
            "source": file_path,
            "filename": filename,
            "file_type": "markdown" if file_path.endswith('.md') else "text",
            "creation_date": os.path.getctime(file_path),
            "last_modified_date": os.path.getmtime(file_path),
        }
        
        chunks = self.text_splitter.split_text(document.page_content)
        
        # Generate embeddings for chunks using the AI service
        chunk_embeddings = [self.ai_service.get_embedding(chunk) for chunk in chunks]
        
        return document, chunks, chunk_embeddings, filename

    def add_to_index(self, document, chunks, chunk_embeddings, filename):
        embedding_array = np.array(chunk_embeddings).astype('float32')
        if self.index is None:
            dimension = len(chunk_embeddings[0])
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)
        
        doc_id, chunk_ids = self.embedding_cache.store_document(
            filename=filename,
            content=document.page_content,
            metadata=document.metadata,
            chunks=chunks
        )
        
        # Add embeddings to FAISS index with custom IDs
        self.index.add_with_ids(embedding_array, np.array(chunk_ids))

        logger.info(f"Added document '{filename}' (ID: {doc_id}) with {len(chunks)} chunks to index and database")
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

    def test_retrieval(self, query: str):
        logger.info(f"Testing retrieval for query: '{query}'")
        
        # Get query embedding
        query_embedding = self.ai_service.get_embedding(query)
        
        # Find similar chunks
        similar_chunks = self.embedding_cache.find_similar_chunks(query_embedding, k=5)
        
        logger.info("Similar chunks:")
        for result in similar_chunks:
            logger.info(f"Document ID: {result['document_id']}, Chunk Index: {result['chunk_index']}, Distance: {result['distance']}")
            logger.info(f"Content preview: {result['content'][:100]}...")
        
        # Get graph data for the most similar chunk's document
        if similar_chunks:
            most_similar_doc_id = similar_chunks[0]['document_id']
            graph_data = self.graph_manager.get_subgraph_for_documents([most_similar_doc_id])
            
            logger.info("\nGraph data:")
            logger.info(f"Nodes: {len(graph_data['nodes'])}")
            logger.info(f"Relationships: {len(graph_data['relationships'])}")
            
            # Display a few nodes and relationships as examples
            for node in graph_data['nodes'][:3]:
                logger.info(f"Node: {node}")
            for rel in graph_data['relationships'][:3]:
                logger.info(f"Relationship: {rel}")

if __name__ == "__main__":
    index_file = config["faiss_index_file"]
    db_path = config["database_path"]
    
    embedder = EmbedFileToFaiss(index_file, db_path)
    
    folder_path = config["text_folder"]
    embedder.process_folder(folder_path)
    
    embedder.save_index(index_file)
    
    # Test retrieval
    test_queries = [
        "What are the common root causes of production issues?",
        "How do we handle database failures?",
        "What's the process for escalating critical issues?",
        "Can you summarize recent network outages?",
        "Who are the key people involved in incident response?"
    ]
    
    for query in test_queries:
        embedder.test_retrieval(query)
        print("\n" + "="*50 + "\n")