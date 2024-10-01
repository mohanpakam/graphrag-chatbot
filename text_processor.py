import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

# Configure logging
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, model="llama3.2"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = OllamaEmbeddings(model=model)

    def chunk_text(self, text: str) -> List[Document]:
        logger.info("Chunking text...")
        chunks = self.text_splitter.create_documents([text])
        logger.info(f"Text chunked into {len(chunks)} documents.")
        return chunks

    def create_embedding(self, text: str) -> List[float]:
        logger.info("Creating embedding...")
        embedding = self.embeddings.embed_query(text)
        logger.info("Embedding created successfully.")
        return embedding

def init_database():
    # This function seems out of place in TextProcessor. Consider moving it to a database management module.
    logger.warning("init_database() called in TextProcessor. Consider moving this functionality.")
    pass