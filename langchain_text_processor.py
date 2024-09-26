import os
from typing import List
from langchain.text_splitter import SpacyTextSplitter, TokenTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
import yaml
from logger_config import LoggerConfig
import tiktoken
import numpy as np
from database_manager import DatabaseManager
import spacy

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
logger = LoggerConfig.setup_logger(__name__)
db_manager = DatabaseManager(config['database_path'])

# Load spaCy model
#nlp = spacy.load("en_core_web_lg")

def get_embeddings():
    if config['ai_service'] == 'openai':
        return OpenAIEmbeddings(api_key=config['openai_api_key'])
    elif config['ai_service'] == 'azure_openai':
        return AzureOpenAIEmbeddings(
            deployment=config['azure_openai_embedding_deployment'],
            model="text-embedding-ada-002",
            openai_api_key=config['azure_openai_api_key'],
            azure_endpoint=config['azure_openai_endpoint'],
            api_version="2023-05-15"
        )
    elif config['ai_service'] == 'ollama':
        return OllamaEmbeddings(model=config['ollama_model'])
    else:
        raise ValueError(f"Unsupported AI service: {config['ai_service']}")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, use_tiktoken: bool) -> List[Document]:
    if use_tiktoken:
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )
    else:
        text_splitter = SpacyTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return text_splitter.create_documents([text])

def process_text_files(folder_path: str):
    embeddings = get_embeddings()
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']
    use_tiktoken = config['ai_service'] in ['openai', 'azure_openai']

    db_manager.init_database()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                chunks = chunk_text(doc.page_content, chunk_size, chunk_overlap, use_tiktoken)
                for i, chunk in enumerate(chunks):
                    embedding = embeddings.embed_query(chunk.page_content)
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    
                    with db_manager.connect() as conn:
                        conn.execute('''
                        INSERT INTO documents (filename, chunk_index, content, embedding)
                        VALUES (?, ?, ?, ?)
                        ''', (filename, i, chunk.page_content, embedding_bytes))
                    
                    if use_tiktoken:
                        token_count = num_tokens_from_string(chunk.page_content, "cl100k_base")
                        logger.info(f"Chunk {i} from {filename} has {token_count} tokens")

            logger.info(f"Processed and stored chunks for {filename}")

    logger.info("Text processing, chunking, and embedding generation complete.")

if __name__ == "__main__":
    text_folder = config['text_folder']
    process_text_files(text_folder)