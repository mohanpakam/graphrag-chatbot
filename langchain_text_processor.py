import os
from typing import List, Any
from langchain.text_splitter import SpacyTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
import yaml
from logger_config import LoggerConfig
import tiktoken
from database_manager import DatabaseManager
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ai_service import get_langchain_ai_service
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import Document as LangChainDocument

# Load configuration
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
logger = LoggerConfig.setup_logger(__name__)
db_manager = DatabaseManager(config['database_path'])

# Set up callback manager for logging
callback_manager = CallbackManager([StdOutCallbackHandler()])

# Get the appropriate LangChain AI service with callback manager
ai_service = get_langchain_ai_service(config['ai_service'], callback_manager)
logger.info(f"Initialized AI service: {type(ai_service).__name__}")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Calculate the number of tokens in a string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, use_tiktoken: bool) -> List[Document]:
    """Split text into chunks using either TokenTextSplitter or SpacyTextSplitter."""
    logger.debug(f"Chunking text with size {chunk_size} and overlap {chunk_overlap}")
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

def chunk_document(document: Document, chunk_size: int, chunk_overlap: int, use_tiktoken: bool) -> List[Document]:
    """Chunk a single document."""
    return chunk_text(document.page_content, chunk_size, chunk_overlap, use_tiktoken)

def convert_chunks_to_graph_documents(chunks: List[Document], llm_transformer: LLMGraphTransformer, filename: str) -> List[Any]:
    """Convert chunks to graph documents."""
    graph_documents = []
    for i, chunk in enumerate(chunks):
        try:
            if not isinstance(chunk, LangChainDocument):
                chunk = LangChainDocument(page_content=chunk.page_content, metadata=chunk.metadata)
            logger.debug(f" Before llm_transformer.convert_to_graph_documents Chunks {chunk}")
            graph_doc = llm_transformer.process_response(chunk)
            logger.debug(f"graph_doc = {graph_doc}")
            graph_documents.extend(graph_doc)
            logger.debug(f"graph_documents.extend {graph_documents}")
        except Exception as chunk_error:
            logger.error(f"Error processing chunk {i} in {filename}: {str(chunk_error)}")
            logger.error(f"Problematic chunk: {chunk}")
            continue  # Skip this chunk and continue with the next one
    logger.info(f"Converted {len(graph_documents)} chunks to graph documents for {filename}")
    logger.debug(f"Graph Info {graph_documents}")
    return graph_documents

def process_graph_documents(graph_documents: List[Any], filename: str, use_tiktoken: bool):
    """Process and store graph documents."""
    for i, graph_doc in enumerate(graph_documents):
        try:
            embedding = ai_service.get_embedding(graph_doc.source.page_content)
            store_graph_document(graph_doc, embedding, filename, i)
            
            if use_tiktoken:
                token_count = num_tokens_from_string(graph_doc.source.page_content, "cl100k_base")
                logger.info(f"Graph document {i} from {filename} has {token_count} tokens")
        except AttributeError as ae:
            logger.error(f"AttributeError processing graph_doc {i}: {str(ae)}")
            logger.error(f"graph_doc content: {graph_doc}")
        except Exception as e:
            logger.error(f"Error processing graph_doc {i}: {str(e)}")

def process_single_file(file_path: str, chunk_size: int, chunk_overlap: int, use_tiktoken: bool, llm_transformer: LLMGraphTransformer):
    """Process a single text file."""
    filename = os.path.basename(file_path)
    logger.info(f"Processing file in Langchain text processor: {filename}")
    
    loader = TextLoader(file_path)
    documents = loader.load()
    
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap, use_tiktoken))
    logger.info(f"Created {len(chunks)} chunks for {filename}")
    
    try:
        graph_documents = convert_chunks_to_graph_documents(chunks, llm_transformer, filename)
        process_graph_documents(graph_documents, filename, use_tiktoken)
        logger.info(f"Processed and stored graph documents for {filename}")
    except Exception as e:
        logger.error(f"Error in graph document conversion process for {filename}: {str(e)}")

def process_text_files(folder_path: str):
    """Process all text files in the given folder."""
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']
    use_tiktoken = config['ai_service'] in ['openai', 'azure_openai']

    db_manager.init_database()
    logger.info("Database initialized")

    llm_transformer = LLMGraphTransformer(llm=ai_service.llm)
    logger.info("LLMGraphTransformer initialized")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            process_single_file(file_path, chunk_size, chunk_overlap, use_tiktoken, llm_transformer)

    logger.info("Text processing, graph conversion, and storage complete.")

def store_graph_document(graph_doc, embedding, filename, index):
    """Store a graph document, its nodes, relationships, and embedding in the database."""
    try:
        # Store the main document data
        doc_data = {
            'filename': filename,
            'chunk_index': index,
            'content': graph_doc.source.page_content,
            'embedding': embedding  # Add embedding to the document data
        }
        db_manager.store_graph_document(doc_data)
        logger.debug(f"Stored graph document with embedding: {filename}, index {index}")
        
        # Store nodes
        for node in graph_doc.nodes:
            node_data = {
                'id': node.id,
                'type': node.type,
                'properties': node.properties,
                'document_filename': filename,
                'document_index': index
            }
            db_manager.store_node(node_data)
        logger.debug(f"Stored {len(graph_doc.nodes)} nodes for document: {filename}, index {index}")
        
        # Store relationships
        for rel in graph_doc.relationships:
            relationship_data = {
                'source_id': rel.source.id,
                'target_id': rel.target.id,
                'type': rel.type,
                'properties': {},  # MockRelationship doesn't have properties, so we're using an empty dict
                'document_filename': filename,
                'document_index': index
            }
            db_manager.store_relationship(relationship_data)
        logger.debug(f"Stored {len(graph_doc.relationships)} relationships for document: {filename}, index {index}")
    except AttributeError as ae:
        logger.error(f"AttributeError in store_graph_document: {str(ae)}")
        logger.error(f"graph_doc content: {graph_doc}")
    except Exception as e:
        logger.error(f"Error in store_graph_document: {str(e)}")

if __name__ == "__main__":
    text_folder = config['text_folder']
    logger.info(f"Starting text processing for folder: {text_folder}")
    process_text_files(text_folder)