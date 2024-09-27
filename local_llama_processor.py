import sqlite3
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
import json
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Ollama
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = Ollama(
    model="llama3.1",  # or any other model you have in Ollama
    callback_manager=callback_manager,
)

# Initialize embeddings
embeddings = OllamaEmbeddings(model="llama3.1")  # Use the same model as the LLM

# Initialize graph transformer
graph_transformer = LLMGraphTransformer(llm=llm)

# Initialize SQLite database
conn = sqlite3.connect('ollama_vss.db')
cursor = conn.cursor()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize vector store
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

def init_database():
    logging.info("Initializing database...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT,
        embedding BLOB
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS graph_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        node_id TEXT,
        node_type TEXT,
        properties TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS graph_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        source_id TEXT,
        target_id TEXT,
        relationship_type TEXT,
        properties TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )
    ''')
    conn.commit()
    logging.info("Database initialized successfully.")

def chunk_text(text: str) -> List[Document]:
    logging.info("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.create_documents([text])
    logging.info(f"Text chunked into {len(chunks)} documents.")
    return chunks

def create_embedding(text: str) -> List[float]:
    logging.info("Creating embedding...")
    embedding = embeddings.embed_query(text)
    logging.info(f"Embedding created successfully. {embedding}")
    return embedding

def create_graph(document: Document) -> Dict[str, Any]:
    logging.info("Creating graph from document...")
    graph_document = graph_transformer.process_response(document)
    logging.info(f"Graph created successfully. Nodes: {len(graph_document.nodes)}, Relationships: {len(graph_document.relationships)}")
    return {
        "nodes": graph_document.nodes,
        "relationships": graph_document.relationships
    }

def store_document(content: str, embedding: List[float], graph: Dict[str, Any]):
    logging.info(f"Storing document in database...")
    cursor.execute('INSERT INTO documents (content, embedding) VALUES (?, ?)',
                   (content, json.dumps(embedding)))
    document_id = cursor.lastrowid
    logging.info(f"Document stored with ID: {document_id}")

    for node in graph['nodes']:
        cursor.execute('''
        INSERT INTO graph_nodes (document_id, node_id, node_type, properties)
        VALUES (?, ?, ?, ?)
        ''', (document_id, str(node.id), node.type, json.dumps(node.properties)))

    for rel in graph['relationships']:
        cursor.execute('''
        INSERT INTO graph_relationships 
        (document_id, source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?, ?)
        ''', (document_id, str(rel.source), str(rel.target), rel.type, json.dumps(rel.properties)))

    conn.commit()
    logging.info("Document stored successfully.")

def process_text(text: str):
    logging.info("Processing text...")
    init_database()
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks, 1):
        logging.info(f"Processing chunk {i}/{len(chunks)}...")
        embedding = create_embedding(chunk.page_content)
        graph = create_graph(chunk)
        store_document(chunk.page_content, embedding, graph)
        # Add document to vector store
        vectorstore.add_texts([chunk.page_content], metadatas=[{"id": str(i)}])
    
    vectorstore.persist()
    logging.info("Text processing completed.")

def generate_response(query: str):
    logging.info(f"Generating response for query: '{query}'")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )
    
    result = qa_chain({"question": query})
    response = result['answer']
    
    logging.info("Response generated successfully.")
    return response

if __name__ == "__main__":
    logging.info("Starting the program...")
    
    sample_text = """
    Title: The Importance of Sustainable Energy

As the world grapples with climate change, the transition to sustainable energy sources has become more critical than ever. Renewable energy technologies such as solar, wind, and hydroelectric power offer cleaner alternatives to fossil fuels.

Benefits of sustainable energy:
- Reduced greenhouse gas emissions
- Improved air quality
- Energy independence
- Job creation in the green sector

Challenges in adopting sustainable energy:
1. Initial infrastructure costs
2. Intermittency of some renewable sources
3. Energy storage solutions
4. existing power grid sucks
5. Electric cars are used too much

Despite these challenges, many countries are making significant strides in renewable energy adoption. As technology improves and costs decrease, sustainable energy is becoming increasingly competitive with traditional fossil fuels.

Investing in sustainable energy is not just an environmental imperative but also an economic opportunity that can drive innovation and growth in the coming decades.
    """
    
    process_text(sample_text)
    
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        response = generate_response(query)
        
        logging.info(f"Query: {query}")
        logging.info(f"Response: {response}")
        print(f"\nResponse: {response}\n")
    
    vectorstore.persist()  # Save the vector store to disk
    conn.close()
    logging.info("Program completed.")