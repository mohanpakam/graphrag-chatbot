import os
import yaml
import sqlite3
import PyPDF2
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SQLiteVSS

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=config['azure_openai_api_key'],
    api_version="2023-05-15",
    azure_endpoint=config['azure_openai_endpoint']
)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        length_function=len
    )
    return text_splitter.split_text(text)

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(input=text, model=config['embedding_model'])
        embeddings.append(response.data[0].embedding)
    return embeddings

def process_pdfs():
    pdf_folder = config['pdf_folder']
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

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = create_chunks(text)
            embeddings = get_embeddings(chunks)

            # Store chunks and embeddings in the database
            for chunk, embedding in zip(chunks, embeddings):
                conn.execute('INSERT INTO documents (content, embedding) VALUES (?, ?)',
                             (chunk, sqlite3.Binary(bytes(embedding))))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    process_pdfs()