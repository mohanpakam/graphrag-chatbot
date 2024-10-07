import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
import faiss
import numpy as np
import pickle

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/google-cloud-credentials.json"

# Initialize Google Cloud AI Platform
aiplatform.init(project="your-project-id", location="your-location")

def get_file_embeddings(file_path, chunk_size=300, chunk_overlap=20):
    # Determine the file type and use the appropriate loader
    if file_path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    document = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(document)

    # Initialize the Vertex AI Embeddings
    embeddings = VertexAIEmbeddings()

    # Generate embeddings for each chunk
    chunk_embeddings = embeddings.embed_documents([chunk.page_content for chunk in chunks])

    # Get the filename from the file_path
    filename = os.path.basename(file_path)

    return chunks, chunk_embeddings, filename

def store_embeddings_locally(chunks, chunk_embeddings, filename, index_file, metadata_file):
    # Convert embeddings to numpy array
    embeddings_array = np.array(chunk_embeddings).astype('float32')
    
    # Create a FAISS index
    dimension = len(chunk_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to the index
    index.add(embeddings_array)
    
    # Save the FAISS index to a file
    faiss.write_index(index, index_file)
    
    # Save metadata (chunks and filename) to a pickle file
    metadata = [{"chunk": chunk.page_content, "filename": filename} for chunk in chunks]
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Embeddings stored locally in {index_file} and {metadata_file}")

def load_embeddings_locally(index_file, metadata_file):
    # Load the FAISS index
    index = faiss.read_index(index_file)
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata

def search_similar_chunks(query_embedding, index, metadata, k=5):
    # Convert query embedding to numpy array
    query_array = np.array([query_embedding]).astype('float32')
    
    # Perform the search
    distances, indices = index.search(query_array, k)
    
    # Get the similar chunks with their filenames
    similar_chunks = [metadata[i] for i in indices[0]]
    
    return similar_chunks, distances[0]

if __name__ == "__main__":
    file_path = "path/to/your/file.txt"  # Can be .txt or .md
    index_file = "embeddings_index.faiss"
    metadata_file = "embeddings_metadata.pkl"
    
    chunks, embeddings, filename = get_file_embeddings(file_path)
    
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Filename: {filename}")

    # Store embeddings locally using FAISS
    store_embeddings_locally(chunks, embeddings, filename, index_file, metadata_file)

    # Load embeddings from local storage
    loaded_index, loaded_metadata = load_embeddings_locally(index_file, metadata_file)

    # Example: Search for similar chunks
    query_embedding = embeddings[0]  # Use the first embedding as an example query
    similar_chunks, distances = search_similar_chunks(query_embedding, loaded_index, loaded_metadata)

    print("Similar chunks:")
    for chunk_info, distance in zip(similar_chunks, distances):
        print(f"Distance: {distance:.4f}")
        print(f"Filename: {chunk_info['filename']}")
        print(f"Chunk: {chunk_info['chunk']}")
        print("---")