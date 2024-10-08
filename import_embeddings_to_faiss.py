import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
import faiss
import numpy as np
import pickle
from dataclasses import dataclass

class EmbedFileToFaiss:
    def __init__(self, project_id, location, credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        aiplatform.init(project=project_id, location=location)
        self.embeddings = VertexAIEmbeddings()
        self.index = None
        self.metadata = []

    def get_file_embeddings(self, file_path, chunk_size=300, chunk_overlap=20):
        if file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(document)
        chunk_embeddings = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
        filename = os.path.basename(file_path)
        return chunks, chunk_embeddings, filename

    def add_to_index(self, chunks, chunk_embeddings, filename):
        embeddings_array = np.array(chunk_embeddings).astype('float32')
        if self.index is None:
            dimension = len(chunk_embeddings[0])
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        self.metadata.extend([{"chunk": chunk.page_content, "filename": filename} for chunk in chunks])

    def process_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.txt', '.md')):
                file_path = os.path.join(folder_path, filename)
                chunks, embeddings, filename = self.get_file_embeddings(file_path)
                self.add_to_index(chunks, embeddings, filename)
                print(f"Processed {filename}")

    def save_index(self, index_file, metadata_file):
        faiss.write_index(self.index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {index_file} and metadata saved to {metadata_file}")

    @staticmethod
    def load_index(index_file, metadata_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata

    @staticmethod
    def search_similar_chunks(query_embedding, index, metadata, k=5):
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_array, k)
        similar_chunks = [
            SearchResult(
                distance=float(distances[0][i]),
                filename=metadata[idx]['filename'],
                chunk_content=metadata[idx]['chunk']
            )
            for i, idx in enumerate(indices[0])
        ]
        return similar_chunks

@dataclass
class SearchResult:
    distance: float
    filename: str
    chunk_content: str

    def __str__(self):
        return f"Distance: {self.distance:.4f}\nFilename: {self.filename}\nChunk: {self.chunk_content}"

if __name__ == "__main__":
    project_id = "your-project-id"
    location = "your-location"
    credentials_path = "path/to/your/google-cloud-credentials.json"
    
    embedder = EmbedFileToFaiss(project_id, location, credentials_path)
    
    folder_path = "path/to/your/text/files/folder"
    embedder.process_folder(folder_path)
    
    index_file = "embeddings_index.faiss"
    metadata_file = "embeddings_metadata.pkl"
    embedder.save_index(index_file, metadata_file)
    
    # Example: Load the index and search for similar chunks
    loaded_index, loaded_metadata = EmbedFileToFaiss.load_index(index_file, metadata_file)
    
    # Assuming we have a query embedding (you'd need to generate this from a query text)
    query_embedding = embedder.embeddings.embed_query("Your query text here")
    
    similar_chunks = EmbedFileToFaiss.search_similar_chunks(query_embedding, loaded_index, loaded_metadata)
    
    print("Similar chunks:")
    for result in similar_chunks:
        print(result)
        print("---")
