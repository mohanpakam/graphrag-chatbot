from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import SQLiteVectorStore
from langchain.llms import HuggingFaceInference
from langchain.chains import RetrievalQAChain
from llama_cpp import Llama
import networkx as nx

def process_text_files(text_files):
    """Processes a list of text files, converts them to graphs, and stores the embeddings in a vector store.

    Args:
        text_files: A list of paths to the text files.

    Returns:
        A list of graph objects.
    """

    graphs = []
    for file_path in text_files:
        with open(file_path, "r") as f:
            text = f.read()

        # Split the text into chunks
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_text(text)

        # Generate embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedding_model.embed_documents(chunks)

        # Create a vector store
        vectorstore = SQLiteVectorStore(embedding_model=embedding_model, database_path="your_database.db")
        vectorstore.add_documents([Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)])

        # Create a graph
        G = nx.Graph()
        for i, result in enumerate(vectorstore.similarity_search(query="What is the main topic of the text?", k=5)):
            G.add_node(f"Chunk {i}")
            for j, other_result in enumerate(vectorstore.similarity_search(query="What is the main topic of the text?", k=5)):
                if i != j:
                    similarity = vectorstore.similarity(results[i], results[j])
                    if similarity > 0.7:
                        G.add_edge(f"Chunk {i}", f"Chunk {j}", weight=similarity)

        graphs.append(G)

    return graphs

# Example usage
text_files = ["file1.txt", "file2.txt"]
graphs = process_text_files(text_files)

# Print the graphs
for i, graph in enumerate(graphs):
    print(f"Graph {i}:")
    print(graph.nodes())
    print(graph.edges())