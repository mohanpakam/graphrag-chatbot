from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import yaml
from openai import AzureOpenAI

app = FastAPI()

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

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(input: ChatInput):
    # Get embedding for the input message
    response = client.embeddings.create(input=input.message, model=config['embedding_model'])
    query_embedding = response.data[0].embedding

    # Search for relevant documents
    conn = sqlite3.connect(config['database_path'])
    conn.enable_load_extension(True)
    conn.load_extension("sqlite_vss")

    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT content, distance
        FROM documents
        ORDER BY vss_search(embedding, ?) ASC
        LIMIT {config['top_k']}
    ''', (sqlite3.Binary(bytes(query_embedding)),))

    relevant_docs = cursor.fetchall()
    conn.close()

    # Prepare context from relevant documents
    context = "\n".join([doc[0] for doc in relevant_docs])

    # Generate response using Azure OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the user's question."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {input.message}"}
    ]
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Replace with your deployed model name
        messages=messages
    )

    return {"response": completion.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)