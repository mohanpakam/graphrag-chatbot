from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from database_manager import DatabaseManager
import yaml
import requests

app = FastAPI()

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
db_manager = DatabaseManager(config['database_path'])

class ChatRequest(BaseModel):
    message: str

def get_embedding(text):
    response = requests.post(
        config['local_ollama_chat_api_url'],
        json={"model": config['local_ollama_model'], "prompt": text}
    )
    if response.status_code == 200:
        return response.json()['embedding']
    else:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    
    with db_manager.connect() as conn:
        # Convert user input to embedding
        user_embedding = get_embedding(user_input)
        
        # Perform similarity search
        cursor = conn.execute('''
            SELECT content, vec_dot(embedding, ?) AS similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT 5
        ''', (np.array(user_embedding, dtype=np.float32).tobytes(),))
        
        similar_docs = cursor.fetchall()
        
        # Process similar documents and generate a response
        # This is a placeholder. You should implement your own logic here.
        response = f"Based on your input, I found {len(similar_docs)} relevant documents. Here's a summary:\n\n"
        for doc, similarity in similar_docs:
            response += f"- {doc[:100]}... (similarity: {similarity:.2f})\n"
        
        return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)