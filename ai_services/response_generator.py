from typing import List, Dict
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ResponseGenerator:
    def __init__(self, llm, embeddings, callback_manager=None):
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=llm,
            memory=self.memory,
            verbose=True,
            callback_manager=callback_manager
        )

    def generate_response(self, prompt: str, context: str) -> str:
        """Generate a response given a prompt and context."""
        full_prompt = f"Context: {context}\n\nHuman: {prompt}"
        response = self.conversation.predict(input=full_prompt)
        return response

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.memory.clear()