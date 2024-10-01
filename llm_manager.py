from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

class LLMManager:
    def __init__(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = Ollama(
            model="llama3.2",
            callback_manager=callback_manager,
        )

    def generate_response(self, prompt: str) -> str:
        return self.llm.invoke(prompt)