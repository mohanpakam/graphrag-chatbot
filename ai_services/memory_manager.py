from typing import List, Dict

class MemoryManager:
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []

    def add_interaction(self, query: str, response: str) -> List[Dict[str, str]]:
        self.chat_history.append({"user": query})
        self.chat_history.append({"ai": response})
        return self.chat_history

    def get_chat_history(self) -> List[Dict[str, str]]:
        return self.chat_history

    def clear_chat_history(self):
        self.chat_history = []