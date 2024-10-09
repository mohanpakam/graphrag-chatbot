from langchain.prompts import PromptTemplate
from typing import List
from ai_services import get_langchain_ai_service


class QueryClassifier:
    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.classification_prompt = PromptTemplate.from_template(
            "Classify the following query into one of these categories:\n"
            "1. Specific Issue Query: Asks about a particular production issue.\n"
            "2. Trend Analysis Query: Asks about patterns or trends across multiple issues.\n"
            "3. General Question: A broad question about production issues or processes.\n\n"
            "Query: {query}\n\n"
            "Classification:"
        )

    def classify_query(self, query: str, conversation_context: List[str] = None) -> str:
        if conversation_context:
            context = "\n".join(conversation_context[-3:])  # Consider last 3 exchanges
            full_query = f"Previous conversation:\n{context}\n\nCurrent query: {query}"
        else:
            full_query = query

        prompt = self.classification_prompt.format(query=full_query)
        response = self.ai_service.generate_response(prompt, "")
        
        # Add logic to detect transitions
        if "specific issue" in response.lower():
            return "specific_issue"
        elif "trend analysis" in response.lower():
            return "trend_analysis"
        elif "general question" in response.lower():
            if conversation_context and "specific issue" in self.classify_query(conversation_context[-1]):
                return "follow_up_general"
            return "general_question"
        else:
            return "general_question"