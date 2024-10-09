from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class NodeRelationshipExtraction:
    def __init__(self, llm):
        self.extraction_template = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract relevant nodes and their relationships from the following text about a production issue:
            {text}

            Focus on:
            1. People involved
            2. Systems affected
            3. Issue details (ID, severity, start time, resolution time)
            4. Root cause
            5. Resolution steps
            6. Impact

            Return the extracted information in the following JSON format:
            {{
                "nodes": [
                    {{"type": "Person", "id": "person_name", "properties": {{"role": "role_if_available"}}}}
                ],
                "relationships": [
                    {{"source": "source_node_id", "type": "RELATIONSHIP_TYPE", "target": "target_node_id", "properties": {{"additional_info": "if_any"}}}}
                ],
                "keywords": ["list", "of", "important", "keywords"],
                "embedding_text": "text to be used for generating embedding"
            }}
            """
        )
        self.extraction_chain = LLMChain(llm=llm, prompt=self.extraction_template, verbose=True)

    def extract_nodes_and_relationships(self, text: str) -> Dict:
        """Extract nodes, relationships, keywords, and embedding text from the given text."""
        response = self.extraction_chain.run(text=text)
        try:
            extracted_data = json.loads(response)
            return extracted_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from extraction response: {response}")
            return {"nodes": [], "relationships": [], "keywords": [], "embedding_text": ""}