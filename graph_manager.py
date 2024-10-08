import logging
from typing import List, Dict
from langchain.schema import Document
from llm_graph_transformer import LLMGraphTransformer
import numpy as np
from ai_service import get_ai_service
import json
from sklearn.feature_extraction.text import TfidfVectorizer

class GraphManager:
    def __init__(self, db_path: str):
        self.graph_transformer = LLMGraphTransformer(db_path)
        self.graph_transformer.init_database()
        self.ai_service = get_ai_service()
        self.tfidf = TfidfVectorizer(max_features=10)  # Extract top 10 keywords

    def extract_entities_and_relationships(self, document: Document):
        # Assume document contains the full text of a production issue report
        prompt = f"""
        Extract the following information from this production issue report:
        1. Issue ID
        2. Systems Affected (e.g., databases, servers, applications)
        3. People Involved (e.g., engineers, managers)
        4. Timestamps (e.g., issue start time, resolution time)
        5. Severity Level
        6. Root Cause
        7. Resolution Steps
        8. Impact (e.g., downtime, data loss)
        9. Relationships between these entities

        Text: {document.page_content}

        Format the output as JSON with the following structure:
        {{
            "issue_id": "string",
            "systems_affected": ["string"],
            "people_involved": ["string"],
            "timestamps": {{"start": "string", "end": "string"}},
            "severity_level": "string",
            "root_cause": "string",
            "resolution_steps": ["string"],
            "impact": "string",
            "relationships": [
                {{"source": "string", "type": "string", "target": "string"}}
            ]
        }}
        """
        extraction_result = self.ai_service.generate_response(prompt, "")
        structured_data = json.loads(extraction_result)
        
        return structured_data

    def process_document(self, document: Document, doc_id: int):
        # Extract entities and relationships from the full document
        structured_data = self.extract_entities_and_relationships(document)
        
        # Store graph data for the whole document
        self.store_graph_data(document, structured_data, doc_id)

    def store_graph_data(self, document: Document, structured_data: dict, doc_id: int):
        # Store issue node
        issue_id = structured_data.get('issue_id', f"ISSUE_{doc_id}")
        self.graph_transformer.store_node(doc_id, 'Issue', {
            'id': issue_id,
            'severity': structured_data.get('severity_level'),
            'root_cause': structured_data.get('root_cause'),
            'impact': structured_data.get('impact')
        })

        # Store systems affected
        for system in structured_data.get('systems_affected', []):
            system_id = f"SYSTEM_{system.replace(' ', '_')}"
            self.graph_transformer.store_node(doc_id, 'System', {'id': system_id, 'name': system})
            self.graph_transformer.store_relationship(doc_id, issue_id, system_id, 'AFFECTS')

        # Store people involved
        for person in structured_data.get('people_involved', []):
            person_id = f"PERSON_{person.replace(' ', '_')}"
            self.graph_transformer.store_node(doc_id, 'Person', {'id': person_id, 'name': person})
            self.graph_transformer.store_relationship(doc_id, person_id, issue_id, 'INVOLVED_IN')

        # Store timestamps
        timestamps = structured_data.get('timestamps', {})
        for time_type, timestamp in timestamps.items():
            time_id = f"TIME_{time_type.upper()}_{doc_id}"
            self.graph_transformer.store_node(doc_id, 'Timestamp', {'id': time_id, 'type': time_type, 'value': timestamp})
            self.graph_transformer.store_relationship(doc_id, issue_id, time_id, 'HAS_TIMESTAMP')

        # Store resolution steps
        for i, step in enumerate(structured_data.get('resolution_steps', [])):
            step_id = f"STEP_{i+1}_{doc_id}"
            self.graph_transformer.store_node(doc_id, 'ResolutionStep', {'id': step_id, 'description': step})
            self.graph_transformer.store_relationship(doc_id, issue_id, step_id, 'HAS_RESOLUTION_STEP')

        # Extract and store keywords
        keywords = self.extract_keywords(document.page_content)
        for keyword in keywords:
            keyword_id = f"KEYWORD_{keyword.replace(' ', '_')}"
            self.graph_transformer.store_node(doc_id, 'Keyword', {'id': keyword_id, 'word': keyword})
            self.graph_transformer.store_relationship(doc_id, issue_id, keyword_id, 'HAS_KEYWORD')

        # Store additional relationships
        for rel in structured_data.get('relationships', []):
            self.graph_transformer.store_relationship(doc_id, rel['source'], rel['target'], rel['type'])

    def extract_keywords(self, text: str) -> List[str]:
        self.tfidf.fit([text])
        feature_names = self.tfidf.get_feature_names_out()
        return feature_names.tolist()

    def get_subgraph_for_documents(self, doc_ids: List[int]) -> Dict:
        subgraph = {
            'nodes': [],
            'relationships': []
        }
        for doc_id in doc_ids:
            nodes = self.graph_transformer.get_nodes(doc_id)
            relationships = self.graph_transformer.get_relationships(doc_id)
            subgraph['nodes'].extend(nodes)
            subgraph['relationships'].extend(relationships)
        return subgraph

    def get_embedding(self, text: str) -> List[float]:
        return self.ai_service.get_embedding(text)

    def get_relevant_nodes(self, doc_id: int, query_type: str) -> List[Dict]:
        # Implement logic to retrieve relevant nodes based on query type
        if query_type == "specific_issue":
            return self.graph_transformer.get_nodes_by_type(doc_id, ['Issue', 'Person', 'System'])
        elif query_type == "root_cause":
            return self.graph_transformer.get_nodes_by_type(doc_id, ['Issue', 'RootCause'])
        # ... (add more query types as needed)
        else:
            return self.graph_transformer.get_nodes(doc_id)