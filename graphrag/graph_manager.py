import logging
from typing import List, Dict
from langchain.schema import Document
from graphrag import LLMGraphTransformer
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

class GraphManager:
    def __init__(self, db_path: str, ai_sevice):
        self.graph_transformer = LLMGraphTransformer(db_path)
        self.graph_transformer.init_database()
        self.ai_service = ai_sevice

    def extract_entities_and_relationships(self, document: Document):
        print(f"To Extract the Entities and Relations \n {document.page_content}")
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

        Respond ONLY with a JSON object in the following structure, without any additional text:
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
                
        try:
            structured_data = json.loads(extraction_result)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from AI response: {extraction_result}")
            # Provide a default structure if JSON parsing fails
            structured_data = {
                "issue_id": "unknown",
                "systems_affected": [],
                "people_involved": [],
                "timestamps": {"start": "", "end": ""},
                "severity_level": "unknown",
                "root_cause": "unknown",
                "resolution_steps": [],
                "impact": "unknown",
                "relationships": []
            }
        print(f"Extracted Result is {extraction_result} and extracted json is {structured_data}")
        return structured_data

    def process_document(self, document: Document, doc_id: int):
        structured_data = self.extract_entities_and_relationships(document)
        self.store_graph_data(document, structured_data, doc_id)

    def store_graph_data(self, document: Document, structured_data: dict, doc_id: int):
        dummy_embedding = np.zeros(768, dtype=np.float32)  # Adjust the size if needed

        issue_id = structured_data.get('issue_id', f"ISSUE_{doc_id}")
        issue_data = {
            'id': issue_id,
            'severity': structured_data.get('severity_level'),
            'root_cause': structured_data.get('root_cause'),
            'impact': structured_data.get('impact')
        }
        self.graph_transformer.store_data({
            'filename': document.metadata.get('filename', 'unknown'),
            'chunk_index': 0,  # Assuming we're storing the whole document as one chunk
            'content': json.dumps(issue_data),
            'embedding': dummy_embedding,
            'metadata': {'type': 'Issue'}
        })

        for system in structured_data.get('systems_affected', []):
            system_id = f"SYSTEM_{system.replace(' ', '_')}"
            system_data = {'id': system_id, 'name': system}
            self.graph_transformer.store_data({
                'filename': document.metadata.get('filename', 'unknown'),
                'chunk_index': 0,
                'content': json.dumps(system_data),
                'embedding': dummy_embedding,
                'metadata': {'type': 'System'}
            })
            self.graph_transformer.store_relationship(issue_id, system_id, 'AFFECTS')

        for person in structured_data.get('people_involved', []):
            person_id = f"PERSON_{person.replace(' ', '_')}"
            person_data = {'id': person_id, 'name': person}
            self.graph_transformer.store_data({
                'filename': document.metadata.get('filename', 'unknown'),
                'chunk_index': 0,
                'content': json.dumps(person_data),
                'embedding': dummy_embedding,
                'metadata': {'type': 'Person'}
            })
            self.graph_transformer.store_relationship(person_id, issue_id, 'INVOLVED_IN')

        timestamps = structured_data.get('timestamps', {})
        for time_type, timestamp in timestamps.items():
            time_id = f"TIME_{time_type.upper()}_{doc_id}"
            time_data = {'id': time_id, 'type': time_type, 'value': timestamp}
            self.graph_transformer.store_data({
                'filename': document.metadata.get('filename', 'unknown'),
                'chunk_index': 0,
                'content': json.dumps(time_data),
                'embedding': dummy_embedding,
                'metadata': {'type': 'Timestamp'}
            })
            self.graph_transformer.store_relationship(issue_id, time_id, 'HAS_TIMESTAMP')

        for i, step in enumerate(structured_data.get('resolution_steps', [])):
            step_id = f"STEP_{i+1}_{doc_id}"
            step_data = {'id': step_id, 'description': step}
            self.graph_transformer.store_data({
                'filename': document.metadata.get('filename', 'unknown'),
                'chunk_index': 0,
                'content': json.dumps(step_data),
                'embedding': dummy_embedding,
                'metadata': {'type': 'ResolutionStep'}
            })
            self.graph_transformer.store_relationship(issue_id, step_id, 'HAS_RESOLUTION_STEP')

        # Keywords are no longer stored in the graph
        # keywords = self.extract_keywords(document.page_content)
        # for keyword in keywords:
        #     keyword_id = f"KEYWORD_{keyword.replace(' ', '_')}"
        #     keyword_data = {'id': keyword_id, 'word': keyword}
        #     self.graph_transformer.store_data({
        #         'filename': document.metadata.get('filename', 'unknown'),
        #         'chunk_index': 0,
        #         'content': json.dumps(keyword_data),
        #         'embedding': dummy_embedding,
        #         'metadata': {'type': 'Keyword'}
        #     })
        #     self.graph_transformer.store_relationship(issue_id, keyword_id, 'HAS_KEYWORD')

        for rel in structured_data.get('relationships', []):
            self.graph_transformer.store_relationship(rel['source'], rel['target'], rel['type'])

    # def extract_keywords(self, text: str) -> List[str]:
    #     self.tfidf.fit([text])
    #     feature_names = self.tfidf.get_feature_names_out()
    #     return feature_names.tolist()

    def get_subgraph_for_documents(self, doc_ids: List[int]) -> Dict:
        subgraph = {
            'nodes': [],
            'relationships': []
        }
        for doc_id in doc_ids:
            # Assuming LLMGraphTransformer has a method to get all data for a document
            document_data = self.graph_transformer.get_document_data(doc_id)
            if document_data:
                subgraph['nodes'].extend(self.extract_nodes_from_document(document_data))
                subgraph['relationships'].extend(self.graph_transformer.get_relationships(doc_id))
        return subgraph

    def extract_nodes_from_document(self, document_data: Dict) -> List[Dict]:
        nodes = []
        content = json.loads(document_data['content'])
        for key, value in content.items():
            if isinstance(value, str):
                nodes.append({
                    'id': f"{key}_{document_data['id']}",
                    'type': key,
                    'properties': {'value': value}
                })
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    nodes.append({
                        'id': f"{key}_{i}_{document_data['id']}",
                        'type': key,
                        'properties': {'value': item}
                    })
        return nodes

    def get_relevant_nodes(self, doc_id: int, query_type: str) -> List[Dict]:
        document_data = self.graph_transformer.get_document_data(doc_id)
        if not document_data:
            return []
        
        nodes = self.extract_nodes_from_document(document_data)
        
        if query_type == "specific_issue":
            return [node for node in nodes if node['type'] in ['Issue', 'Person', 'System']]
        elif query_type == "root_cause":
            return [node for node in nodes if node['type'] in ['Issue', 'root_cause']]
        else:
            return nodes

    def get_embedding(self, text: str) -> List[float]:
        return self.ai_service.get_embedding(text)