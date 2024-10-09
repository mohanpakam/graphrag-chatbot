import json
from typing import Dict, Any
from langchain.schema import Document
from ai_services import get_langchain_ai_service
from config import LoggerConfig

logger = LoggerConfig.setup_logger(__name__)
config = LoggerConfig.load_config()

class StructuredDataExtractor:
    def __init__(self):
        self.ai_service = get_langchain_ai_service(config['ai_service'])

    def extract_structured_data(self, document: Document) -> Dict[str, Any]:
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
            logger.info(f"Successfully extracted structured data: {json.dumps(structured_data, indent=2)}")
            return structured_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from AI response: {extraction_result}")
            return {}