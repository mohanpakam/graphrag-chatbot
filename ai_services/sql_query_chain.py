from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

class SQLQueryChain:
    def __init__(self, llm):
        
        
            #         Consider the following aspects based on the schema:
            # - People involved in the issue
            # - Systems affected
            # - Timestamps (start and resolution times)
            # - Root cause
            # - Resolution steps
            # - Impact of the issue
            
            #Given the Following Query to help with JOINS:
            # SELECT lii.llm_json_response, lii.severity_level,     llp.name AS person_name,     les.name AS system_name,     ltstart.timestamp AS start_time,     ltend.timestamp AS end_time,     lii.root_cause,     lrs.description AS resolution_step FROM     llm_extracted_issues lii  JOIN     llm_extracted_issue_people lisp ON lii.id = lisp.issue_id AND llp.id = lisp.person_id  JOIN     llm_extracted_people llp ON lisp.person_id = llp.id  JOIN     llm_extracted_issue_systems liis ON lii.id = liis.issue_id  JOIN 	llm_extracted_systems les ON liis.system_id = les.id  JOIN     llm_extracted_relationships lerp ON les.name = lerp.source OR les.name = lerp.target  left JOIN     llm_extracted_timestamps ltstart ON lii.id = ltstart.issue_id AND ltstart.type = 'start'  left JOIN     llm_extracted_timestamps ltend ON lii.id = ltend.issue_id AND ltend.type = 'end'  JOIN     llm_extracted_resolution_steps lrs ON lrs.issue_id = lii.issue_id WHERE     lii.issue_id = '1';
            # 
            
        self.schema_info = self.load_schema_from_json()
        self.sql_query_template = PromptTemplate(
            input_variables=["query"],
            template="""
            Given the following database schema in JSON format. :
            {schema}
            Write a SQL query to only llm_json_response column that answers the following natural language query.ONLY SQL Query do not include any explanation. If the query is not possible to answer using the given schema, return "Query not possible to answer with the given schema.". If the query is ambiguous, return "Query is ambiguous. Please provide more details.".
            {query}
            SQL Query:
            """
        )
        self.sql_query_chain = LLMChain(llm=llm, prompt=self.sql_query_template, verbose=True)

    def load_schema_from_json(self) -> str:
        with open('structured/init_database.json', 'r') as f:
            schema = json.load(f)
        return json.dumps(schema)

    def generate_sql_query(self, natural_language_query: str) -> str:
        """Generate an SQL query based on the natural language query."""
        response = self.sql_query_chain.run(query=natural_language_query, schema=self.schema_info)
        return response.strip()