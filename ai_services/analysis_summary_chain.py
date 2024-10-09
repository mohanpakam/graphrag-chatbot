from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class AnalysisSummaryChain:
    def __init__(self, llm):
        self.analysis_summary_template = PromptTemplate(
            input_variables=["query", "result"],
            template="""
            Given the following query and its result about production issues:
            Query: {query}
            Result: {result}

            Provide a comprehensive analysis summary, considering:
            1. Key insights from the data
            2. Trends or patterns observed
            3. Notable people or systems involved
            4. Common root causes or resolution steps
            5. Recommendations for improvement

            Analysis Summary:
            """
        )
        self.analysis_summary_chain = LLMChain(llm=llm, prompt=self.analysis_summary_template, verbose=True)

    def generate_analysis_summary(self, query: str, result: str) -> str:
        """Generate an analysis summary based on the query and result."""
        response = self.analysis_summary_chain.run(query=query, result=result)
        return response.strip()