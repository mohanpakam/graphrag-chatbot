from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class TrendAxesChain:
    def __init__(self, llm):
        self.trend_axes_template = PromptTemplate(
            input_variables=["query", "columns"],
            template="""
            Given the following query about production issues and available columns:
            Query: {query}
            Available columns: {columns}

            Determine the most appropriate columns for the x and y axes of a trend graph.
            Consider time-based columns for the x-axis and quantitative measures for the y-axis.
            Return your answer in the format:
            x: <column_name>
            y: <column_name>

            Explanation: [Brief explanation of your choice]
            """
        )
        self.trend_axes_chain = LLMChain(llm=llm, prompt=self.trend_axes_template, verbose=True)

    def determine_trend_axes(self, query: str, available_columns: List[str]) -> Dict[str, str]:
        """Determine the appropriate axes for trend visualization."""
        response = self.trend_axes_chain.run(query=query, columns=", ".join(available_columns))
        axes = {}
        explanation = ""
        for line in response.strip().split('\n'):
            if line.startswith('x:') or line.startswith('y:'):
                axis, column = line.split(': ')
                axes[axis.strip()] = column.strip()
            elif line.startswith('Explanation:'):
                explanation = line.replace('Explanation:', '').strip()
        return {"axes": axes, "explanation": explanation}