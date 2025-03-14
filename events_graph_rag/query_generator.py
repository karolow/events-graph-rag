"""
Query generation and expansion logic.
"""

import re
from typing import Dict, List, Optional

from langchain_core.language_models.base import BaseLanguageModel

from events_graph_rag.config import logger
from events_graph_rag.llm_factory import LLMFactory
from events_graph_rag.prompts import query_expansion_prompt


class QueryGenerator:
    """Handles query generation and expansion for search operations."""

    def __init__(self, expansion_llm: Optional[BaseLanguageModel] = None):
        """Initialize the query generator."""
        self.expansion_llm = expansion_llm or LLMFactory.create_llm()
        logger.info("Initialized QueryGenerator with expansion LLM")

    def extract_cypher_query(self, text: str) -> str:
        """Extract the Cypher query from the LLM's response."""
        # Extract the query from between triple backticks
        cypher_pattern = r"```(?:cypher)?\s*([\s\S]+?)\s*```"
        match = re.search(cypher_pattern, text, re.IGNORECASE)

        if match:
            query = match.group(1).strip()
            logger.debug(f"Extracted Cypher query: {query}")
            return query

        # If no query found between backticks, log a warning and return empty string
        logger.warning("Could not extract a Cypher query from LLM response")
        return ""

    def extract_key_search_terms(self, query: str) -> str:
        """Extract key search terms from the user query for more focused vector search."""
        # Simply delegate to the LLM-based expansion, falling back to original query if needed
        try:
            return self._llm_expand_query(query)
        except Exception as e:
            logger.warning(f"Error in query expansion: {str(e)}")
            return query

    def is_count_query(self, cypher_query: str, results: List[Dict]) -> bool:
        """Determine if a query is a COUNT query."""
        # Check if the query contains COUNT
        has_count_keyword = "COUNT" in cypher_query.upper()

        # Check if results only contain count values
        has_count_results = (
            all(
                key.lower().endswith("count")
                for result in results
                for key in result.keys()
                if not key.lower().startswith("event")
            )
            if results
            else False
        )

        return has_count_keyword and has_count_results

    def _llm_expand_query(self, query: str) -> str:
        """Use an LLM to expand the query with semantically related terms.

        This method sends the query to an LLM and asks it to extract and expand
        key search terms with related concepts for better semantic matching.
        """
        if not self.expansion_llm:
            logger.warning("No expansion LLM available for query expansion")
            return query

        # Use the imported query expansion prompt
        formatted_prompt = query_expansion_prompt.format(query=query)

        # Get the expanded query from the LLM
        response = self.expansion_llm.invoke(formatted_prompt)

        # Extract content from the response
        if hasattr(response, "content"):
            expanded_query = response.content.strip()
        else:
            expanded_query = str(response).strip()

        logger.info(f"LLM expanded query: '{query}' → '{expanded_query}'")
        return expanded_query
