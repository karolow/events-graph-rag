"""
Hybrid search implementation combining graph and vector search for cultural events.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from events_graph_rag.config import (
    LLM_CONFIG,
    NEO4J_CONFIG,
    RERANKER_CONFIG,
    SEARCH_CONFIG,
    VECTOR_INDEX_CONFIG,
    logger,
)
from events_graph_rag.document_processor import DocumentProcessor
from events_graph_rag.llm_factory import LLMFactory
from events_graph_rag.neo4j_client import Neo4jClient
from events_graph_rag.prompts import cypher_prompt, qa_prompt
from events_graph_rag.query_generator import QueryGenerator
from events_graph_rag.vector_client import VectorClient


class HybridSearch:
    """Hybrid search implementation combining graph and vector search for cultural events."""

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        cypher_llm: Optional[BaseLanguageModel] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        vector_client: Optional[VectorClient] = None,
        query_generator: Optional[QueryGenerator] = None,
        document_processor: Optional[DocumentProcessor] = None,
        verbose: bool = False,
    ):
        """Initialize the hybrid search."""
        # Set up logging
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Initialize language models
        if llm:
            self.llm = llm
        else:
            self.llm = LLMFactory.create_llm()
        logger.info(f"Initialized QA LLM with provider: {LLM_CONFIG.get('provider')}")

        if cypher_llm:
            self.cypher_llm = cypher_llm
        else:
            # Use a dedicated Cypher LLM by default
            self.cypher_llm = LLMFactory.create_cypher_llm()
        logger.info(
            f"Initialized Cypher LLM with provider: {LLM_CONFIG.get('cypher', {}).get('provider', LLM_CONFIG.get('provider'))}"
        )

        # Initialize clients
        self.neo4j_client = neo4j_client or Neo4jClient()
        logger.info(f"Initialized Neo4j client: {NEO4J_CONFIG['url']}")

        self.vector_client = vector_client or VectorClient()
        logger.info(
            f"Initialized Vector client with index: {VECTOR_INDEX_CONFIG['index_name']}"
        )

        # Initialize helpers
        self.query_generator = query_generator or QueryGenerator(expansion_llm=self.llm)
        logger.info("Initialized QueryGenerator with expansion LLM")

        self.document_processor = document_processor or DocumentProcessor()
        logger.info("Initialized DocumentProcessor")

        # Initialize reranker configuration
        if not RERANKER_CONFIG:
            raise ValueError("RERANKER_CONFIG must be provided")

        if "model" not in RERANKER_CONFIG:
            raise ValueError("RERANKER_CONFIG must contain 'model'")

        if "top_k" not in RERANKER_CONFIG:
            raise ValueError("RERANKER_CONFIG must contain 'top_k'")

        self.reranker_config = RERANKER_CONFIG

        logger.info(
            f"Initialized Reranker config: {self.reranker_config['model']} with top_k={self.reranker_config['top_k']}"
        )

        # Set up chain components
        self._setup_chain_components()

        logger.info("Hybrid Search initialized successfully")

    def _setup_chain_components(self) -> None:
        """Set up the chain components for the hybrid search."""
        logger.info("Setting up chain components")

        # Create the Cypher generation chain using the imported prompt
        self.cypher_chain = (
            RunnablePassthrough.assign(
                schema=lambda _: self.neo4j_client.get_filtered_schema()
            )
            | cypher_prompt
            | self.cypher_llm
            | StrOutputParser()
        )
        logger.info("Cypher generation chain created")

        # Create the QA chain using the imported prompt
        self.qa_chain = (
            RunnablePassthrough.assign(
                schema=lambda _: self.neo4j_client.get_filtered_schema()
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("QA chain created")

    def search(self, query: str) -> Dict[str, Any]:
        """
        Execute a hybrid search combining graph and vector search for cultural events.

        The search follows this process:
        1. Generate a Cypher query using LLM
        2. Execute graph search using the Cypher query
        3. Determine if vector search is needed based on graph results:
           - If graph search returns sufficient results (>= min_graph_results), skip vector search
           - If graph search returns few or no results, perform vector search
        4. If vector search is performed, exclude events already found by graph search
        5. Format results from both searches
        6. Generate a final answer using the QA LLM

        Args:
            query: The user query string

        Returns:
            Dict containing search results and metadata
        """
        start_time = time.time()
        logger.info(f"Starting hybrid search for query: '{query}'")

        # Initialize result structure
        result = {
            "query": query,
            "graph_results": [],
            "vector_results": [],
            "answer": "",
            "metadata": {
                "graph_search_time": 0,
                "vector_search_time": 0,
                "total_time": 0,
            },
        }

        # Generate and execute Cypher query
        cypher_start = time.time()
        cypher_query = self._generate_cypher_query(query)
        cypher_generation_time = time.time() - cypher_start
        logger.info(
            f"Cypher query generation completed in {cypher_generation_time:.2f}s"
        )

        # Execute graph search if we have a valid Cypher query
        graph_results = []
        event_ids = []
        need_vector_search = True

        if cypher_query and cypher_query.strip():
            graph_search_start = time.time()
            graph_results = self._execute_graph_search(cypher_query)
            graph_search_time = time.time() - graph_search_start
            logger.info(
                f"Graph search completed in {graph_search_time:.2f}s, returned {len(graph_results)} results"
            )

            # Extract event IDs from graph results
            event_ids = self._extract_event_ids(graph_results)
            logger.info(f"Extracted {len(event_ids)} event IDs from graph results")

            # Log event names for debugging
            self._log_event_names(graph_results)

            # Determine if we need vector search
            need_vector_search = self._need_vector_search(graph_results)
            result["metadata"]["graph_search_time"] = graph_search_time
            result["graph_results"] = graph_results
        else:
            logger.info("No valid Cypher query generated, skipping graph search")

        # Execute vector search if needed
        if need_vector_search:
            vector_search_start = time.time()

            # Get the configured top_k value from SEARCH_CONFIG
            if "vector_top_k" not in SEARCH_CONFIG:
                raise ValueError("SEARCH_CONFIG must contain 'vector_top_k'")

            top_k = SEARCH_CONFIG["vector_top_k"]
            logger.info(f"Using vector_top_k={top_k} from SEARCH_CONFIG")

            vector_results = self._execute_vector_search(
                query, top_k=top_k, exclude_ids=event_ids
            )
            vector_search_time = time.time() - vector_search_start
            logger.info(f"Vector search process completed in {vector_search_time:.2f}s")

            result["metadata"]["vector_search_time"] = vector_search_time
            result["vector_results"] = vector_results
        else:
            logger.info("Skipping vector search as graph results are sufficient")

        # Format results
        format_start = time.time()
        graph_text = self._format_graph_results(graph_results)
        vector_text = self._format_vector_results(result.get("vector_results", []))
        format_time = time.time() - format_start
        logger.info(f"Results formatting completed in {format_time:.2f}s")

        # Generate answer
        answer_start = time.time()
        answer = self._generate_answer(query, graph_text, vector_text)
        answer_time = time.time() - answer_start
        logger.info(f"Answer generation completed in {answer_time:.2f}s")

        result["answer"] = answer
        result["metadata"]["total_time"] = time.time() - start_time
        logger.info(
            f"Total hybrid search completed in {result['metadata']['total_time']:.2f}s"
        )

        return result

    def _log_event_names(self, events: List[Dict[str, Any]]) -> None:
        """Log event names and IDs for debugging purposes."""
        if not events:
            logger.info("No events to log")
            return

        logger.info("Event IDs and names:")
        for i, event in enumerate(events[:5]):  # Log only first 5 events
            # Use the Neo4jClient helper method to extract event info
            event_info = self.neo4j_client.extract_event_info(event)

            # Format additional information if available
            additional_info = ""
            for key, value in event_info.items():
                if key not in ["id", "name"] and value and not isinstance(value, dict):
                    additional_info += f" - {key.capitalize()}: {value}"

            logger.info(
                f"  {i + 1}. ID: {event_info['id']} - Name: {event_info['name']}{additional_info}"
            )

        if len(events) > 5:
            logger.info(f"  ... and {len(events) - 5} more events")

    def _extract_event_ids(self, graph_results: List[Dict[str, Any]]) -> List[str]:
        """Extract event IDs from graph search results."""
        # Use the Neo4jClient's extract_event_ids method
        return self.neo4j_client.extract_event_ids(graph_results)

    def _log_top_vector_results(
        self, results: List[Document], prefix: str = "Top vector"
    ) -> None:
        """Log top vector search results for debugging purposes."""
        if not results:
            logger.info(f"{prefix} results: None")
            return

        logger.info(f"{prefix} results:")
        for i, doc in enumerate(results):
            # Access metadata correctly
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            # Try to get score from different possible fields
            score = metadata.get("reranker_score", metadata.get("vector_score", "N/A"))
            event_id = metadata.get("event_id", "unknown_id")
            event_name = metadata.get("name", "unnamed_event")
            participants = metadata.get("number_of_participants", "N/A")

            # Format score properly based on its type
            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)

            logger.info(
                f"  {i + 1}. Score: {score_str} - ID: {event_id} - Name: {event_name} - Participants: {participants}"
            )

    def _generate_cypher_query(self, query: str) -> str:
        """Generate a Cypher query from the user query."""
        logger.info(f"Generating Cypher query for: '{query}'")
        start_time = time.time()

        try:
            cypher_response = self.cypher_chain.invoke({"query": query})
            cypher_query = self.query_generator.extract_cypher_query(cypher_response)

            if cypher_query:
                generation_time = time.time() - start_time
                logger.info(
                    f"Generated Cypher query in {generation_time:.2f}s: {cypher_query}"
                )
                return cypher_query
            else:
                logger.warning("Failed to extract Cypher query from LLM response")
                return ""
        except Exception as e:
            logger.error(f"Error generating Cypher query: {str(e)}")
            return ""

    def _execute_graph_search(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a graph search using the Neo4j client."""
        logger.info("Executing graph search with Cypher query")
        start_time = time.time()

        try:
            results = self.neo4j_client.query(cypher_query)
            query_time = time.time() - start_time
            logger.info(
                f"Neo4j query executed in {query_time:.2f}s, returned {len(results)} raw results"
            )

            # Filter out embedding attributes
            filter_start = time.time()
            filtered_results = self.neo4j_client.filter_results_attributes(results)
            filter_time = time.time() - filter_start
            logger.info(
                f"Results filtered in {filter_time:.2f}s, {len(filtered_results)} results after filtering"
            )

            return filtered_results
        except Exception as e:
            logger.error(f"Error executing graph search: {str(e)}")
            return []

    def _need_vector_search(self, graph_results: List[Dict[str, Any]]) -> bool:
        """Determine if vector search is needed based on graph results."""
        # If no graph results, definitely need vector search
        if not graph_results or len(graph_results) == 0:
            logger.info("No graph results found, vector search is needed")
            return True

        # Get the configured minimum threshold
        if "min_graph_results" not in SEARCH_CONFIG:
            raise ValueError("SEARCH_CONFIG must contain 'min_graph_results'")

        min_graph_results = SEARCH_CONFIG["min_graph_results"]

        # If we have enough graph results, we might not need vector search
        if len(graph_results) >= min_graph_results:
            logger.info(
                f"Found {len(graph_results)} graph results, which is above the minimum threshold of {min_graph_results}. "
                f"Skipping vector search for efficiency."
            )
            return False

        logger.info(
            f"Found only {len(graph_results)} graph results, which is below the minimum threshold of {min_graph_results}. "
            f"Vector search will be performed to supplement the results."
        )
        return True

    def _execute_vector_search(
        self, query: str, top_k: int, exclude_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """Execute vector search and rerank results."""
        start_time = time.time()
        logger.info(f"Executing vector search for query: '{query}' with top_k={top_k}")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        try:
            # Expand the query for better vector search
            expansion_start = time.time()
            expanded_query = self.query_generator._llm_expand_query(query)
            expansion_time = time.time() - expansion_start
            logger.info(
                f"Query expansion completed in {expansion_time:.2f}s: '{expanded_query}'"
            )

            # Perform vector search
            search_start = time.time()
            vector_results = self.vector_client.search(
                expanded_query, top_k=top_k, exclude_ids=exclude_ids or []
            )
            search_time = time.time() - search_start
            logger.info(
                f"Vector search returned {len(vector_results)} results in {search_time:.2f}s"
            )

            # Log top vector results for debugging
            self._log_top_vector_results(vector_results[:5])

            # Rerank results
            rerank_start = time.time()
            reranked_results = self.vector_client.rerank(
                query=query,
                documents=vector_results,
                top_k=top_k,
                model=self.reranker_config["model"],
            )
            rerank_time = time.time() - rerank_start
            logger.info(f"Reranking completed in {rerank_time:.2f}s")

            # Log top reranked results
            self._log_top_vector_results(reranked_results[:5], prefix="Top reranked")

            total_time = time.time() - start_time
            logger.info(f"Total vector search process completed in {total_time:.2f}s")

            return reranked_results
        except Exception as e:
            logger.error(f"Error executing vector search: {str(e)}")
            return []

    def _format_graph_results(self, graph_results: List[Dict[str, Any]]) -> str:
        """Format graph results for the QA prompt."""
        if not graph_results:
            logger.info("No graph results to format")
            return "No results found from graph search."

        logger.info(f"Formatting {len(graph_results)} graph results for prompt")
        formatted = self.document_processor.format_graph_results(graph_results)
        logger.debug(f"Formatted graph results length: {len(formatted)} characters")
        return formatted

    def _format_vector_results(self, vector_results: List[Document]) -> str:
        """Format vector results for the QA prompt."""
        if not vector_results:
            logger.info("No vector results to format")
            return "No results found from vector search."

        logger.info(f"Formatting {len(vector_results)} vector results for prompt")
        formatted = self.document_processor.format_vector_results(vector_results)
        logger.debug(f"Formatted vector results length: {len(formatted)} characters")
        return formatted

    def _generate_answer(
        self, query: str, graph_results: str, vector_results: str
    ) -> str:
        """Generate the final answer using the QA chain."""
        logger.info("Generating final answer")
        start_time = time.time()

        try:
            answer = self.qa_chain.invoke(
                {
                    "question": query,
                    "graph_results": graph_results,
                    "vector_results": vector_results,
                }
            )

            generation_time = time.time() - start_time
            logger.info(f"Answer generated in {generation_time:.2f}s")
            logger.debug(f"Answer length: {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return (
                "I encountered an error while generating an answer. Please try again."
            )

    def __call__(self, inputs: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Call the hybrid search."""
        if isinstance(inputs, str):
            inputs = {"query": inputs}

        return self.search(inputs.get("query", ""))
