import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from events_graph_rag.config import LLM_CONFIG, logger
from events_graph_rag.hybrid_search import HybridSearch
from events_graph_rag.llm_factory import LLMFactory
from events_graph_rag.neo4j_client import Neo4jClient


class BaseEvaluator:
    """Base class for evaluation of search queries."""

    def __init__(
        self,
        test_cases: Optional[Dict[str, List[int]]] = None,
        test_cases_file: Optional[str] = None,
        model_name: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            test_cases: Dictionary mapping queries to expected event IDs
            test_cases_file: Path to a JSON file containing test cases
            model_name: Name of the model being evaluated (for tracking purposes)
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Set model name from parameter or config
        self.model_name = model_name
        logger.info(f"Using model: {self.model_name}")

        # Load test cases
        self.test_cases = {}
        if test_cases:
            self.test_cases = test_cases
        elif test_cases_file:
            self.load_test_cases(test_cases_file)

        logger.info(f"Initialized evaluator with {len(self.test_cases)} test cases")

    def load_test_cases(self, file_path: str) -> None:
        """
        Load test cases from a JSON file.

        Args:
            file_path: Path to the JSON file containing test cases
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Test cases file not found: {file_path}")
                return

            with open(path, "r", encoding="utf-8") as f:
                self.test_cases = json.load(f)

            logger.info(f"Loaded {len(self.test_cases)} test cases from {file_path}")
        except Exception as e:
            logger.error(f"Error loading test cases: {str(e)}")
            self.test_cases = {}

    def save_test_cases(self, file_path: str) -> None:
        """
        Save test cases to a JSON file.

        Args:
            file_path: Path to save the JSON file
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.test_cases, f, indent=2)
            logger.info(f"Saved {len(self.test_cases)} test cases to {file_path}")
        except Exception as e:
            logger.error(f"Error saving test cases: {str(e)}")

    def add_test_case(self, query: str, expected_ids: List[int]) -> None:
        """
        Add a test case to the evaluator.

        Args:
            query: The query string
            expected_ids: List of expected event IDs
        """
        self.test_cases[query] = expected_ids
        logger.info(f"Added test case: '{query}' -> {expected_ids}")

    def remove_test_case(self, query: str) -> None:
        """
        Remove a test case from the evaluator.

        Args:
            query: The query string to remove
        """
        if query in self.test_cases:
            del self.test_cases[query]
            logger.info(f"Removed test case: '{query}'")
        else:
            logger.warning(f"Test case not found: '{query}'")

    def calculate_metrics(
        self, expected_ids: List[int], actual_ids: List[Union[int, str]]
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a single test case.

        Args:
            expected_ids: List of expected event IDs
            actual_ids: List of actual event IDs returned by the search

        Returns:
            Dictionary containing precision, recall, and F1 score
        """
        # Convert all IDs to integers for comparison
        try:
            actual_ids_int = [int(id) for id in actual_ids if str(id).strip()]
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting IDs to integers: {str(e)}")
            actual_ids_int = []

        # Convert to sets for easier comparison
        expected_set = set(expected_ids)
        actual_set = set(actual_ids_int)

        # Calculate true positives, false positives, and false negatives
        true_positives = len(expected_set.intersection(actual_set))
        false_positives = len(actual_set - expected_set)
        false_negatives = len(expected_set - actual_set)

        # Calculate precision, recall, and F1 score
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def format_results(
        self, results: List[Dict[str, Any]], include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Format evaluation results into a summary.

        Args:
            results: List of evaluation results for each test case
            include_details: Whether to include detailed results for each test case

        Returns:
            Dictionary containing evaluation summary
        """
        # Calculate average metrics
        avg_precision = (
            sum(r["metrics"]["precision"] for r in results) / len(results)
            if results
            else 0
        )
        avg_recall = (
            sum(r["metrics"]["recall"] for r in results) / len(results)
            if results
            else 0
        )
        avg_f1 = (
            sum(r["metrics"]["f1"] for r in results) / len(results) if results else 0
        )

        # Count perfect matches
        perfect_matches = sum(1 for r in results if r["metrics"]["f1"] == 1.0)

        # Calculate average execution time
        avg_time = (
            sum(r["execution_time"] for r in results) / len(results) if results else 0
        )

        summary = {
            "model_name": self.model_name,
            "total_queries": len(results),
            "perfect_matches": perfect_matches,
            "perfect_match_rate": perfect_matches / len(results) if results else 0,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_execution_time": avg_time,
        }

        if include_details:
            summary["details"] = results

        return summary


class GraphQueryEvaluator(BaseEvaluator):
    """Evaluator for testing graph queries using Cypher generation and execution."""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        hybrid_search: Optional[HybridSearch] = None,
        test_cases: Optional[Dict[str, List[int]]] = None,
        test_cases_file: Optional[str] = None,
        model_name: Optional[str] = None,
        few_shot_examples_file: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the graph query evaluator.

        Args:
            neo4j_client: Neo4j client instance
            hybrid_search: HybridSearch instance for Cypher generation
            test_cases: Dictionary mapping queries to expected event IDs
            test_cases_file: Path to a JSON file containing test cases
            model_name: Name of the model being evaluated
            few_shot_examples_file: Path to save successful query-cypher pairs for few-shot prompting
            verbose: Whether to enable verbose logging
        """
        # Initialize Neo4j client
        self.neo4j_client = neo4j_client or Neo4jClient()
        logger.info("Initialized Neo4j client for graph query evaluation")

        # Initialize HybridSearch with the specified model if provided
        if model_name and not hybrid_search:
            # Get the provider for the model name
            provider = LLMFactory.get_provider_for_model(model_name)

            if provider:
                # Create LLMs with the specified model
                llm = LLMFactory.create_llm(provider=provider, model_name=model_name)
                cypher_llm = LLMFactory.create_cypher_llm(
                    provider=provider, model_name=model_name
                )

                # Create HybridSearch with the LLMs
                self.hybrid_search = HybridSearch(
                    llm=llm, cypher_llm=cypher_llm, verbose=verbose
                )
                logger.info(
                    f"Initialized HybridSearch with model: {model_name} (provider: {provider})"
                )
            else:
                logger.warning(
                    f"Model {model_name} not found in available models. Using default configuration."
                )
                self.hybrid_search = hybrid_search or HybridSearch(verbose=verbose)
        else:
            self.hybrid_search = hybrid_search or HybridSearch(verbose=verbose)
            logger.info("Initialized HybridSearch with default configuration")

        # Get the actual model name from HybridSearch if not provided
        if model_name is None:
            if hasattr(self.hybrid_search.cypher_llm, "model_name"):
                model_name = self.hybrid_search.cypher_llm.model_name
            elif hasattr(self.hybrid_search.cypher_llm, "model"):
                model_name = self.hybrid_search.cypher_llm.model
            else:
                model_name = "unknown"

        super().__init__(test_cases, test_cases_file, model_name, verbose)

        # Set up few-shot examples collection
        self.few_shot_examples_file = few_shot_examples_file
        self.few_shot_examples = {}

        # Load existing few-shot examples if file exists
        if few_shot_examples_file and os.path.exists(few_shot_examples_file):
            try:
                with open(few_shot_examples_file, "r", encoding="utf-8") as f:
                    self.few_shot_examples = json.load(f)
                logger.info(
                    f"Loaded {len(self.few_shot_examples)} few-shot examples from {few_shot_examples_file}"
                )
            except Exception as e:
                logger.error(f"Error loading few-shot examples: {str(e)}")
                self.few_shot_examples = {}

    def evaluate_query(self, query: str, expected_ids: List[int]) -> Dict[str, Any]:
        """
        Evaluate a single graph query.

        Args:
            query: The query string
            expected_ids: List of expected event IDs

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating graph query: '{query}'")
        start_time = time.time()

        # Generate Cypher query using the HybridSearch instance
        cypher_query = self.hybrid_search._generate_cypher_query(query)
        cypher_generation_time = round(time.time() - start_time, 2)

        if not cypher_query:
            logger.warning("Failed to generate Cypher query")
            return {
                "query": query,
                "expected_ids": expected_ids,
                "actual_ids": [],
                "cypher_query": None,
                "metrics": self.calculate_metrics(expected_ids, []),
                "execution_time": round(time.time() - start_time, 2),
                "cypher_generation_time": cypher_generation_time,
                "cypher_execution_time": 0,
                "success": False,
                "error": "Failed to generate Cypher query",
            }

        # Execute Cypher query
        cypher_execution_start = time.time()
        try:
            results = self.neo4j_client.query(cypher_query)
            cypher_execution_time = round(time.time() - cypher_execution_start, 2)

            # Extract event IDs from results
            actual_ids = self.neo4j_client.extract_event_ids(results)

            # Log found IDs
            logger.info(f"Found IDs: {actual_ids}")
            logger.info(f"Expected IDs: {expected_ids}")

            # Calculate metrics
            metrics = self.calculate_metrics(expected_ids, actual_ids)

            # If this is a perfect match, store it as a few-shot example
            if metrics["f1"] == 1.0 and self.few_shot_examples_file:
                self._add_few_shot_example(query, cypher_query)

            return {
                "query": query,
                "expected_ids": expected_ids,
                "actual_ids": actual_ids,
                "cypher_query": cypher_query,
                "metrics": metrics,
                "execution_time": round(time.time() - start_time, 2),
                "cypher_generation_time": cypher_generation_time,
                "cypher_execution_time": cypher_execution_time,
                "success": True,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return {
                "query": query,
                "expected_ids": expected_ids,
                "actual_ids": [],
                "cypher_query": cypher_query,
                "metrics": self.calculate_metrics(expected_ids, []),
                "execution_time": round(time.time() - start_time, 2),
                "cypher_generation_time": cypher_generation_time,
                "cypher_execution_time": round(time.time() - cypher_execution_start, 2),
                "success": False,
                "error": str(e),
            }

    def _add_few_shot_example(self, query: str, cypher_query: str) -> None:
        """
        Add a successful query-cypher pair to the few-shot examples.

        Args:
            query: The natural language query
            cypher_query: The generated Cypher query
        """
        # Only add if it's not already in the examples
        if query not in self.few_shot_examples:
            self.few_shot_examples[query] = cypher_query
            logger.info(f"Added new few-shot example: '{query}'")

            # Save to file if specified
            if self.few_shot_examples_file:
                try:
                    with open(self.few_shot_examples_file, "w", encoding="utf-8") as f:
                        json.dump(self.few_shot_examples, f, indent=2)
                    logger.info(
                        f"Saved {len(self.few_shot_examples)} few-shot examples to {self.few_shot_examples_file}"
                    )
                except Exception as e:
                    logger.error(f"Error saving few-shot examples: {str(e)}")

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all test cases.

        Returns:
            Dictionary containing evaluation summary
        """
        if not self.test_cases:
            logger.warning("No test cases to evaluate")
            return {"total_queries": 0, "model_name": self.model_name}

        logger.info(f"Evaluating {len(self.test_cases)} graph queries")
        start_time = time.time()

        results = []
        for query, expected_ids in self.test_cases.items():
            result = self.evaluate_query(query, expected_ids)
            results.append(result)

        # Format results
        summary = self.format_results(results, include_details=True)
        summary["total_evaluation_time"] = time.time() - start_time

        # Add few-shot examples stats
        if self.few_shot_examples_file:
            summary["few_shot_examples_count"] = len(self.few_shot_examples)
            summary["few_shot_examples_file"] = self.few_shot_examples_file

        logger.info(f"Evaluation completed in {summary['total_evaluation_time']:.2f}s")
        logger.info(f"Perfect match rate: {summary['perfect_match_rate']:.2f}")
        logger.info(f"Average F1 score: {summary['avg_f1']:.2f}")

        if self.few_shot_examples_file:
            logger.info(f"Collected {len(self.few_shot_examples)} few-shot examples")

        return summary


class EndToEndEvaluator(BaseEvaluator):
    """Evaluator for testing end-to-end hybrid search queries."""

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        test_cases: Optional[Dict[str, List[int]]] = None,
        test_cases_file: Optional[str] = None,
        model_name: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the end-to-end evaluator.

        Args:
            hybrid_search: HybridSearch instance
            test_cases: Dictionary mapping queries to expected event IDs
            test_cases_file: Path to a JSON file containing test cases
            model_name: Name of the model being evaluated
            verbose: Whether to enable verbose logging
        """
        # Initialize HybridSearch with the specified model if provided
        if model_name and not hybrid_search:
            # Get the provider for the model name
            provider = LLMFactory.get_provider_for_model(model_name)

            if provider:
                # Create LLMs with the specified model
                llm = LLMFactory.create_llm(provider=provider, model_name=model_name)
                cypher_llm = LLMFactory.create_cypher_llm(
                    provider=provider, model_name=model_name
                )

                # Create HybridSearch with the LLMs
                self.hybrid_search = HybridSearch(
                    llm=llm, cypher_llm=cypher_llm, verbose=verbose
                )
                logger.info(
                    f"Initialized HybridSearch with model: {model_name} (provider: {provider})"
                )
            else:
                logger.warning(
                    f"Model {model_name} not found in available models. Using default configuration."
                )
                self.hybrid_search = hybrid_search or HybridSearch(verbose=verbose)
        else:
            self.hybrid_search = hybrid_search or HybridSearch(verbose=verbose)
            logger.info("Initialized HybridSearch with default configuration")

        # Get the actual model name from HybridSearch if not provided
        if model_name is None:
            if hasattr(self.hybrid_search.llm, "model_name"):
                model_name = self.hybrid_search.llm.model_name
            elif hasattr(self.hybrid_search.llm, "model"):
                model_name = self.hybrid_search.llm.model
            else:
                model_name = "unknown"

        super().__init__(test_cases, test_cases_file, model_name, verbose)

    def evaluate_query(self, query: str, expected_ids: List[int]) -> Dict[str, Any]:
        """
        Evaluate a single end-to-end query.

        Args:
            query: The query string
            expected_ids: List of expected event IDs

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating end-to-end query: '{query}'")
        start_time = time.time()

        try:
            # Execute hybrid search
            search_result = self.hybrid_search.search(query)

            # Extract event IDs from both graph and vector results
            graph_ids = []
            for result in search_result.get("graph_results", []):
                event_info = self.hybrid_search.neo4j_client.extract_event_info(result)
                event_id = event_info["id"]
                if event_id != "unknown_id" and event_id not in graph_ids:
                    graph_ids.append(event_id)

            vector_ids = []
            for doc in search_result.get("vector_results", []):
                if hasattr(doc, "metadata") and "event_id" in doc.metadata:
                    event_id = doc.metadata["event_id"]
                    if event_id not in vector_ids:
                        vector_ids.append(event_id)

            # Combine IDs from both sources
            all_ids = list(set(graph_ids + vector_ids))

            # Log found IDs
            logger.info(f"Found IDs: {all_ids}")
            logger.info(f"Expected IDs: {expected_ids}")

            # Calculate metrics
            metrics = self.calculate_metrics(expected_ids, all_ids)

            return {
                "query": query,
                "expected_ids": expected_ids,
                "actual_ids": all_ids,
                "graph_ids": graph_ids,
                "vector_ids": vector_ids,
                "metrics": metrics,
                "execution_time": round(time.time() - start_time, 2),
                "graph_search_time": round(
                    search_result.get("metadata", {}).get("graph_search_time", 0), 2
                ),
                "vector_search_time": round(
                    search_result.get("metadata", {}).get("vector_search_time", 0), 2
                ),
                "total_search_time": round(
                    search_result.get("metadata", {}).get("total_time", 0), 2
                ),
                "success": True,
                "error": None,
                "answer": search_result.get("answer", ""),
            }
        except Exception as e:
            logger.error(f"Error executing hybrid search: {str(e)}")
            return {
                "query": query,
                "expected_ids": expected_ids,
                "actual_ids": [],
                "graph_ids": [],
                "vector_ids": [],
                "metrics": self.calculate_metrics(expected_ids, []),
                "execution_time": round(time.time() - start_time, 2),
                "graph_search_time": 0,
                "vector_search_time": 0,
                "total_search_time": 0,
                "success": False,
                "error": str(e),
                "answer": "",
            }

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all test cases.

        Returns:
            Dictionary containing evaluation summary
        """
        if not self.test_cases:
            logger.warning("No test cases to evaluate")
            return {"total_queries": 0, "model_name": self.model_name}

        logger.info(f"Evaluating {len(self.test_cases)} end-to-end queries")
        start_time = time.time()

        results = []
        for query, expected_ids in self.test_cases.items():
            result = self.evaluate_query(query, expected_ids)
            results.append(result)

        # Format results
        summary = self.format_results(results, include_details=True)
        summary["total_evaluation_time"] = time.time() - start_time

        # Calculate additional metrics
        if results:
            summary["avg_graph_search_time"] = sum(
                r.get("graph_search_time", 0) for r in results
            ) / len(results)
            summary["avg_vector_search_time"] = sum(
                r.get("vector_search_time", 0) for r in results
            ) / len(results)
            summary["avg_total_search_time"] = sum(
                r.get("total_search_time", 0) for r in results
            ) / len(results)

        logger.info(f"Evaluation completed in {summary['total_evaluation_time']:.2f}s")
        logger.info(f"Perfect match rate: {summary['perfect_match_rate']:.2f}")
        logger.info(f"Average F1 score: {summary['avg_f1']:.2f}")

        return summary
