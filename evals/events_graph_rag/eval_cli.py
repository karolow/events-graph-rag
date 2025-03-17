"""
Command-line interface for running evaluations on the hybrid search system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from evals.events_graph_rag.evaluator import EndToEndEvaluator, GraphQueryEvaluator
from events_graph_rag.config import LLM_CONFIG, logger
from events_graph_rag.llm_factory import LLMFactory


def get_available_models() -> List[str]:
    """Get a list of all available models from all providers."""
    available_models = []
    for provider, models in LLM_CONFIG.get("available_models", {}).items():
        available_models.extend(models)
    return available_models


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate graph and hybrid search queries."
    )

    # Evaluation type
    parser.add_argument(
        "--type",
        choices=["graph", "end-to-end", "both"],
        default="both",
        help="Type of evaluation to run (default: both)",
    )

    # Test cases
    parser.add_argument(
        "--test-cases",
        type=str,
        help="Path to JSON file containing test cases",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results",
    )

    # Get available models
    available_models = get_available_models()

    # Model name
    parser.add_argument(
        "--model-name",
        type=str,
        default="default",
        choices=["default"] + available_models,
        help=f"Name of the model to use for evaluation. Available models: {', '.join(available_models)}",
    )

    # Few-shot examples file
    parser.add_argument(
        "--few-shot-examples",
        type=str,
        help="Path to save successful query-cypher pairs for few-shot prompting",
    )

    # Verbose mode
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Add test case
    parser.add_argument(
        "--add-test-case",
        action="store_true",
        help="Add a test case to the test cases file",
    )

    # Query for adding test case
    parser.add_argument(
        "--query",
        type=str,
        help="Query string for the test case",
    )

    # Expected IDs for adding test case
    parser.add_argument(
        "--expected-ids",
        type=str,
        help="Comma-separated list of expected event IDs for the test case",
    )

    # List available models
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models by provider",
    )

    return parser.parse_args()


def add_test_case(args):
    """Add a test case to the test cases file."""
    if not args.query:
        logger.error("Query is required for adding a test case")
        sys.exit(1)

    if not args.expected_ids:
        logger.error("Expected IDs are required for adding a test case")
        sys.exit(1)

    if not args.test_cases:
        logger.error("Test cases file is required for adding a test case")
        sys.exit(1)

    # Parse expected IDs
    try:
        expected_ids = [int(id.strip()) for id in args.expected_ids.split(",")]
    except ValueError:
        logger.error("Expected IDs must be comma-separated integers")
        sys.exit(1)

    # Load existing test cases or create new file
    test_cases = {}
    test_cases_path = Path(args.test_cases)

    if test_cases_path.exists():
        try:
            with open(test_cases_path, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
        except Exception as e:
            logger.error(f"Error loading test cases file: {str(e)}")
            sys.exit(1)

    # Add the new test case
    test_cases[args.query] = expected_ids

    # Save the updated test cases
    try:
        with open(test_cases_path, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2)
        logger.info(f"Added test case: '{args.query}' -> {expected_ids}")
    except Exception as e:
        logger.error(f"Error saving test cases file: {str(e)}")
        sys.exit(1)


def run_graph_evaluation(args) -> Dict:
    """Run graph query evaluation."""
    # Use the default model if specified
    model_name = None if args.model_name == "default" else args.model_name

    logger.info(f"Running graph query evaluation with model: {model_name or 'default'}")

    evaluator = GraphQueryEvaluator(
        test_cases_file=args.test_cases,
        model_name=model_name,
        few_shot_examples_file=args.few_shot_examples,
        verbose=args.verbose,
    )

    results = evaluator.evaluate_all()

    return results


def run_end_to_end_evaluation(args) -> Dict:
    """Run end-to-end evaluation."""
    # Use the default model if specified
    model_name = None if args.model_name == "default" else args.model_name

    logger.info(f"Running end-to-end evaluation with model: {model_name or 'default'}")

    evaluator = EndToEndEvaluator(
        test_cases_file=args.test_cases,
        model_name=model_name,
        verbose=args.verbose,
    )

    results = evaluator.evaluate_all()

    return results


def save_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")


def main():
    """Main entry point for the evaluation CLI."""
    args = parse_args()

    # List available models if requested
    if args.list_models:
        print("Available models by provider:")
        for provider, models in LLM_CONFIG.get("available_models", {}).items():
            print(f"\n{provider.upper()}:")
            for model in models:
                print(f"  - {model}")
        return

    # Add a test case if requested
    if args.add_test_case:
        add_test_case(args)
        return

    # Check if test cases file exists
    if args.test_cases and not Path(args.test_cases).exists():
        logger.error(f"Test cases file not found: {args.test_cases}")
        sys.exit(1)

    # Run evaluations
    results = {}

    if args.type in ["graph", "both"]:
        graph_results = run_graph_evaluation(args)
        results["graph"] = graph_results

        # Print few-shot examples stats if available
        if "few_shot_examples_count" in graph_results:
            logger.info(
                f"Collected {graph_results['few_shot_examples_count']} few-shot examples"
            )
            logger.info(
                f"Few-shot examples saved to: {graph_results['few_shot_examples_file']}"
            )

    if args.type in ["end-to-end", "both"]:
        end_to_end_results = run_end_to_end_evaluation(args)
        results["end_to_end"] = end_to_end_results

    # Print summary
    if "graph" in results:
        logger.info("Graph Query Evaluation Summary:")
        logger.info(f"  Model: {results['graph']['model_name']}")
        logger.info(f"  Total queries: {results['graph']['total_queries']}")
        logger.info(f"  Perfect matches: {results['graph']['perfect_matches']}")
        logger.info(
            f"  Perfect match rate: {results['graph']['perfect_match_rate']:.2f}"
        )
        logger.info(f"  Average F1 score: {results['graph']['avg_f1']:.2f}")

    if "end_to_end" in results:
        logger.info("End-to-End Evaluation Summary:")
        logger.info(f"  Model: {results['end_to_end']['model_name']}")
        logger.info(f"  Total queries: {results['end_to_end']['total_queries']}")
        logger.info(f"  Perfect matches: {results['end_to_end']['perfect_matches']}")
        logger.info(
            f"  Perfect match rate: {results['end_to_end']['perfect_match_rate']:.2f}"
        )
        logger.info(f"  Average F1 score: {results['end_to_end']['avg_f1']:.2f}")

    # Save results if output path is provided
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
