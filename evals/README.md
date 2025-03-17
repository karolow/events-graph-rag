# Events Graph RAG Evaluation System

This evaluation system allows you to test the accuracy of the hybrid search system for cultural events. It provides two types of evaluations:

1. **Graph Query Evaluation**: Tests the accuracy of Cypher query generation and execution
2. **End-to-End Evaluation**: Tests the complete hybrid search pipeline including both graph and vector search

## Test Cases

Test cases are defined in a JSON file with the following format:

```json
{
  "Query 1": [1, 2, 3],
  "Query 2": [4, 5, 6],
  ...
}
```

Where:
- Each key is a natural language query
- Each value is a list of expected event IDs that should be returned for that query

A sample test cases file is provided at `sample_test_cases.json`.

## Running Evaluations

You can run evaluations using the `eval_cli.py` script:

```bash
# Run both graph and end-to-end evaluations
eval-events --test-cases evals/events_graph_rag/sample_test_cases.json --output results.json

# Run only graph query evaluation
eval-events --type graph --test-cases evals/events_graph_rag/sample_test_cases.json --output graph_results.json

# Run only end-to-end evaluation
eval-events --type end-to-end --test-cases evals/events_graph_rag/sample_test_cases.json --output e2e_results.json

# Enable verbose logging
eval-events --test-cases evals/events_graph_rag/sample_test_cases.json --verbose

# Specify model name for tracking purposes
eval-events --test-cases evals/events_graph_rag/sample_test_cases.json --model-name "gemini-2.0-flash"

# Collect few-shot examples from successful graph queries
eval-events --type graph --test-cases evals/events_graph_rag/sample_test_cases.json --few-shot-examples few_shot_examples.json
```

## Adding Test Cases

You can add test cases to an existing test cases file using the `--add-test-case` option:

```bash
eval-events --add-test-case --test-cases evals/events_graph_rag/sample_test_cases.json --query "Find events featuring jazz music" --expected-ids "7,15,32"
```

## Few-Shot Examples Collection

The evaluation system can collect successful query-cypher pairs for future few-shot prompting. These examples are stored in a JSON file with the following format:

```json
{
  "Find events in which Alice Jones participated alone": "MATCH (p:Person {name: 'Alice Jones'})-[:PARTICIPATED_IN]->(e:Event) WHERE NOT EXISTS((e)<-[:PARTICIPATED_IN]-(:Person)) OR COUNT((e)<-[:PARTICIPATED_IN]-(:Person)) = 1 RETURN e",
  "Find outdoor music events with more than 100 participants": "MATCH (e:Event)-[:BELONGS_TO]->(c:Category {name: 'Music'}), (e)-[:TAKES_PLACE_IN]->(l:Location {type: 'Outdoor'}) WHERE e.number_of_participants > 100 RETURN e"
}
```

To collect few-shot examples, use the `--few-shot-examples` option:

```bash
eval-events --type graph --test-cases evals/events_graph_rag/sample_test_cases.json --few-shot-examples few_shot_examples.json
```

Only queries that achieve a perfect F1 score (1.0) are added to the few-shot examples file. If a query is already in the file, it won't be added again.

## Evaluation Metrics

The evaluation system calculates the following metrics:

- **Precision**: The proportion of retrieved events that are relevant
- **Recall**: The proportion of relevant events that are retrieved
- **F1 Score**: The harmonic mean of precision and recall
- **Perfect Match Rate**: The proportion of queries that have a perfect F1 score (1.0)

## Evaluation Results

The evaluation results are saved in a JSON file with the following structure:

```json
{
  "graph": {
    "model_name": "gemini-2.0-flash",
    "total_queries": 10,
    "perfect_matches": 7,
    "perfect_match_rate": 0.7,
    "avg_precision": 0.85,
    "avg_recall": 0.9,
    "avg_f1": 0.87,
    "avg_execution_time": 1.2,
    "total_evaluation_time": 12.5,
    "few_shot_examples_count": 7,
    "few_shot_examples_file": "few_shot_examples.json",
    "details": [
      {
        "query": "Find events in which Alice Jones participated alone",
        "expected_ids": [121],
        "actual_ids": [121],
        "cypher_query": "MATCH (p:Person {name: 'Alice Jones'})-[:PARTICIPATED_IN]->(e:Event) WHERE NOT EXISTS((e)<-[:PARTICIPATED_IN]-(:Person)) OR COUNT((e)<-[:PARTICIPATED_IN]-(:Person)) = 1 RETURN e",
        "metrics": {
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0,
          "true_positives": 1,
          "false_positives": 0,
          "false_negatives": 0
        },
        "execution_time": 0.8,
        "cypher_generation_time": 0.5,
        "cypher_execution_time": 0.3,
        "success": true,
        "error": null
      },
      ...
    ]
  },
  "end_to_end": {
    "model_name": "gemini-2.0-flash",
    "total_queries": 10,
    "perfect_matches": 8,
    "perfect_match_rate": 0.8,
    "avg_precision": 0.9,
    "avg_recall": 0.95,
    "avg_f1": 0.92,
    "avg_execution_time": 2.5,
    "avg_graph_search_time": 0.8,
    "avg_vector_search_time": 1.2,
    "avg_total_search_time": 2.0,
    "total_evaluation_time": 25.0,
    "details": [
      {
        "query": "Find events in which Alice Jones participated alone",
        "expected_ids": [121],
        "actual_ids": [121],
        "graph_ids": [121],
        "vector_ids": [],
        "metrics": {
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0,
          "true_positives": 1,
          "false_positives": 0,
          "false_negatives": 0
        },
        "execution_time": 2.0,
        "graph_search_time": 0.8,
        "vector_search_time": 0.0,
        "total_search_time": 0.8,
        "success": true,
        "error": null,
        "answer": "Alice Jones participated alone in one event: [Event details...]"
      },
      ...
    ]
  }
}
```

## Programmatic Usage

You can also use the evaluation classes programmatically in your own code:

```python
from evals.events_graph_rag.evaluator import GraphQueryEvaluator, EndToEndEvaluator

# Define test cases
test_cases = {
    "Find events in which Alice Jones participated alone": [121],
    "Find outdoor music events with more than 100 participants": [1, 34, 77]
}

# Create evaluators
graph_evaluator = GraphQueryEvaluator(
    test_cases=test_cases,
    model_name="gemini-2.0-flash",
    few_shot_examples_file="few_shot_examples.json",
    verbose=True
)

e2e_evaluator = EndToEndEvaluator(
    test_cases=test_cases,
    model_name="gemini-2.0-flash",
    verbose=True
)

# Run evaluations
graph_results = graph_evaluator.evaluate_all()
e2e_results = e2e_evaluator.evaluate_all()

# Access results
print(f"Graph perfect match rate: {graph_results['perfect_match_rate']}")
print(f"End-to-end perfect match rate: {e2e_results['perfect_match_rate']}")
print(f"Collected {graph_results['few_shot_examples_count']} few-shot examples") 