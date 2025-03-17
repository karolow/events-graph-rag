import logging
import os
from typing import Any, Dict

import dotenv

# Load environment variables
dotenv.load_dotenv(".env", override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Neo4j connection parameters
NEO4J_CONFIG = {
    "url": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    "username": os.environ.get("NEO4J_USERNAME", ""),
    "password": os.environ.get("NEO4J_PASSWORD", ""),
}

# Data source configuration
DATA_SOURCE_CONFIG = {
    "default_csv_url": "https://raw.githubusercontent.com/karolow/datasets/refs/heads/main/events_kmo_sample.csv"
}

# Vector index configuration
VECTOR_INDEX_CONFIG = {
    "index_name": "events_vector_index",  # Primary index name
    "fallback_index_name": "events",  # Fallback index name
    "node_label": "Event",
    "text_node_property": "combined_text",
    "embedding_node_property": "embedding",
    "retrieval_query": """
    WITH node, score
    MATCH (e:Event) WHERE e = node
    RETURN e.combined_text AS text,
           elementId(e) AS id,
           {
               event_id: e.id, 
               name: e.name, 
               number_of_participants: e.number_of_participants,
               start_date: e.start_date,
               start_time: e.start_time,
               location: CASE 
                   WHEN EXISTS((e)-[:TAKES_PLACE_IN]->(:Location)) 
                   THEN [(e)-[:TAKES_PLACE_IN]->(l:Location) | l.name][0]
                   ELSE null
               END,
               category: CASE 
                   WHEN EXISTS((e)-[:BELONGS_TO]->(:Category)) 
                   THEN [(e)-[:BELONGS_TO]->(c:Category) | c.name][0]
                   ELSE null
               END
           } AS metadata,
           score
    """,
}

# LLM model configuration
LLM_CONFIG = {
    "provider": "gemini",  # Options: "groq", "gemini", "openai", "anthropic"
    "groq_model": "llama-3.3-70b-versatile",
    "gemini_model": "gemini-2.0-flash",
    "openai_model": "gpt-4o",
    "anthropic_model": "claude-3.7-sonnet",
    "temperature": 0,
    # Separate configuration for Cypher generation
    "cypher": {
        "provider": "gemini",  # Options: "groq", "gemini", "openai", "anthropic"
        "groq_model": "deepseek-r1-distill-qwen-32b",
        "gemini_model": "gemini-2.0-flash",
        "openai_model": "gpt-4o",
        "anthropic_model": "claude-3-7-sonnet-latest",
        "temperature": 0,
    },
    # Available models by provider for CLI selection
    "available_models": {
        "groq": [
            "qwen-2.5-coder-32b",
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-qwen-32b",
        ],
        "gemini": ["gemini-2.0-flash", "gemini-2.0-pro"],
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3.7-sonnet", "claude-3.7-sonnet-thinking"],
    },
}

# Reranker configuration
RERANKER_CONFIG = {
    "model": "jina-reranker-v2-base-multilingual",
    "top_k": 20,
}

# Search configuration
SEARCH_CONFIG = {
    "vector_top_k": 20,
    "max_workers": 2,
    "max_graph_results_for_vector": 50,
    "use_reranker": True,
    "min_graph_results": 3,
}


def get_config() -> Dict[str, Any]:
    """
    Returns the complete configuration dictionary.
    """
    return {
        "neo4j": NEO4J_CONFIG,
        "data_source": DATA_SOURCE_CONFIG,
        "vector_index": VECTOR_INDEX_CONFIG,
        "llm": LLM_CONFIG,
        "reranker": RERANKER_CONFIG,
        "search": SEARCH_CONFIG,
    }
