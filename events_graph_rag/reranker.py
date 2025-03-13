"""Simple Jina AI reranker implementation without abstractions."""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import dotenv
from langchain_core.documents import Document

# Load environment variables
dotenv.load_dotenv(".env", override=True)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Simple configuration for reranker."""

    model: str = "jina-reranker-v2-base-multilingual"
    top_k: int = 100


@dataclass
class RerankerResult:
    """Result from reranking."""

    score: float
    document: Dict[str, Any]


class JinaReranker:
    """Simple Jina reranker implementation."""

    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize Jina reranker.

        Args:
            config: Optional reranker configuration
        """
        self.config = config or RerankerConfig()
        logger.info(f"Initializing JinaReranker with config: {self.config}")

        # Check if JINA_API_KEY is set
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            logger.error("JINA_API_KEY environment variable is not set!")
            raise ValueError("JINA_API_KEY environment variable is required")

        self.url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[RerankerResult]:
        """Rerank documents based on query.

        Args:
            query: Query text
            documents: List of documents to rerank (each with 'text' and 'metadata' keys)

        Returns:
            List of reranked documents with scores
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []

        logger.info(f"Reranking {len(documents)} documents with query: '{query}'")

        try:
            # Extract text from documents
            texts = [doc.get("text", "") for doc in documents]

            # Prepare request data
            data = {
                "model": self.config.model,
                "query": query,
                "documents": texts,
                "top_n": self.config.top_k,
            }
            logger.info(f"Reranking {len(texts)} documents with query: {query}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url, headers=self.headers, json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Jina API error: {response.status} - {error_text}"
                        )

                    result = await response.json()

                    # Process results
                    reranked = []
                    for item in result["results"]:
                        doc_idx = item["index"]
                        score = item["relevance_score"]
                        reranked.append(
                            RerankerResult(score=score, document=documents[doc_idx])
                        )

                    # Sort by score in descending order
                    reranked.sort(key=lambda x: x.score, reverse=True)
                    return reranked[: self.config.top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Return original documents in case of error
            return [RerankerResult(score=0.0, document=doc) for doc in documents]


def rerank_documents(
    query: str, documents: List[Document], config: Optional[RerankerConfig] = None
) -> List[Document]:
    """Utility function to rerank langchain documents.

    Args:
        query: Query text
        documents: List of langchain documents to rerank
        config: Optional reranker configuration

    Returns:
        Reranked langchain documents
    """
    logger.info(f"Starting document reranking with {len(documents)} documents")

    # Convert langchain documents to dictionaries
    docs_dict = []
    for doc in documents:
        docs_dict.append({"text": doc.page_content, "metadata": doc.metadata})

    # Create reranker
    try:
        reranker = JinaReranker(config or RerankerConfig())
        logger.info("JinaReranker initialized successfully")

        # Run the reranker synchronously
        loop = asyncio.new_event_loop()
        try:
            logger.info("Starting reranking process...")
            reranked_results = loop.run_until_complete(
                reranker.rerank(query, docs_dict)
            )
            logger.info(f"Reranking complete, received {len(reranked_results)} results")
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error during reranker initialization or execution: {str(e)}")
        # Return original documents in case of error
        return documents

    # Convert back to langchain documents
    reranked_docs = []
    for result in reranked_results:
        # Create a new Document with the same content but updated metadata
        metadata = result.document["metadata"].copy()
        # Add the reranker score to metadata
        metadata["reranker_score"] = result.score

        reranked_docs.append(
            Document(page_content=result.document["text"], metadata=metadata)
        )

    return reranked_docs
