from typing import List, Optional

from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from events_graph_rag.config import (
    NEO4J_CONFIG,
    SEARCH_CONFIG,
    VECTOR_INDEX_CONFIG,
    logger,
)
from events_graph_rag.reranker import rerank_documents


class VectorClient:
    """Client for vector database operations."""

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        index_name: str = None,
        fallback_index_name: str = None,
    ):
        """Initialize the vector client."""
        # Connection parameters
        self.url = url or NEO4J_CONFIG["url"]
        self.username = username or NEO4J_CONFIG["username"]
        self.password = password or NEO4J_CONFIG["password"]

        # Vector index parameters
        self.index_name = index_name or VECTOR_INDEX_CONFIG["index_name"]
        self.fallback_index_name = (
            fallback_index_name or VECTOR_INDEX_CONFIG["fallback_index_name"]
        )
        self.node_label = VECTOR_INDEX_CONFIG["node_label"]
        self.text_node_property = VECTOR_INDEX_CONFIG["text_node_property"]
        self.embedding_node_property = VECTOR_INDEX_CONFIG["embedding_node_property"]
        self.retrieval_query = VECTOR_INDEX_CONFIG["retrieval_query"]

        # Initialize embeddings and vector index
        self.embeddings = OpenAIEmbeddings()
        self.vector_index = self._initialize_vector_index()

    def _initialize_vector_index(self):
        """Initialize the vector index from the database."""
        try:
            vector_index = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self.url,
                username=self.username,
                password=self.password,
                index_name=self.index_name,
                node_label=self.node_label,
                text_node_property=self.text_node_property,
                embedding_node_property=self.embedding_node_property,
                retrieval_query=self.retrieval_query,
            )
            logger.info(
                f"Successfully loaded vector index '{self.index_name}' from Neo4j"
            )
            return vector_index
        except Exception as e:
            # Fallback to using the fallback index name if primary fails
            logger.warning(f"Failed to load '{self.index_name}': {e}")
            logger.info(f"Trying fallback index name '{self.fallback_index_name}'...")
            try:
                vector_index = Neo4jVector.from_existing_index(
                    self.embeddings,
                    url=self.url,
                    username=self.username,
                    password=self.password,
                    index_name=self.fallback_index_name,
                    node_label=self.node_label,
                    text_node_property=self.text_node_property,
                    embedding_node_property=self.embedding_node_property,
                    retrieval_query=self.retrieval_query,
                )
                logger.info(
                    f"Successfully loaded vector index '{self.fallback_index_name}' from Neo4j"
                )
                return vector_index
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback index: {fallback_error}")
                raise RuntimeError(
                    "Failed to initialize vector index"
                ) from fallback_error

    def get_retriever(self, top_k: int = 20):
        """Get a retriever for the vector index."""
        return self.vector_index.as_retriever(search_kwargs={"k": top_k})

    def search(
        self, query: str, exclude_ids: Optional[List[str]] = None, top_k: int = None
    ) -> List[Document]:
        """Perform a vector search.

        Args:
            query: The search query
            exclude_ids: List of event IDs to exclude from results
            top_k: Maximum number of results to return

        Returns:
            List of Document objects
        """
        if not query:
            raise ValueError("query must be provided")

        if top_k is None:
            if "vector_top_k" not in SEARCH_CONFIG:
                raise ValueError("SEARCH_CONFIG must contain 'vector_top_k'")
            top_k = SEARCH_CONFIG["vector_top_k"]

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        try:
            # Perform the basic vector search
            results = self.vector_index.similarity_search(query, k=top_k)

            # If exclude_ids is provided, filter the results
            if exclude_ids:
                logger.info(
                    f"Filtering out {len(exclude_ids)} event IDs from vector search results"
                )
                results = self.filter_documents_by_event_ids(
                    results, exclude_ids, exclude=True
                )

                # If we filtered out too many results, get more
                if len(results) < top_k / 2:
                    logger.info(
                        f"After filtering, only {len(results)} results remain. Getting more results."
                    )
                    # Get more results to compensate for the filtered ones
                    additional_results = self.vector_index.similarity_search(
                        query, k=top_k
                    )
                    additional_filtered = self.filter_documents_by_event_ids(
                        additional_results, exclude_ids, exclude=True
                    )

                    # Add unique additional results
                    existing_ids = {
                        doc.metadata.get("event_id")
                        for doc in results
                        if "event_id" in doc.metadata
                    }
                    for doc in additional_filtered:
                        if (
                            "event_id" in doc.metadata
                            and doc.metadata["event_id"] not in existing_ids
                        ):
                            results.append(doc)
                            existing_ids.add(doc.metadata["event_id"])
                            if len(results) >= top_k:
                                break

            return results[:top_k]
        except Exception as e:
            logger.error(f"Error executing vector search: {e}")
            return []

    def filter_documents_by_event_ids(
        self, documents: List[Document], event_ids: List[str], exclude: bool = False
    ) -> List[Document]:
        """Filter vector search results based on event IDs.

        Args:
            documents: List of documents to filter
            event_ids: List of event IDs to include or exclude
            exclude: If True, exclude the event_ids; if False, include only the event_ids

        Returns:
            Filtered list of documents
        """
        filtered_docs = []
        event_ids_set = set(str(event_id) for event_id in event_ids)

        for doc in documents:
            if not hasattr(doc, "metadata") or not doc.metadata:
                continue

            # Try different possible keys for event ID in metadata
            event_id = None
            for key in ["event_id", "id", "eventId", "event"]:
                if key in doc.metadata:
                    event_id = str(doc.metadata[key])
                    break

            # If no event ID found, skip this document
            if not event_id:
                continue

            # Include or exclude based on the exclude parameter
            if exclude:
                # If exclude is True, keep documents NOT in event_ids_set
                if event_id not in event_ids_set:
                    filtered_docs.append(doc)
            else:
                # If exclude is False, keep documents IN event_ids_set
                if event_id in event_ids_set:
                    filtered_docs.append(doc)

        return filtered_docs

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
        model: str,
    ) -> List[Document]:
        """Rerank documents using the specified reranker.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Maximum number of results to return
            model: Reranker model to use

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not model:
            raise ValueError("model must be provided")

        try:
            # Use the reranker utility function with dictionary config
            reranked_docs = rerank_documents(
                query,
                documents,
                {"model": model, "top_k": top_k},  # Pass config as dictionary
            )

            # Sort documents by reranker score (highest first)
            sorted_docs = sorted(
                reranked_docs,
                key=lambda x: x.metadata.get("reranker_score", 0),
                reverse=True,
            )

            logger.info(f"Reranked {len(sorted_docs)} documents")
            return sorted_docs[:top_k]

        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return documents  # Return original documents on error
