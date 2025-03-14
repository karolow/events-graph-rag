from typing import Any, Dict, List

from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document processing operations for Events data."""

    @staticmethod
    def format_docs_for_prompt(docs: List[Document]) -> str:
        """Format events for inclusion in a prompt."""
        formatted_docs = []

        for doc in docs:
            # Extract structured information from combined_text format
            event_info = {
                "Event": "Unknown Event",
                "Project": "",
                "Location": "",
            }

            for line in doc.page_content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    event_info[key.strip()] = value.strip()

            doc_info = f"Event: {event_info['Event']}"

            if event_info["Project"]:
                doc_info += f"\nProject: {event_info['Project']}"

            if event_info["Location"]:
                doc_info += f"\nLocation: {event_info['Location']}"

            formatted_docs.append(doc_info)

        return "\n\n".join(formatted_docs)

    @staticmethod
    def create_documents_from_search_results(
        results: List[Dict[str, Any]],
    ) -> List[Document]:
        """Create Document objects from search results."""
        documents = []

        for result in results:
            if "text" not in result or not result["text"]:
                continue

            # Create metadata dictionary
            metadata = {}
            for key in ["event_id", "id", "eventId", "event_name", "name"]:
                if key in result:
                    metadata[key] = result[key]

            # Create Document object
            doc = Document(page_content=result["text"], metadata=metadata)
            documents.append(doc)

        return documents

    @staticmethod
    def create_documents_from_vector_results(
        vector_results: List[tuple[str, float, Dict[str, str]]],
    ) -> List[Document]:
        """Create Document objects from vector search results."""
        documents = []

        for text, similarity_score, event_info in vector_results:
            # Create a Document object with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "event_id": event_info["id"],
                    "name": event_info["name"],
                    "vector_score": similarity_score,
                },
            )
            documents.append(doc)

        return documents

    @staticmethod
    def format_graph_results(results: List[Dict[str, Any]]) -> str:
        """Format graph results for inclusion in a prompt."""
        if not results:
            return "No results found from graph search."

        formatted_results = []

        for i, result in enumerate(results, 1):
            result_str = f"Result {i}:\n"

            # Format each key-value pair in the result
            for key, value in result.items():
                if isinstance(value, dict):
                    # Format nested dictionary
                    result_str += f"  {key}:\n"
                    for sub_key, sub_value in value.items():
                        result_str += f"    {sub_key}: {sub_value}\n"
                else:
                    # Format simple key-value pair
                    result_str += f"  {key}: {value}\n"

            formatted_results.append(result_str)

        return "\n".join(formatted_results)

    @staticmethod
    def format_vector_results(documents: List[Document]) -> str:
        """Format vector search results for inclusion in a prompt."""
        if not documents:
            return "No results found from vector search."

        formatted_docs = []

        for i, doc in enumerate(documents, 1):
            # Start with the document number
            doc_str = f"Document {i}:\n"

            # Add content
            doc_str += f"Content:\n{doc.page_content}\n"

            # Add metadata if available
            if hasattr(doc, "metadata") and doc.metadata:
                doc_str += "Metadata:\n"

                # First add the most important attributes in a specific order
                important_keys = [
                    "event_id",
                    "name",
                    "number_of_participants",
                    "start_date",
                    "start_time",
                    "location",
                    "category",
                ]
                for key in important_keys:
                    if key in doc.metadata:
                        value = doc.metadata[key]
                        # Format the value appropriately
                        if value is None:
                            formatted_value = "Not specified"
                        elif key == "number_of_participants" and value is not None:
                            formatted_value = str(value)
                        else:
                            formatted_value = str(value)
                        doc_str += f"  {key}: {formatted_value}\n"

                # Then add any remaining metadata
                for key, value in doc.metadata.items():
                    # Skip already processed keys and scores
                    if key in important_keys or key in [
                        "vector_score",
                        "score",
                        "reranker_score",
                    ]:
                        continue
                    doc_str += f"  {key}: {value}\n"

            formatted_docs.append(doc_str)

        return "\n".join(formatted_docs)
