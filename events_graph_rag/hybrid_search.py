# %%
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from reranker import RerankerConfig, rerank_documents
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv(".env", override=True)

# Neo4j connection parameters
url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "")
password = os.environ.get("NEO4J_PASSWORD", "")


# %%
# Initialize Neo4j graph
graph = Neo4jGraph(url=url, username=username, password=password, enhanced_schema=True)
graph.refresh_schema()

# %%
# Initialize vector embeddings
embeddings = OpenAIEmbeddings()

# Load the existing vector index from the database
# The embeddings were already created during data loading
try:
    vector_index = Neo4jVector.from_existing_index(
        embeddings,
        url=url,
        username=username,
        password=password,
        index_name="events_vector_index",
        node_label="Event",
        text_node_property="combined_text",
        embedding_node_property="embedding",
        retrieval_query="""
        WITH node, score
        MATCH (e:Event) WHERE e = node
        RETURN e.combined_text AS text,
               elementId(e) AS id,
               {event_id: e.id, name: e.name} AS metadata,
               score
        """,
    )
    logger.info("Successfully loaded vector index from Neo4j")
except Exception as e:
    # Fallback to using the index name "events" if "events_vector_index" fails
    logger.warning(f"Failed to load 'events_vector_index': {e}")
    logger.info("Trying fallback index name 'events'...")
    vector_index = Neo4jVector.from_existing_index(
        embeddings,
        url=url,
        username=username,
        password=password,
        index_name="events",
        node_label="Event",
        text_node_property="combined_text",
        embedding_node_property="embedding",
        retrieval_query="""
        WITH node, score
        MATCH (e:Event) WHERE e = node
        RETURN e.combined_text AS text,
               elementId(e) AS id,
               {event_id: e.id, name: e.name} AS metadata,
               score
        """,
    )
    logger.info("Successfully loaded vector index 'events' from Neo4j")

# Initialize LLMs for different tasks
cypher_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
qa_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# Create a custom prompt template for the Cypher generation
cypher_generation_template = """
You are an expert Neo4j Cypher query generator for a cultural events database.

Schema of the database:
{schema}

The user has asked the following question:
{query}

Generate a Cypher query to answer the user's question.

IMPORTANT - SEMANTIC MAPPINGS:
When users search for certain terms, expand your search to include related concepts:
- "art event" → search for categories like: art, exhibition, gallery, museum, visual arts, painting, sculpture
- "music event" → search for categories like: music, concert, recital, performance, orchestra, band, choir
- "theater event" → search for categories like: theater, drama, play, performance, stage, acting
- "workshop" → search for categories like: workshop, class, seminar, training, education
- "festival" → search for categories like: festival, celebration, fair, carnival
- "conference" → search for categories like: conference, symposium, convention, meeting, summit

IMPORTANT - DATE HANDLING:
When working with dates:
1. The database stores dates in ISO format: 'YYYY-MM-DD' in the start_date and end_date fields
2. Times are stored separately in start_time and end_time fields in 'HH:MM' format
3. There's also a start_date_year_month field in format 'YYYY-MM' for easier month-based filtering
4. User queries might use various formats like 'MM/DD/YYYY', 'DD/MM/YYYY', or natural language
5. For date matching, use the following approach:
   - For exact date queries like "9/24/2021", convert to ISO format: WHERE e.start_date = '2021-09-24'
   - For month queries like "September 2021", use: WHERE e.start_date_year_month = '2021-09'
   - For time-specific queries, use the time field: WHERE e.start_time = '19:00'
   - For date ranges, use comparison operators: WHERE e.start_date >= '2021-09-01' AND e.start_date <= '2021-09-30'

Important guidelines:
1. If you're counting entities that could appear multiple times, use COUNT(DISTINCT entity) to avoid duplicate counting.
2. Always use case-insensitive comparison with toLower() function (not TOLOWER).
3. For keyword searches, be comprehensive by:
   - Using multiple CONTAINS clauses to catch word variations
   - Combining them with OR operators
   - Example: WHERE (toLower(c.name) CONTAINS 'art' OR toLower(c.name) CONTAINS 'exhibition' OR toLower(c.name) CONTAINS 'gallery')
4. When appropriate, use regular expressions for more flexible matching:
   - Example: WHERE e.name =~ '(?i).*(art|exhibition|gallery).*'
5. IMPORTANT: Events have these relationships:
   - (Event)-[:HAS_TOPIC]->(Tag)
   - (Event)-[:BELONGS_TO]->(Category)
   - (Event)-[:TAKES_PLACE_IN]->(Location)
   - (Event)-[:PART_OF]->(Project)
   - (Coordinator)-[:COORDINATES]->(Event)
   - (Guest)-[:PARTICIPATES_IN]->(Event)
6. When handling participant exclusions, use NOT EXISTS pattern.
7. Always use proper Neo4j syntax and avoid syntax errors.
8. For COUNT queries (when the user asks "how many"), just return the count without collecting IDs:
   - Use: RETURN COUNT(DISTINCT e) AS eventCount
   - This is more efficient for large result sets

Here are examples of well-formed Cypher queries:

Example 1: Find all music events with more than 50 participants
```cypher
MATCH (e:Event)-[:BELONGS_TO]->(c:Category)
WHERE (toLower(c.name) CONTAINS 'music' OR toLower(c.name) CONTAINS 'concert' OR toLower(c.name) CONTAINS 'performance') 
  AND e.number_of_participants > 50
RETURN e.name AS eventName, e.id AS eventId, e.number_of_participants AS participants
ORDER BY e.number_of_participants DESC
```

Example 2: Find events coordinated by John Smith
```cypher
MATCH (c:Coordinator)-[:COORDINATES]->(e:Event)
WHERE toLower(c.name) = 'john smith'
RETURN e.name AS eventName, e.id AS eventId, c.name AS coordinator
```

Example 3: Find art events with multiple coordinators
```cypher
MATCH (c:Coordinator)-[:COORDINATES]->(e:Event)-[:BELONGS_TO]->(cat:Category)
WHERE toLower(cat.name) CONTAINS 'art' OR toLower(cat.name) CONTAINS 'exhibition' OR toLower(cat.name) CONTAINS 'gallery'
WITH e, COUNT(DISTINCT c) AS coordinatorCount
WHERE coordinatorCount > 1
RETURN e.name AS eventName, e.id AS eventId, coordinatorCount
ORDER BY coordinatorCount DESC
```

Example 4: Find events where Alice participated but Bob did not
```cypher
MATCH (p1:Person {{name: 'Alice'}})-[:PARTICIPATED_IN]->(e:Event)
WHERE NOT EXISTS {{
  MATCH (p2:Person {{name: 'Bob'}})-[:PARTICIPATED_IN]->(e)
}}
RETURN e.name AS eventName, e.id AS eventId
```

Example 5: Count the best coordinator for an art event with many participants
```cypher
MATCH (c:Coordinator)-[:COORDINATES]->(e:Event)-[:BELONGS_TO]->(cat:Category)
WHERE (toLower(cat.name) CONTAINS 'art' OR toLower(cat.name) CONTAINS 'exhibition' OR toLower(cat.name) CONTAINS 'gallery' OR toLower(cat.name) CONTAINS 'museum')
  AND e.number_of_participants > 100
WITH c, COUNT(DISTINCT e) AS eventCount, SUM(e.number_of_participants) AS totalParticipants
RETURN c.name AS coordinator, eventCount, totalParticipants
ORDER BY eventCount DESC, totalParticipants DESC
LIMIT 5
```

Example 6: Count events on a specific date (9/24/2021)
```cypher
MATCH (e:Event)
WHERE e.start_date = '2021-09-24'
RETURN COUNT(e) AS eventCount
```

Example 7: Count events in a specific month (September 2021)
```cypher
MATCH (e:Event)
WHERE e.start_date_year_month = '2021-09'
RETURN COUNT(e) AS eventCount
```

Example 8: Count evening events (starting at or after 6 PM)
```cypher
MATCH (e:Event)
WHERE e.start_time >= '18:00'
RETURN COUNT(e) AS eventCount
```

Example 9: Count events by month in 2021
```cypher
MATCH (e:Event)
WHERE e.start_date STARTS WITH '2021-'
WITH substring(e.start_date, 0, 7) AS month, COUNT(e) AS eventCount
RETURN month, eventCount
ORDER BY month
```

IMPORTANT: Format your response as follows:
1. First provide a brief explanation of your approach
2. Then provide ONLY the Cypher query enclosed in triple backticks like:
```cypher
YOUR QUERY HERE
```
3. Do not include any explanatory text within the triple backticks

Cypher Query:
"""

cypher_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema", "query"],
)


class DocumentProcessor:
    """Handles document processing operations for Events data."""

    @staticmethod
    def extract_title_from_content(content: str) -> str:
        """Extract event name from document content."""
        # Directly split lines and look for Event: prefix
        for line in content.split("\n"):
            if line.lower().startswith("event:"):
                return line.split(":", 1)[1].strip()
        # Fallback to first line if no event: prefix found
        return content.split("\n")[0].strip()

    @staticmethod
    def format_docs_for_prompt(docs: list) -> str:
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


class HybridNeo4jSearchChain(GraphCypherQAChain):
    vector_retriever: BaseRetriever
    document_processor: DocumentProcessor
    expansion_llm: Optional[BaseLanguageModel] = None
    reranker_config: Optional[RerankerConfig] = None
    _expansion_executor: ThreadPoolExecutor = None
    _expanded_query_cache: Dict[str, str] = {}

    @classmethod
    def from_llm(
        cls,
        cypher_llm: BaseLanguageModel,
        qa_llm: BaseLanguageModel,
        graph: Neo4jGraph,
        vector_retriever: BaseRetriever,
        cypher_prompt: BasePromptTemplate | None = None,
        qa_prompt: BasePromptTemplate | None = None,
        document_processor: DocumentProcessor | None = None,
        expansion_llm: BaseLanguageModel | None = None,
        reranker_config: RerankerConfig | None = None,
        **kwargs: Any,
    ) -> "HybridNeo4jSearchChain":
        if cypher_prompt or qa_prompt:
            # Create a standard chain first
            standard_chain = GraphCypherQAChain.from_llm(
                cypher_llm=cypher_llm, qa_llm=qa_llm, graph=graph, **kwargs
            )

            from langchain_core.runnables import RunnablePassthrough

            # Format the cypher generation input
            format_cypher_input = RunnablePassthrough.assign(
                schema=lambda _: graph.get_schema,
            )

            # Extract content from the AI message
            def extract_message_content(message: Any) -> str:
                if hasattr(message, "content"):
                    return message.content
                return str(message)

            # Create generation chain with custom prompt + content extraction if specified
            if cypher_prompt:
                cypher_generation_chain = (
                    format_cypher_input
                    | cypher_prompt
                    | cypher_llm
                    | extract_message_content
                )
                standard_chain.cypher_generation_chain = cypher_generation_chain

            # Modify the QA chain if we have a custom QA prompt
            if qa_prompt:
                # Create a QA chain with the custom prompt
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.runnables import RunnablePassthrough

                # This assigns the schema to the QA input
                format_qa_input = RunnablePassthrough.assign(
                    schema=lambda _: graph.get_schema,
                )

                qa_chain = format_qa_input | qa_prompt | qa_llm | StrOutputParser()
                standard_chain.qa_chain = qa_chain

        else:
            # Use the default prompts
            standard_chain = GraphCypherQAChain.from_llm(
                cypher_llm=cypher_llm, qa_llm=qa_llm, graph=graph, **kwargs
            )

        # Create our custom chain instance with document processor
        chain = cls(
            cypher_generation_chain=standard_chain.cypher_generation_chain,
            qa_chain=standard_chain.qa_chain,
            graph=graph,
            graph_schema=standard_chain.graph_schema,
            vector_retriever=vector_retriever,
            document_processor=document_processor or DocumentProcessor(),
            expansion_llm=qa_llm,
            reranker_config=reranker_config,
            **kwargs,
        )

        # Initialize the thread pool executor for async query expansion
        chain._expansion_executor = ThreadPoolExecutor(max_workers=2)

        return chain

    def _get_events_from_graph_results(self, results: List[dict]) -> List[str]:
        """Extract event IDs from graph search results."""
        event_ids = []

        # Check if this is a COUNT query result (only contains count values)
        is_count_query = (
            all(
                key.lower().endswith("count")
                for result in results
                for key in result.keys()
                if not key.lower().startswith("event")
            )
            if results
            else False
        )

        # If this is a count query, return empty list as we don't need event IDs for count queries
        if is_count_query and results:
            logger.info(
                "Detected COUNT query result. Skipping event ID extraction for count-only query."
            )
            return event_ids

        for result in results:
            # Check for direct event_id or eventId in the result
            if "event_id" in result:
                event_ids.append(str(result["event_id"]))
                continue

            # Check for eventId (camelCase) which is what the Cypher query returns
            if "eventId" in result:
                event_ids.append(str(result["eventId"]))
                continue

            # Check if the result contains an Event node with id property
            if "e" in result:
                event_node = result["e"]
                # Use getattr for objects, get for dictionaries
                if hasattr(event_node, "__dict__"):  # It's an object
                    event_id = getattr(event_node, "id", None)
                    if event_id is not None:
                        event_ids.append(str(event_id))
                elif isinstance(event_node, dict) and "id" in event_node:
                    event_ids.append(str(event_node["id"]))
                continue

            # Check for event node with different key
            if "event" in result:
                event_node = result["event"]
                if hasattr(event_node, "__dict__"):  # It's an object
                    event_id = getattr(event_node, "id", None)
                    if event_id is not None:
                        event_ids.append(str(event_id))
                elif isinstance(event_node, dict) and "id" in event_node:
                    event_ids.append(str(event_node["id"]))
                continue

            # Check for any key that might contain an event node
            for key, value in result.items():
                if isinstance(value, dict) and "id" in value:
                    event_ids.append(str(value["id"]))
                    break
                elif hasattr(value, "__dict__"):  # It's an object
                    event_id = getattr(value, "id", None)
                    if event_id is not None:
                        event_ids.append(str(event_id))
                        break

        # Remove duplicates and ensure all IDs are strings
        return list(set(event_ids))

    def _filter_vector_results_by_events(
        self, vector_docs: List[Document], event_ids: List[str]
    ) -> List[Document]:
        """Filter vector search results to only include events from the graph search."""
        filtered_docs = []

        # Ensure all event_ids are strings for consistent comparison
        event_ids_set = set(str(event_id) for event_id in event_ids)

        for doc in vector_docs:
            # Check if the document has metadata
            if not hasattr(doc, "metadata") or not doc.metadata:
                continue

            # Try different possible keys for event ID in metadata
            event_id = None
            for key in ["event_id", "id", "eventId", "event"]:
                if key in doc.metadata:
                    event_id = str(doc.metadata[key])
                    break

            # If we found an event ID and it's in our set, include this document
            if event_id and event_id in event_ids_set:
                filtered_docs.append(doc)

        return filtered_docs

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        # First perform graph search
        query = inputs["query"]

        # Start async LLM query expansion immediately
        # This will run in parallel with the graph search
        self._start_async_query_expansion(query)

        # Get a filtered version of the schema with only essential information
        schema = self._get_filtered_schema()

        # Prepare inputs for graph search
        graph_inputs = {
            "query": query,
            "schema": schema,
        }

        # Generate and execute Cypher query
        llm_response = self.cypher_generation_chain.invoke(graph_inputs)
        logger.info(f"LLM response: {llm_response}")

        # Extract the actual Cypher query from the LLM's response
        cypher_query = self._extract_cypher_query(llm_response)
        logger.info(f"Extracted Cypher query: {cypher_query}")

        # Execute the Cypher query
        graph_results = self.graph.query(cypher_query)
        logger.info(f"Graph search returned {len(graph_results)} results")

        # Extract event IDs from graph results
        event_ids = self._get_events_from_graph_results(graph_results)
        logger.info(f"Extracted {len(event_ids)} event IDs from graph results")

        # Extract event names for logging
        event_names = self._extract_event_names_from_graph_results(graph_results)
        logger.info(f"Events from graph search: {event_names}")

        # Check if this is a COUNT query
        is_count_query = (
            "COUNT" in cypher_query.upper()
            and all(
                key.lower().endswith("count")
                for result in graph_results
                for key in result.keys()
                if not key.lower().startswith("event")
            )
            if graph_results
            else False
        )

        # Filter graph results to only include essential attributes
        filtered_results = self._filter_results_attributes(graph_results)

        # Initialize vector results as empty
        vector_results_text = []
        vector_docs = []

        # For COUNT queries, we don't do targeted vector search
        # Instead, we'll rely solely on the graph results
        if is_count_query:
            logger.info(
                "COUNT query detected. Skipping vector search as this is a count-only query."
            )
        # Only perform vector search if we have graph results with event IDs
        elif event_ids:
            # Perform targeted vector search only on the events found by graph search
            # Now returns tuples of (text, similarity_score, event_info)
            vector_results = self._perform_targeted_vector_search(query, event_ids)

            # Convert the vector results to Document objects for reranking
            if vector_results:
                for text, similarity_score, event_info in vector_results:
                    # Create a Document object with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "event_id": event_info["id"],
                            "name": event_info["name"],
                            "vector_score": similarity_score,  # Store original vector score
                            "score": 0.0,  # Will be updated by reranker
                        },
                    )
                    vector_docs.append(doc)

                # Rerank documents if reranker is configured
                if self.reranker_config and vector_docs:
                    # Use the simplified reranker utility function
                    vector_docs = rerank_documents(
                        query, vector_docs, self.reranker_config
                    )

                    # Log reranking results
                    logger.info(f"Reranked {len(vector_docs)} documents")
                    for doc in vector_docs:
                        event_name = doc.metadata.get("name", "Unknown")
                        event_id = doc.metadata.get("event_id", "Unknown")
                        reranker_score = doc.metadata.get("reranker_score", 0)
                        logger.info(
                            f"Reranked Event: {event_name} (ID: {event_id}) - Relevance score: {reranker_score:.4f}"
                        )

                # Sort documents by reranker score (highest first) to ensure best results are used first
                vector_docs = sorted(
                    vector_docs,
                    key=lambda x: x.metadata.get("reranker_score", 0),
                    reverse=True,
                )

                # Extract text content from all documents for QA chain
                vector_results_text = [doc.page_content for doc in vector_docs]

            logger.info(
                f"Performed vector search on {len(event_ids)} events from graph search"
            )

        # Call the QA chain with filtered results
        qa_result = self.qa_chain.invoke(
            {
                "schema": schema,
                "question": query,
                "graph_results": filtered_results,
                "vector_results": vector_results_text,
            }
        )

        # Return the result as a dictionary
        return {
            "result": qa_result,
            "graph_results": graph_results,
            "vector_docs": vector_docs,
        }

    def _extract_key_search_terms(self, query: str) -> str:
        """Extract key search terms from the user query for more focused vector search.

        This method identifies the most important semantic terms in the query,
        filtering out structural language like "Find all" or "with more than",
        and expands key terms with related concepts for better semantic matching.
        """
        # Check if we have a cached expanded query
        if query in self._expanded_query_cache:
            logger.info(f"Using cached expanded query for: '{query}'")
            return self._expanded_query_cache[query]

        # Try to use the LLM-based expansion
        try:
            # Check if we have an async expansion in progress
            if hasattr(self, "_expansion_future") and query in getattr(
                self, "_expansion_future", {}
            ):
                # Try to get the result if it's ready
                future = self._expansion_future[query]
                if future.done():
                    expanded_query = future.result()
                    logger.info(f"Using async LLM expanded query: '{expanded_query}'")
                    return expanded_query
        except Exception as e:
            logger.warning(f"Error retrieving async expansion: {str(e)}")

        # If we reach here, we don't have an expanded query yet
        # Extract simple key terms as a fallback
        import re

        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', query)

        # Basic list of filter words
        filter_words = {
            "find",
            "all",
            "with",
            "more",
            "than",
            "and",
            "or",
            "the",
            "that",
            "for",
            "in",
            "on",
            "at",
        }

        # Split the query and filter out common words
        words = query.lower().split()
        key_terms = [
            word for word in words if word not in filter_words and len(word) > 2
        ]

        # Combine quoted phrases and key terms
        all_terms = quoted_phrases + key_terms

        # If we have no terms, fall back to the original query
        if not all_terms:
            logger.info(f"No key terms extracted, using original query: '{query}'")
            return query

        # Join the terms into a search query
        search_terms = " ".join(all_terms)
        logger.info(f"Using simple fallback query expansion: '{search_terms}'")
        return search_terms

    def _llm_expand_query(self, query: str) -> str:
        """Use an LLM to expand the query with semantically related terms.

        This method sends the query to an LLM and asks it to extract and expand
        key search terms with related concepts for better semantic matching.
        """
        if not self.expansion_llm:
            logger.warning("No expansion LLM available for query expansion")
            return query

        try:
            # Create a prompt for the LLM that includes word filtering guidance
            prompt = f"""Extract and expand the key search terms from this query. 
            Focus on the most important semantic concepts and add related terms that would help in a vector search.
            
            FILTER OUT common words like:
            - Articles (the, a, an)
            - Prepositions (in, on, at, to, from, by, of)
            - Conjunctions (and, or, but)
            - Common verbs (find, show, list, display, return, give)
            - Filler words (all, with, more, than, about, for)
            
            KEEP important semantic terms related to:
            - Event types (concert, exhibition, workshop, festival, conference)
            - Art forms (music, visual art, theater, dance)
            - Specific genres (jazz, classical, rock, contemporary)
            - Locations and names
            
            For example, if the query mentions 'jazz concert', you might add terms like 'music', 'performance', 'band', etc.
            
            ONLY return the expanded search terms as a space-separated list of words. Do not include any explanations or other text.
            
            Query: {query}
            
            Expanded search terms: """

            # Get the expanded query from the LLM
            response = self.expansion_llm.invoke(prompt)

            # Extract content from the response
            if hasattr(response, "content"):
                expanded_query = response.content.strip()
            else:
                expanded_query = str(response).strip()

            # Cache the result
            self._expanded_query_cache[query] = expanded_query

            logger.info(f"LLM expanded query: '{query}' → '{expanded_query}'")
            return expanded_query
        except Exception as e:
            logger.warning(f"Error in LLM query expansion: {str(e)}")
            return query

    def _start_async_query_expansion(self, query: str):
        """Start asynchronous query expansion using an LLM.

        This method initiates the query expansion in a separate thread so it doesn't
        block the main search process.
        """
        if not hasattr(self, "_expansion_future"):
            self._expansion_future = {}

        # Submit the expansion task to the thread pool
        self._expansion_future[query] = self._expansion_executor.submit(
            self._llm_expand_query, query
        )
        logger.info(f"Started async query expansion for: '{query}'")

    def _perform_targeted_vector_search(
        self, query: str, event_ids: List[str]
    ) -> List[tuple[str, float, Dict[str, str]]]:
        """Perform vector search only on specific events from graph search."""
        # If no event IDs, return empty list
        if not event_ids:
            logger.warning("No event IDs provided for vector search")
            return []

        logger.info(f"Performing vector search on event IDs: {event_ids}")

        # Start async LLM query expansion for future searches
        self._start_async_query_expansion(query)

        # Extract key search terms for more focused vector search
        search_query = self._extract_key_search_terms(query)
        logger.info(f"Using search query for vector search: '{search_query}'")

        # Create a Cypher query to get the combined_text for these specific events
        event_ids_str = ", ".join([f"'{event_id}'" for event_id in event_ids])
        cypher_query = f"""
        MATCH (e:Event)
        WHERE e.id IN [{event_ids_str}]
        RETURN e.combined_text AS text, e.id AS event_id, e.name AS event_name
        """

        try:
            # Execute the query to get the text content
            results = self.graph.query(cypher_query)
            logger.info(f"Retrieved {len(results)} event texts for vector search")

            if not results:
                logger.warning("No text content found for the specified events")
                return []

            # Extract the text content and keep track of which text belongs to which event
            texts = []
            event_info = []
            for result in results:
                if "text" in result and result["text"]:
                    texts.append(result["text"])
                    event_info.append(
                        {
                            "id": result.get("event_id", "unknown"),
                            "name": result.get("event_name", "unknown"),
                        }
                    )
                    logger.debug(
                        f"Processing event: {result.get('event_name', 'unknown')} (ID: {result.get('event_id', 'unknown')})"
                    )

            if not texts:
                logger.warning("No valid text content found for the specified events")
                return []

            # Get the embeddings directly from our embeddings object
            # We know we're using OpenAIEmbeddings from the initialization
            embeddings = OpenAIEmbeddings()

            # Create embeddings for the search query (not the original query)
            logger.info(f"Creating embedding for search query: '{search_query}'")
            query_embedding = embeddings.embed_query(search_query)

            # Create embeddings for the texts
            logger.info(f"Creating embeddings for {len(texts)} event texts")
            text_embeddings = embeddings.embed_documents(texts)

            # Calculate similarity scores
            similarities = []
            for i, text_embedding in enumerate(text_embeddings):
                similarity = 1 - cosine(query_embedding, text_embedding)
                similarities.append((similarity, texts[i], event_info[i]))
                logger.info(
                    f"Event: {event_info[i]['name']} (ID: {event_info[i]['id']}) - Similarity score: {similarity:.4f}"
                )

            # Sort by similarity (highest first)
            similarities.sort(reverse=True)
            logger.info(
                f"Top event match: {similarities[0][2]['name']} (ID: {similarities[0][2]['id']}) with score {similarities[0][0]:.4f}"
            )

            # Return all texts with their similarity scores and event info
            # This allows us to rerank all documents and provide more comprehensive results
            return [(text, sim, info) for sim, text, info in similarities]

        except Exception as e:
            logger.error(f"Error performing targeted vector search: {e}")
            return []  # Return empty list in case of error

    def _get_event_ids_from_vector_docs(self, vector_docs: List[Document]) -> List[str]:
        """Extract event IDs from vector search documents."""
        event_ids = []

        for doc in vector_docs:
            if hasattr(doc, "metadata") and doc.metadata:
                for key in ["event_id", "id", "eventId", "event"]:
                    if key in doc.metadata:
                        event_ids.append(str(doc.metadata[key]))
                        break

        return list(set(event_ids))

    def _get_vector_context_for_events(
        self, vector_docs: List[Document], event_ids: List[str]
    ) -> List[str]:
        """Get vector search context for specific events."""
        context = []
        event_ids_set = set(str(event_id) for event_id in event_ids)

        for doc in vector_docs:
            if hasattr(doc, "metadata") and doc.metadata:
                for key in ["event_id", "id", "eventId", "event"]:
                    if key in doc.metadata and str(doc.metadata[key]) in event_ids_set:
                        context.append(doc.page_content)
                        break

        return context

    def _extract_event_names_from_graph_results(
        self, graph_results: List[dict]
    ) -> List[str]:
        """Extract event names from graph results for logging."""
        event_names = []

        for result in graph_results:
            name = None

            # Try to find event name in different possible locations
            if "e" in result and hasattr(result["e"], "name"):
                name = result["e"].name
            elif (
                "e" in result
                and isinstance(result["e"], dict)
                and "name" in result["e"]
            ):
                name = result["e"]["name"]
            elif "event" in result and hasattr(result["event"], "name"):
                name = result["event"].name
            elif (
                "event" in result
                and isinstance(result["event"], dict)
                and "name" in result["event"]
            ):
                name = result["event"]["name"]

            # If no name found but we have an eventId, use that
            if not name and "eventId" in result:
                name = f"Event ID: {result['eventId']}"

            if name:
                event_names.append(name)

        return event_names

    def _get_filtered_schema(self) -> str:
        """Get a filtered version of the schema with only essential information."""
        full_schema = self.graph.get_schema

        # Extract only the node types, relationship types, and key properties
        # Focus on the most relevant parts for our queries
        filtered_schema = ""

        # Extract node information for Event, Category, Tag, Location, Project, Coordinator, Guest
        key_nodes = [
            "Event",
            "Category",
            "Tag",
            "Location",
            "Project",
            "Coordinator",
            "Guest",
        ]
        node_pattern = r"Node:\s*(\w+)[\s\S]*?Properties:\s*([\s\S]*?)(?=\n\n|\Z)"

        for match in re.finditer(node_pattern, full_schema, re.DOTALL):
            node_type = match.group(1)
            if node_type in key_nodes:
                properties = match.group(2).strip()
                # Filter out embedding properties
                properties = re.sub(r"embedding:.*?\n", "", properties)
                filtered_schema += f"Node: {node_type}\nProperties: {properties}\n\n"

        # Extract relationship information
        rel_pattern = r"Relationship:\s*([\s\S]*?)(?=\n\n|\Z)"
        for match in re.finditer(rel_pattern, full_schema, re.DOTALL):
            filtered_schema += f"Relationship: {match.group(1)}\n\n"

        return filtered_schema

    def _filter_results_attributes(self, results: List[dict]) -> List[dict]:
        """Filter graph results to only include essential attributes."""
        filtered_results = []

        for result in results:
            filtered_result = {}

            for key, value in result.items():
                # Skip embedding attributes
                if key == "embedding" or key.endswith("_embedding"):
                    continue

                # Handle Neo4j node objects
                if hasattr(value, "__dict__"):
                    # Convert Neo4j node to dictionary with selected attributes
                    node_dict = {}
                    for attr_name in dir(value):
                        # Skip private attributes, methods, and embeddings
                        if (
                            attr_name.startswith("_")
                            or callable(getattr(value, attr_name))
                            or attr_name == "embedding"
                            or attr_name.endswith("_embedding")
                        ):
                            continue

                        # Add the attribute to our filtered dictionary
                        node_dict[attr_name] = getattr(value, attr_name)

                    filtered_result[key] = node_dict
                elif isinstance(value, dict):
                    # Filter dictionary attributes
                    filtered_dict = {
                        k: v
                        for k, v in value.items()
                        if k != "embedding" and not k.endswith("_embedding")
                    }
                    filtered_result[key] = filtered_dict
                else:
                    # Keep primitive values as is
                    filtered_result[key] = value

            filtered_results.append(filtered_result)

        return filtered_results

    def _extract_cypher_query(self, text: str) -> str:
        """Extract the Cypher query from the LLM's response.

        This method extracts the Cypher query from between triple backticks
        in the LLM's response.
        """
        # Extract the query from between triple backticks
        import re

        # Look for a query between ```cypher and ``` markers
        cypher_pattern = r"```(?:cypher)?\s*([\s\S]+?)\s*```"
        match = re.search(cypher_pattern, text, re.IGNORECASE)

        if match:
            query = match.group(1).strip()
            logger.debug(f"Extracted Cypher query: {query}")
            return query

        # If no query found between backticks, log a warning and return empty string
        logger.warning("Could not extract a Cypher query from LLM response")
        return ""


qa_template = """
You are an expert in analyzing Neo4j graph database query results for cultural events.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

The user has asked: {question}

Based on the Cypher query results from the Neo4j database, provide a detailed and accurate answer.

Context from database query results:
Graph Results:
{graph_results}

Vector Search Context (additional information):
{vector_results}

Important guidelines:
1. If the results are empty, explain that no matching events were found.
2. If there are specific counts or statistics in the query results, include them in your answer.
3. Format lists of events in a readable way.
4. If the question asked "how many", make sure to count the results and provide the number.
5. Always answer based purely on the data in the query results, not on general knowledge.
6. When returning lists of events, format them in a clear, structured way.
7. For events with dates, format them in a human-readable way.
8. For events with locations, include the location information in your answer.
9. For events with coordinators or guests, mention them when relevant.
10. Prioritize information from the Graph Results, and use Vector Search Context as supplementary information.

Based solely on the provided database results, answer the user's question:
"""

qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["schema", "question", "graph_results", "vector_results"],
)

# Create reranker configuration
reranker_config = RerankerConfig(
    model="jina-reranker-v2-base-multilingual",
    top_k=20,
)

# Create the hybrid search chain with custom prompts and reranker
hybrid_search_chain = HybridNeo4jSearchChain.from_llm(
    cypher_llm=cypher_llm,
    qa_llm=qa_llm,
    graph=graph,
    vector_retriever=vector_index.as_retriever(search_kwargs={"k": 20}),
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    reranker_config=reranker_config,
    verbose=True,
    use_function_response=True,
    allow_dangerous_requests=True,
)

# %%
# Example query based on use cases from README
result = hybrid_search_chain.invoke(
    {
        "query": "Find jazz concerts with more than 50 participants that took place after 31.03.2021"
    }
)

# Log the result - extract the actual result from the dictionary
if isinstance(result, dict) and "result" in result:
    logger.info(f"Query result: {result['result']}")
else:
    logger.info(f"Query result: {result}")

# Add a comment explaining the issue and solution
# The issue was that COUNT queries were only returning counts, not event IDs
# We've modified the code to handle COUNT queries by:
# 1. Detecting COUNT queries and skipping event ID extraction
# 2. Updating the Cypher generation prompt to only return counts for count queries
# 3. Modifying the _call method to skip vector search for COUNT queries
# This approach is more efficient for large result sets, as collecting all event IDs
# could be expensive and potentially overwhelm the model with too much data.
# For detailed information about specific events, users should phrase their queries
# to ask for specific events rather than counts.
# %%
