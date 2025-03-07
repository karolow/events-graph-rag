# %%
import logging
import os
from typing import Any

import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Neo4jVector
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
        text_node_properties=["combined_text"],
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
        text_node_properties=["combined_text"],
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

# Initialize LLM
# llm = ChatOpenAI(temperature=0, model="gpt-4o")
llm = ChatMistralAI(temperature=0, model_name="mistral-large-latest")

# Create a custom prompt template for the Cypher generation
cypher_generation_template = """
You are an expert in generating Cypher queries for Neo4j.

Neo4j Graph Schema:
{schema}

The user has asked the following question:
{query}

Based on semantic search, these events might be relevant:
{relevant_docs}

Generate a Cypher query to answer the user's question.
Important guidelines:
1. If you're counting entities that could appear multiple times, use COUNT(DISTINCT entity) to avoid duplicate counting.
2. Always use case-insensitive comparison with TOLOWER() function.
3. For keyword searches, be comprehensive by:
   - Using multiple CONTAINS clauses to catch word variations
   - Combining them with OR operators
   - Example: WHERE (TOLOWER(e.name) CONTAINS 'workshop' OR TOLOWER(e.name) CONTAINS 'workshops')
4. When appropriate, use regular expressions for more flexible matching:
   - Example: WHERE e.name =~ '(?i).*workshop.*'
5. IMPORTANT: Events have these relationships:
   - (Event)-[:HAS_TOPIC]->(Tag)
   - (Event)-[:BELONGS_TO]->(Category)
   - (Event)-[:TAKES_PLACE_IN]->(Location)
   - (Event)-[:PART_OF]->(Project)
   - (Coordinator)-[:COORDINATES]->(Event)
   - (Guest)-[:PARTICIPATES_IN]->(Event)
6. When handling participant exclusions:
   - Use NOT EXISTS subqueries to ensure excluded participants are not present
   - Example for excluding a participant:
     WHERE NOT EXISTS {{ MATCH (e)<-[:PARTICIPATES_IN]-(g2:Guest) WHERE TOLOWER(g2.name) = 'monika malcherek' }}
7. For participant inclusion/exclusion patterns:
   - Use separate MATCH clauses for required participants
   - Use WHERE NOT EXISTS for exclusions

Remember to always use case-insensitive comparison with TOLOWER() function for all text comparisons.

Take into account the semantically relevant events listed above.
If the question mentions specific topics or concepts, expand your search to include variations and related terms.

Cypher Query:
"""

cypher_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema", "query", "relevant_docs"],
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
            **kwargs,
        )

        return chain

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        # First retrieve relevant documents using vector search
        query = inputs["query"]
        relevant_docs = self.vector_retriever.invoke(query)
        logger.info(f"Relevant events retrieved: {len(relevant_docs)}")

        # Extract event names from content
        relevant_events = [
            self.document_processor.extract_title_from_content(doc.page_content)
            for doc in relevant_docs
        ]
        logger.info(f"Extracted event names: {relevant_events}")

        # Format documents for the prompt
        formatted_docs = self.document_processor.format_docs_for_prompt(relevant_docs)

        # Add the retrieved events to the context
        enhanced_inputs = {
            "query": query,
            "relevant_events": relevant_events,
            "relevant_docs": formatted_docs,
            "schema": self.graph.get_schema,
        }

        # Call the parent class method with enhanced inputs
        return super()._call(enhanced_inputs, run_manager)


qa_template = """
You are an expert in analyzing Neo4j graph database query results for cultural events.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

The user has asked: {question}

Based on the Cypher query results from the Neo4j database, provide a detailed and accurate answer.

Context from database query results:
{function_response}

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

Based solely on the provided database results, answer the user's question:
"""

qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["schema", "question", "function_response"],
)

# Create the hybrid search chain with custom prompts
hybrid_search_chain = HybridNeo4jSearchChain.from_llm(
    cypher_llm=llm,
    qa_llm=llm,
    graph=graph,
    vector_retriever=vector_index.as_retriever(search_kwargs={"k": 5}),
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    verbose=True,
    use_function_response=True,
    allow_dangerous_requests=True,
)

# %%
# Example query based on use cases from README
result = hybrid_search_chain.invoke(
    # {"query": "Find music outdoor events with more than 100 participants"}
    {"query": "Find theatre events with more than 100 participants"}
)

logger.info(f"Query result: {result}")

# %%
