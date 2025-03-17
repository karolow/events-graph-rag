import re
from typing import Any

from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings

from events_graph_rag.config import (
    DATA_SOURCE_CONFIG,
    NEO4J_CONFIG,
    VECTOR_INDEX_CONFIG,
    logger,
)


class Neo4jClient:
    """Client for Neo4j graph database operations."""

    def __init__(
        self,
        url: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        """Initialize Neo4j client."""
        self.url = url or NEO4J_CONFIG["url"]
        self.username = username or NEO4J_CONFIG["username"]
        self.password = password or NEO4J_CONFIG["password"]
        self.graph = Neo4jGraph(
            url=self.url,
            username=self.username,
            password=self.password,
            enhanced_schema=True,
        )
        self.graph.refresh_schema()
        logger.info("Connected to Neo4j database")

    def query(self, cypher_query: str) -> list[dict[str, Any]]:
        """Execute a Cypher query and return the results."""
        logger.info(f"Executing Cypher query: {cypher_query}")
        results = self.graph.query(cypher_query)
        logger.info(f"Query returned {len(results)} results")
        return results

    def get_schema(self) -> str:
        """Get the database schema."""
        return self.graph.get_schema

    def get_filtered_schema(self) -> str:
        """Get a filtered version of the schema with only essential information."""
        full_schema = self.get_schema()

        # Extract only the node types, relationship types, and key properties
        filtered_schema = ""

        # Extract node information for key node types
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

    def get_event_texts(self, event_ids: list[str]) -> list[dict[str, Any]]:
        """Get text content for specific events."""
        if not event_ids:
            return []

        event_ids_str = ", ".join([f"'{event_id}'" for event_id in event_ids])
        cypher_query = f"""
        MATCH (e:Event)
        WHERE e.id IN [{event_ids_str}]
        RETURN e.combined_text AS text, e.id AS event_id, e.name AS event_name
        """

        return self.query(cypher_query)

    def filter_results_attributes(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                        if (
                            not attr_name.startswith("_")
                            and not callable(getattr(value, attr_name))
                            and attr_name != "embedding"
                            and not attr_name.endswith("_embedding")
                        ):
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

    def extract_event_info(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Extract event information from various possible structures in event data.

        This helper method handles different formats of event data that might come
        from graph search results, normalizing the access to common fields.

        Args:
            event: A dictionary containing event data in various possible formats

        Returns:
            Dictionary with normalized event information (id, name, and other available fields)
        """
        # Initialize with default values
        event_info = {
            "id": "unknown_id",
            "name": "unnamed_event",
        }

        # Copy any other fields that might be useful
        for key, value in event.items():
            if key not in ["e", "event"] and not isinstance(value, dict):
                event_info[key] = value

        # Check for direct keys in the top-level dictionary
        if "eventId" in event:
            event_info["id"] = event["eventId"]
        elif "event_id" in event:
            event_info["id"] = event["event_id"]

        if "eventName" in event:
            event_info["name"] = event["eventName"]
        elif "name" in event and not isinstance(event["name"], dict):
            event_info["name"] = event["name"]

        # Check for nested event objects under 'e' key
        if "e" in event and isinstance(event["e"], dict):
            if "id" in event["e"] and (event_info["id"] == "unknown_id"):
                event_info["id"] = event["e"]["id"]
            if "name" in event["e"] and (event_info["name"] == "unnamed_event"):
                event_info["name"] = event["e"]["name"]

            # Copy any other fields from the nested object
            for key, value in event["e"].items():
                if key not in ["id", "name"] and key not in event_info:
                    event_info[key] = value

        # Check for nested event objects under 'event' key
        if "event" in event and isinstance(event["event"], dict):
            if "id" in event["event"] and (event_info["id"] == "unknown_id"):
                event_info["id"] = event["event"]["id"]
            if "name" in event["event"] and (event_info["name"] == "unnamed_event"):
                event_info["name"] = event["event"]["name"]

            # Copy any other fields from the nested object
            for key, value in event["event"].items():
                if key not in ["id", "name"] and key not in event_info:
                    event_info[key] = value

        return event_info

    def extract_event_ids(self, results: list[dict[str, Any]]) -> list[str]:
        """Extract event IDs from graph search results."""
        if not results:
            return []

        event_ids = []
        for result in results:
            # Use the helper method to extract event info
            event_info = self.extract_event_info(result)
            event_id = event_info["id"]

            # Only add non-default IDs that aren't already in the list
            if event_id != "unknown_id" and event_id not in event_ids:
                event_ids.append(event_id)

        return event_ids

    # Data loading functionality from load_data_to_neo4j.py
    def load_events_from_csv(self, csv_url: str | None = None) -> None:
        """Load events data from CSV into Neo4j."""
        csv_url = csv_url or DATA_SOURCE_CONFIG["default_csv_url"]

        q_load_events = f"""
        LOAD CSV WITH HEADERS
        FROM '{csv_url}' AS row
        FIELDTERMINATOR ';'
        // Create Event node with a unique identifier (using the CSV id column)
        MERGE (e:Event {{id: row.id}})
        SET e.name = row.event,
            e.start_date_original = CASE WHEN row.start <> '' AND row.start <> '-' THEN row.start ELSE null END,
            e.end_date_original = CASE WHEN row.end <> '' AND row.end <> '-' THEN row.end ELSE null END,
            
            // Store the date in ISO format (YYYY-MM-DD) for easier querying
            e.start_date = CASE 
                WHEN row.start <> '' AND row.start <> '-' 
                THEN substring(row.start, 0, 10)
                ELSE null 
            END,
            
            // Also store year-month for easier filtering
            e.start_date_year_month = CASE 
                WHEN row.start <> '' AND row.start <> '-' 
                THEN substring(row.start, 0, 7)
                ELSE null 
            END,
            
            // Store the time separately
            e.start_time = CASE 
                WHEN row.start <> '' AND row.start <> '-' 
                THEN substring(row.start, 11)
                ELSE null 
            END,
            
            // Do the same for end date
            e.end_date = CASE 
                WHEN row.end <> '' AND row.end <> '-' 
                THEN substring(row.end, 0, 10)
                ELSE null 
            END,
            
            // Also store year-month for easier filtering
            e.end_date_year_month = CASE 
                WHEN row.end <> '' AND row.end <> '-' 
                THEN substring(row.end, 0, 7)
                ELSE null 
            END,
            
            // Store the time separately
            e.end_time = CASE 
                WHEN row.end <> '' AND row.end <> '-' 
                THEN substring(row.end, 11)
                ELSE null 
            END,
            
            e.number_of_participants = CASE WHEN row.number_of_participants <> '' AND row.number_of_participants <> '-' THEN toInteger(row.number_of_participants) ELSE null END

        // Create Project node
        MERGE (p:Project {{name: row.project}})

        // Create Category node and relationship if category exists and is not "-"
        WITH e, p, row
        WHERE row.category <> '' AND row.category <> '-'
        MERGE (c:Category {{name: row.category}})
        MERGE (e)-[:BELONGS_TO]->(c)

        // Create Location node and relationship if location exists and is not "-"
        WITH e, p, row
        WHERE row.location <> '' AND row.location <> '-'
        MERGE (l:Location {{name: row.location}})
        MERGE (e)-[:TAKES_PLACE_IN]->(l)

        // Connect Event to Project
        WITH e, p, row
        MERGE (e)-[:PART_OF]->(p)

        // Process coordinators (may be multiple, comma-separated)
        WITH e, p, row
        FOREACH (coord IN CASE WHEN row.coordinator <> '' AND row.coordinator <> '-' THEN split(row.coordinator, ',') ELSE [] END |
            MERGE (coordinator:Coordinator {{name: trim(coord)}})
            MERGE (coordinator)-[:COORDINATES]->(e)
            MERGE (coordinator)-[:COORDINATES]->(p)
        )

        // Process guests/participants (comma-separated)
        WITH e, p, row
        FOREACH (guest IN CASE WHEN row.guests <> '' AND row.guests <> '-' THEN split(row.guests, ',') ELSE [] END |
            MERGE (g:Guest {{name: trim(guest)}})
            MERGE (g)-[:PARTICIPATES_IN]->(e)
        )

        // Process tags (comma-separated)
        WITH e, p, row
        FOREACH (tag IN CASE WHEN row.tags <> '' AND row.tags <> '-' THEN split(row.tags, ',') ELSE [] END |
            MERGE (t:Tag {{name: trim(tag)}})
            MERGE (e)-[:HAS_TOPIC]->(t)
        )
        """

        self.query(q_load_events)
        logger.info("Loaded events data from CSV")

    def create_vector_index(self) -> None:
        """Create vector index for Event nodes."""
        try:
            create_index_query = """
            CALL db.index.vector.createNodeIndex(
              'events_vector_index',
              'Event',
              'embedding',
              1536,
              'cosine'
            )
            """
            self.query(create_index_query)
            logger.info("Created vector index for Event nodes")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("Vector index already exists")
            else:
                try:
                    check_index_query = """
                    SHOW INDEXES
                    WHERE name = 'events_vector_index'
                    YIELD name
                    RETURN count(*) > 0 AS exists
                    """
                    result = self.query(check_index_query)
                    if result and result[0].get("exists", False):
                        logger.info(
                            "Vector index already exists (verified with SHOW INDEXES)"
                        )
                    else:
                        alt_create_index_query = """
                        CALL db.createNodeVectorIndex(
                          'events_vector_index',
                          'Event',
                          'embedding',
                          1536,
                          'cosine'
                        )
                        """
                        try:
                            self.query(alt_create_index_query)
                            logger.info(
                                "Created vector index for Event nodes (using alternative method)"
                            )
                        except Exception as inner_e:
                            logger.error(f"Failed to create vector index: {inner_e}")
                            logger.warning(
                                "Proceeding without vector index. You may need to create it manually."
                            )
                except Exception as check_e:
                    logger.error(f"Failed to check for existing index: {check_e}")
                    logger.warning(
                        "Proceeding without vector index. You may need to create it manually."
                    )

    def create_combined_text_property(self) -> None:
        """Create combined text property for events."""
        create_combined_property_query = """
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[:PART_OF]->(p:Project)
        OPTIONAL MATCH (e)-[:TAKES_PLACE_IN]->(l:Location)
        WITH e, 
             e.name AS event_name,
             COALESCE(p.name, "") AS project_name,
             COALESCE(l.name, "") AS location_name,
             COALESCE(e.start_date, "") AS start_date,
             COALESCE(e.start_time, "") AS start_time
        SET e.combined_text = event_name + 
                             CASE WHEN project_name <> "" THEN "\n" + project_name ELSE "" END +
                             CASE WHEN location_name <> "" THEN "\nLocation: " + location_name ELSE "" END +
                             CASE WHEN start_date <> "" THEN "\nDate: " + start_date ELSE "" END +
                             CASE WHEN start_time <> "" THEN " at " + start_time ELSE "" END
        RETURN count(e) as updated_count
        """
        update_result = self.query(create_combined_property_query)
        logger.info(
            f"Updated {update_result[0]['updated_count']} events with combined text for embeddings"
        )

    def generate_embeddings(self) -> None:
        """Generate and store embeddings for all events."""
        embeddings = OpenAIEmbeddings()

        index_name = VECTOR_INDEX_CONFIG.get("index_name", "events_vector_index")
        node_label = VECTOR_INDEX_CONFIG.get("node_label", "Event")
        text_node_property = VECTOR_INDEX_CONFIG.get(
            "text_node_property", "combined_text"
        )
        embedding_node_property = VECTOR_INDEX_CONFIG.get(
            "embedding_node_property", "embedding"
        )
        retrieval_query = VECTOR_INDEX_CONFIG.get(
            "retrieval_query",
            """
        WITH node, score
        MATCH (e:Event) WHERE e = node
        RETURN e.combined_text AS text,
               elementId(e) AS id,
               {event_id: e.id, name: e.name} AS metadata,
               score
        """,
        )

        Neo4jVector.from_existing_graph(
            embeddings,
            url=self.url,
            username=self.username,
            password=self.password,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=[text_node_property],
            embedding_node_property=embedding_node_property,
            retrieval_query=retrieval_query,
        )
        logger.info("Generated embeddings for all events")

    def load_data_and_create_embeddings(self, csv_url: str | None = None) -> None:
        """Load data from CSV and create embeddings."""
        self.graph.refresh_schema()
        logger.info("Neo4j Graph Schema loaded")

        logger.info("Loading events data from CSV...")
        self.load_events_from_csv(csv_url)

        logger.info("Creating vector index...")
        self.create_vector_index()

        logger.info("Creating combined text property...")
        self.create_combined_text_property()

        logger.info("Generating embeddings...")
        self.generate_embeddings()

        logger.info("Data loading and embedding generation complete!")
