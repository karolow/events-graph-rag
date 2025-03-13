import logging
import os

import dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv(".env", override=True)

url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "")
password = os.environ.get("NEO4J_PASSWORD", "")

graph = Neo4jGraph(url=url, username=username, password=password, enhanced_schema=True)

q_load_events = """
LOAD CSV WITH HEADERS
FROM 'https://raw.githubusercontent.com/karolow/datasets/refs/heads/main/events_kmo_sample.csv' AS row
FIELDTERMINATOR ';'
// Create Event node with a unique identifier (using the CSV id column)
MERGE (e:Event {id: row.id})
SET e.name = row.event,
    e.start_date_original = CASE WHEN row.start <> '' AND row.start <> '-' THEN row.start ELSE null END,
    e.end_date_original = CASE WHEN row.end <> '' AND row.end <> '-' THEN row.end ELSE null END,
    
    // Store the date in ISO format (YYYY-MM-DD) for easier querying
    // The format in CSV is now: "2021-10-02 19:00"
    e.start_date = CASE 
        WHEN row.start <> '' AND row.start <> '-' 
        THEN substring(row.start, 0, 10) // Extract just the YYYY-MM-DD part
        ELSE null 
    END,
    
    // Also store year-month for easier filtering
    e.start_date_year_month = CASE 
        WHEN row.start <> '' AND row.start <> '-' 
        THEN substring(row.start, 0, 7) // Extract just the YYYY-MM part
        ELSE null 
    END,
    
    // Store the time separately
    e.start_time = CASE 
        WHEN row.start <> '' AND row.start <> '-' 
        THEN substring(row.start, 11) // Extract just the HH:MM part
        ELSE null 
    END,
    
    // Do the same for end date
    e.end_date = CASE 
        WHEN row.end <> '' AND row.end <> '-' 
        THEN substring(row.end, 0, 10) // Extract just the YYYY-MM-DD part
        ELSE null 
    END,
    
    // Also store year-month for easier filtering
    e.end_date_year_month = CASE 
        WHEN row.end <> '' AND row.end <> '-' 
        THEN substring(row.end, 0, 7) // Extract just the YYYY-MM part
        ELSE null 
    END,
    
    // Store the time separately
    e.end_time = CASE 
        WHEN row.end <> '' AND row.end <> '-' 
        THEN substring(row.end, 11) // Extract just the HH:MM part
        ELSE null 
    END,
    
    e.number_of_participants = CASE WHEN row.number_of_participants <> '' AND row.number_of_participants <> '-' THEN toInteger(row.number_of_participants) ELSE null END

// Create Project node
MERGE (p:Project {name: row.project})

// Create Category node and relationship if category exists and is not "-"
WITH e, p, row
WHERE row.category <> '' AND row.category <> '-'
MERGE (c:Category {name: row.category})
MERGE (e)-[:BELONGS_TO]->(c)

// Create Location node and relationship if location exists and is not "-"
WITH e, p, row
WHERE row.location <> '' AND row.location <> '-'
MERGE (l:Location {name: row.location})
MERGE (e)-[:TAKES_PLACE_IN]->(l)

// Connect Event to Project
WITH e, p, row
MERGE (e)-[:PART_OF]->(p)

// Process coordinators (may be multiple, comma-separated)
WITH e, p, row
FOREACH (coord IN CASE WHEN row.coordinator <> '' AND row.coordinator <> '-' THEN split(row.coordinator, ',') ELSE [] END |
    MERGE (coordinator:Coordinator {name: trim(coord)})
    MERGE (coordinator)-[:COORDINATES]->(e)
    MERGE (coordinator)-[:COORDINATES]->(p)
)

// Process guests/participants (comma-separated)
WITH e, p, row
FOREACH (guest IN CASE WHEN row.guests_surnames <> '' AND row.guests_surnames <> '-' THEN split(row.guests_surnames, ',') ELSE [] END |
    MERGE (g:Guest {name: trim(guest)})
    MERGE (g)-[:PARTICIPATES_IN]->(e)
)

// Process tags (comma-separated)
WITH e, p, row
FOREACH (tag IN CASE WHEN row.tags <> '' AND row.tags <> '-' THEN split(row.tags, ',') ELSE [] END |
    MERGE (t:Tag {name: trim(tag)})
    MERGE (e)-[:HAS_TOPIC]->(t)
)
"""

graph.query(q_load_events)


# Create index for vector search
def create_vector_index():
    try:
        # Try to create the vector index directly
        # If it already exists, this will fail with a specific error message
        create_index_query = """
        CALL db.index.vector.createNodeIndex(
          'events_vector_index',
          'Event',
          'embedding',
          1536,
          'cosine'
        )
        """
        graph.query(create_index_query)
        logger.info("Created vector index for Event nodes")
    except Exception as e:
        # Check if the error is because the index already exists
        if "already exists" in str(e):
            logger.info("Vector index already exists")
        else:
            # Try an alternative approach for older Neo4j versions
            try:
                # Check if we can list indexes using SHOW INDEXES
                check_index_query = """
                SHOW INDEXES
                WHERE name = 'events_vector_index'
                YIELD name
                RETURN count(*) > 0 AS exists
                """
                result = graph.query(check_index_query)
                if result and result[0].get("exists", False):
                    logger.info(
                        "Vector index already exists (verified with SHOW INDEXES)"
                    )
                else:
                    # Try to create the index with a different syntax for older Neo4j versions
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
                        graph.query(alt_create_index_query)
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


# Create the combined text property for events to include project and location
def create_combined_text_property():
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
    update_result = graph.query(create_combined_property_query)
    logger.info(
        f"Updated {update_result[0]['updated_count']} events with combined text for embeddings"
    )


# Generate and store embeddings for all events
def generate_embeddings():
    # Initialize vector embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector index for Events data using the combined text field
    vector_index = Neo4jVector.from_existing_graph(
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
    logger.info("Generated embeddings for all events")
    return vector_index


graph.refresh_schema()
logger.info("Neo4j Graph Schema loaded")

logger.info("Creating vector index...")
create_vector_index()

logger.info("Creating combined text property...")
create_combined_text_property()

logger.info("Generating embeddings...")
generate_embeddings()

logger.info("Data loading and embedding generation complete!")
