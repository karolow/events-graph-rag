import os

import dotenv
from langchain_neo4j import Neo4jGraph

dotenv.load_dotenv(".env", override=True)

# Neo4j connection parameters
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
    e.start_date = CASE WHEN row.start <> '' AND row.start <> '-' THEN row.start ELSE null END,
    e.end_date = CASE WHEN row.end <> '' AND row.end <> '-' THEN row.end ELSE null END,
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
graph.refresh_schema()
print(graph.get_schema)
