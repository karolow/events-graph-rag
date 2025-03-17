# Hybrid (graph/semantic) RAG tool

Using natural language to explore databases unlocks new opportunities for non-technical users. This tool, built with Neo4j and LangChain, combines semantic search and graph queries to provide intuitive, code-free access to data insights.

## Launch Neo4j

```bash
docker run \
    --name neo4j \
    --restart always \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    --mount type=bind,source=$(pwd)/db_data,destination=/data \
    neo4j:latest
```
source: https://neo4j.com/docs/operations-manual/current/docker/introduction/

## Data Loading and Embedding Generation

The project uses a two-step process:

1. **Data Loading**: Load event data from CSV into Neo4j to:
   - Create the graph structure with nodes and relationships
   - Generate combined text properties for each event
   - Create embeddings for all events using OpenAI's embedding model
   - Store embeddings in the Neo4j database

2. **Semantic Search**: Search the graph database to:
   - Perform semantic searches using the pre-generated embeddings
   - Generate Cypher queries based on natural language questions
   - Return detailed answers based on the graph database

## CLI Usage

The project provides a command-line interface for easy interaction with the system:

### Installation

```bash
uv venv --python 3.13
uv sync
```

### Available Commands

#### Load Data

Load events data from a CSV file and create embeddings:

```bash
# Load data using the default sample dataset
events load

# Load data from a custom CSV URL
events load --csv-url https://example.com/events_data.csv
```

#### Search Events

Search for events using natural language queries:

```bash
# Basic search
events search --query "Find jazz concerts with more than 50 participants"

# Verbose mode (shows detailed results including Cypher query and raw results)
events search --query "Find outdoor music events" --verbose
```

### Examples

```bash
# Load sample data
events load

# Search for specific events
events search --query "What are the music events that have more than 1 coordinator and more than 50 participants?"

# Get detailed search results
events search --query "Find events in which Alice Jones participated" --verbose
```

## Relationships

```
(:Event)-[:HAS_TOPIC]->(:Tag)
(:Event)-[:BELONGS_TO]->(:Category)
(:Event)-[:TAKES_PLACE_IN]->(:Location)
(:Event)-[:PART_OF]->(:Project)
(:Coordinator)-[:COORDINATES]->(:Event)
(:Coordinator)-[:COORDINATES]->(:Project)
(:Guest)-[:PARTICIPATES_IN]->(:Event)
```
### Event properties:

| KeyValue <id> | 4:c6515374-8168-481f-a45b-bcfd3d32f193:14 |
|---------------|------------------------------------------|
| id | "15" |
| name | "call for applications" |
| number_of_participants | 12 |
| start_date | "Sun Aug 01 2021 00:00:00 GMT+0200 (Central European Summer Time)" |
| end_date | "Wed Aug 25 2021 00:00:00 GMT+0200 (Central European Summer Time)" |
| combined_text | <to create embeddings from "project", "event" & "location"> |
| embedding | <actual embedding> |

## Use cases

1. Find all events coordinated by a given coordinator
2. Find music outdoor events with more than 100 participants 
3. Find events in which one person and not the other person participated
4. Find events in places untypical for culture events
5. Find music workshops with more than one coordinator

## Battle-tested examples

1. What are the music events that have more than 1 coordinator and more than 50 participants has taken place after 31.01.2021?
2. Find events in which Alice Jones participated alone / with other people.
3. Find outdoor music events with more than 100 participants.
4. Find jazz concerts with more than 50 participants that took place after 31.03.2021

**Hybrid Search Flow (Textual Representation)**

1.  **User Query:** The process begins with a user submitting a query.
2.  **Initiate Search:** The search process starts.
3.  **Generate Cypher:** A Cypher query is generated based on the user's query.
4.  **Cypher LLM:** A Language Model assists in crafting the Cypher query.
5.  **Extract Cypher:** The generated Cypher query is extracted as text.
6.  **Execute Graph Query:** The Cypher query is executed against the graph database.
7.  **Filter Results:** The initial results from the graph query are filtered.
8.  **Vector Search Needed?:** A decision is made whether to perform vector search or not, based on the graph query results.

9. **Based on decision:**

    *   If **Yes:**
        * **Execute Vector Search:** The vector search process begins.
        * **Expand Query:** The search query is expanded for better vector search.
        * **Vector Search:** The vector search is executed.
        * **Re-rank:** The vector search results are re-ranked.
        * **Format Vector:** The re-ranked vector search results are formatted.
    *   **No:**
        * **Format Graph:** The graph search results are formatted.

10.  **Compose Answer:** Information from the graph and/or vector search is combined to generate an answer.
11. **Final Answer:** The composed answer is presented.

This textual representation conveys the essential steps and branching logic of the hybrid search flow.

## TODO: 
- convert dates to days of week

## Notes: 
- I used this tutorial https://huggingface.co/learn/cookbook/en/rag_with_knowledge_graphs_neo4j