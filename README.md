Based on https://huggingface.co/learn/cookbook/en/rag_with_knowledge_graphs_neo4j

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

1. **Data Loading**: Run `load_data_to_neo4j.py` to:
   - Load event data from CSV into Neo4j
   - Create the graph structure with nodes and relationships
   - Generate combined text properties for each event
   - Create embeddings for all events using OpenAI's embedding model
   - Store embeddings in the Neo4j database

2. **Semantic Search**: Use `hybrid_search.py` to:
   - Perform semantic searches using the pre-generated embeddings
   - Generate Cypher queries based on natural language questions
   - Return detailed answers based on the graph database

## Relationships

(:Event)-[:HAS_TOPIC]->(:Tag)
(:Event)-[:BELONGS_TO]->(:Category)
(:Event)-[:TAKES_PLACE_IN]->(:Location)
(:Event)-[:PART_OF]->(:Project)
(:Coordinator)-[:COORDINATES]->(:Event)
(:Coordinator)-[:COORDINATES]->(:Project)
(:Guest)-[:PARTICIPATES_IN]->(:Event)

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