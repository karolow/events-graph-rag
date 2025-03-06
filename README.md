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

source: https://neo4j.com/docs/operations-manual/current/docker/introduction/

## Relationships

(:Event)-[:HAS_TOPIC]->(:Tag)
(:Event)-[:BELONGS_TO]->(:Category)
(:Event)-[:TAKES_PLACE_IN]->(:Location)
(:Event)-[:PART_OF]->(:Project)
(:Coordinator)-[:COORDINATES]->(:Event)
(:Coordinator)-[:COORDINATES]->(:Project)
(:Guest)-[:PARTICIPATES_IN]->(:Event)