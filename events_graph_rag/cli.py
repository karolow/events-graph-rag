import click

from events_graph_rag.config import logger
from events_graph_rag.hybrid_search import HybridSearch
from events_graph_rag.neo4j_client import Neo4jClient


@click.group()
def cli() -> None:
    """Events Graph RAG CLI tool."""
    pass


@cli.command()
@click.option(
    "--csv-url",
    default=None,
    help="URL to CSV file with events data. Defaults to sample dataset.",
)
def load(csv_url: str | None) -> None:
    """Load events data from CSV and create embeddings."""
    client = Neo4jClient()
    client.load_data_and_create_embeddings(csv_url)
    click.echo("Data loading and embedding generation complete!")


@cli.command()
@click.option(
    "--query",
    required=True,
    help="Natural language query to search for events.",
)
@click.option(
    "--verbose/--quiet",
    default=False,
    help="Show detailed search results including Cypher query and raw results.",
)
def search(query: str, verbose: bool) -> None:
    """Search events in the graph database using hybrid search."""
    logger.info(f"Executing search query: {query}")

    # Initialize the hybrid search
    hybrid_search = HybridSearch()

    # Execute the search
    results = hybrid_search(query)

    # Display the answer
    click.echo(f"Answer: {results['answer']}")

    # Log additional details if verbose mode is enabled
    if verbose:
        if "cypher_query" in results:
            logger.info(f"Cypher Query: {results['cypher_query']}")
            click.echo(f"\nCypher Query: {results['cypher_query']}")

        if "graph_results" in results:
            graph_count = len(results["graph_results"])
            logger.info(f"Found {graph_count} graph results")
            click.echo(f"\nGraph Results: {graph_count} items found")

            # Show first result as example if available
            if results["graph_results"] and graph_count > 0:
                logger.debug(f"First graph result: {results['graph_results'][0]}")
                click.echo(f"First result: {results['graph_results'][0]}")

        if "vector_results" in results:
            vector_count = len(results["vector_results"])
            logger.info(f"Found {vector_count} vector results")
            click.echo(f"\nVector Results: {vector_count} documents found")

    logger.info("Search completed successfully")


if __name__ == "__main__":
    cli()
