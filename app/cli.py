"""CLI for Agent Memory."""

import asyncio
from typing import Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from app.core.config import settings
from app.services.memory_service import memory_service
from app.models.schemas import MemoryType, Episode, GraphNode, GraphEdge, GraphQuery

app = typer.Typer(help="Agent Memory CLI")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8002, "--port", "-p", help="Port to bind"),
):
    """Start the API server."""
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


@app.command()
def interactive():
    """Interactive memory session."""
    asyncio.run(_interactive_async())


async def _interactive_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] v{settings.app_version}"))
    console.print("[dim]Commands: add, search, episode, graph, stats, help, quit[/dim]")
    console.print()

    await memory_service.initialize()

    while True:
        try:
            cmd = Prompt.ask("[bold green]memory>[/bold green]").strip().lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                _show_help()
            elif cmd == "add":
                content = Prompt.ask("Content")
                memory_type = Prompt.ask("Type", choices=["short", "long"], default="short")
                if memory_type == "short":
                    item = memory_service.add_short_term(content)
                    console.print(f"[green]Added to short-term: {item.id}[/green]")
                else:
                    importance = float(Prompt.ask("Importance (0-1)", default="0.5"))
                    item = await memory_service.add_long_term(content, importance)
                    console.print(f"[green]Added to long-term: {item.id}[/green]")
            elif cmd == "search":
                query = Prompt.ask("Query")
                results = await memory_service.search_memory(
                    type("Query", (), {"query": query, "top_k": 5, "min_importance": 0.0, "memory_types": []})()
                )
                _print_search_results(results)
            elif cmd == "episode":
                await _handle_episode()
            elif cmd == "graph":
                await _handle_graph()
            elif cmd == "stats":
                stats = memory_service.get_stats()
                console.print(stats)
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
        except (KeyboardInterrupt, EOFError):
            break


def _show_help():
    console.print("[bold]Available commands:[/bold]")
    console.print("  add       - Add memory (short/long term)")
    console.print("  search    - Search memories")
    console.print("  episode   - Manage episodes")
    console.print("  graph     - Manage knowledge graph")
    console.print("  stats     - Show statistics")
    console.print("  help      - Show this help")
    console.print("  quit      - Exit")


def _print_search_results(results: list):
    if not results:
        console.print("[yellow]No results[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("Type", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Content")

    for r in results:
        content = r.item.content[:80] + "..." if len(r.item.content) > 80 else r.item.content
        table.add_row(r.memory_type.value, f"{r.score:.3f}", content)

    console.print(table)


async def _handle_episode():
    action = Prompt.ask("Action", choices=["add", "list", "get", "search", "delete"])
    if action == "add":
        title = Prompt.ask("Title")
        steps = []
        console.print("Enter steps (empty to finish):")
        while True:
            step = Prompt.ask("  Step", default="")
            if not step:
                break
            steps.append({"action": step, "timestamp": str(datetime.utcnow())})
        importance = float(Prompt.ask("Importance", default="0.5"))
        episode = Episode(title=title, steps=steps, importance=importance)
        episode_id = await memory_service.add_episode(episode)
        console.print(f"[green]Episode created: {episode_id}[/green]")
    elif action == "list":
        episodes = await memory_service.list_episodes(limit=10)
        table = Table(title="Episodes")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Importance", style="green")
        table.add_column("Steps")
        for ep in episodes:
            table.add_row(str(ep.id)[:8], ep.title, str(ep.importance), str(len(ep.steps)))
        console.print(table)
    elif action == "get":
        episode_id = Prompt.ask("Episode ID")
        episode = await memory_service.get_episode(episode_id)
        if episode:
            console.print(f"Title: {episode.title}")
            console.print(f"Steps: {len(episode.steps)}")
            for i, step in enumerate(episode.steps):
                console.print(f"  {i+1}. {step}")
        else:
            console.print("[red]Not found[/red]")
    elif action == "search":
        query = Prompt.ask("Search query")
        episodes = await memory_service.search_episodes(query)
        for ep in episodes:
            console.print(f"  {ep.title} ({ep.id})")
    elif action == "delete":
        episode_id = Prompt.ask("Episode ID")
        # Would need to add delete to service
        console.print("[yellow]Delete not implemented in service[/yellow]")


async def _handle_graph():
    action = Prompt.ask("Action", choices=["add-node", "add-edge", "neighbors", "path"])
    if action == "add-node":
        label = Prompt.ask("Label")
        node_type = Prompt.ask("Type")
        node = GraphNode(label=label, type=node_type)
        node_id = await memory_service.add_graph_node(node)
        console.print(f"[green]Node created: {node_id}[/green]")
    elif action == "add-edge":
        source = Prompt.ask("Source ID")
        target = Prompt.ask("Target ID")
        relationship = Prompt.ask("Relationship")
        edge = GraphEdge(source=source, target=target, relationship=relationship)
        edge_id = await memory_service.add_graph_edge(edge)
        console.print(f"[green]Edge created: {edge_id}[/green]")
    elif action == "neighbors":
        node_id = Prompt.ask("Node ID")
        depth = int(Prompt.ask("Depth", default="2"))
        query = GraphQuery(node_id=node_id, depth=depth)
        neighbors = await memory_service.get_graph_neighbors(query)
        for n in neighbors:
            console.print(f"  {n['source']} --[{n['relationship']}]--> {n['target']}")
    elif action == "path":
        source = Prompt.ask("Source")
        target = Prompt.ask("Target")
        path = await memory_service.find_graph_path(source, target)
        if path:
            console.print(" -> ".join(path))
        else:
            console.print("[yellow]No path found[/yellow]")


from datetime import datetime


@app.command()
def health():
    """Check health of all memory stores."""
    asyncio.run(_health_async())


async def _health_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Health Check"))
    await memory_service.initialize()
    health = await memory_service.get_health()

    table = Table(title="Health Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Count", style="yellow")

    table.add_row("Vector Store", "✅ Ready" if health.vector_store_ready else "❌ Not Ready", str(health.long_term_count))
    table.add_row("Graph Store", "✅ Ready" if health.graph_store_ready else "❌ Not Ready", f"{health.graph_nodes} nodes, {health.graph_edges} edges")
    table.add_row("Episodic Store", "✅ Ready" if health.episodic_store_ready else "❌ Not Ready", str(health.episodic_count))
    table.add_row("Short-term", "✅ Active", str(health.short_term_count))

    console.print(table)


@app.command()
def stats():
    """Show memory statistics."""
    asyncio.run(_stats_async())


async def _stats_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Statistics"))
    await memory_service.initialize()
    stats = memory_service.get_stats()
    console.print(stats)


if __name__ == "__main__":
    app()