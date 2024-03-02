"""API routes for Agent Memory."""

from typing import Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends

from app.models.schemas import (
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    Episode,
    GraphNode,
    GraphEdge,
    GraphQuery,
    ConsolidationRequest,
    HealthResponse,
)
from app.services.memory_service import memory_service

router = APIRouter(prefix="/api/v1")


# ========== Short-term Memory ==========

@router.post("/memory/short-term", response_model=MemoryItem)
async def add_short_term(content: str, metadata: dict[str, Any] | None = None) -> MemoryItem:
    """Add item to short-term memory."""
    return memory_service.add_short_term(content, metadata)


@router.get("/memory/short-term", response_model=list[MemoryItem])
async def get_short_term(limit: int = 10) -> list[MemoryItem]:
    """Get recent short-term memories."""
    return memory_service.get_short_term(limit)


@router.delete("/memory/short-term")
async def clear_short_term() -> dict[str, int]:
    """Clear short-term memory."""
    count = memory_service.clear_short_term()
    return {"cleared": count}


# ========== Long-term Memory ==========

@router.post("/memory/long-term", response_model=MemoryItem)
async def add_long_term(
    content: str,
    importance: float = 0.5,
    metadata: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
) -> MemoryItem:
    """Add item to long-term memory."""
    return await memory_service.add_long_term(content, importance, metadata, embedding)


@router.post("/memory/search", response_model=list[MemorySearchResult])
async def search_memory(query: MemoryQuery) -> list[MemorySearchResult]:
    """Search across memory types."""
    return await memory_service.search_memory(query)


@router.post("/memory/consolidate")
async def consolidate_memory(request: ConsolidationRequest) -> dict[str, int]:
    """Consolidate short-term to long-term memory."""
    return await memory_service.consolidate(request)


# ========== Episodic Memory ==========

@router.post("/episodes", response_model=dict[str, str])
async def add_episode(episode: Episode) -> dict[str, str]:
    """Add an episode."""
    episode_id = await memory_service.add_episode(episode)
    return {"episode_id": episode_id}


@router.get("/episodes/{episode_id}", response_model=Episode)
async def get_episode(episode_id: str) -> Episode:
    """Get an episode by ID."""
    episode = await memory_service.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


@router.get("/episodes", response_model=list[Episode])
async def list_episodes(limit: int = 50, min_importance: float = 0.0) -> list[Episode]:
    """List episodes."""
    return await memory_service.list_episodes(limit, min_importance)


@router.get("/episodes/search/{query}", response_model=list[Episode])
async def search_episodes(query: str, limit: int = 10) -> list[Episode]:
    """Search episodes by text."""
    return await memory_service.search_episodes(query, limit)


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str) -> dict[str, bool]:
    """Delete an episode."""
    success = episodic_store.delete_episode(episode_id)
    if not success:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"deleted": True}


# ========== Graph Memory ==========

@router.post("/graph/nodes", response_model=dict[str, str])
async def add_graph_node(node: GraphNode) -> dict[str, str]:
    """Add a node to the knowledge graph."""
    node_id = await memory_service.add_graph_node(node)
    return {"node_id": node_id}


@router.post("/graph/edges", response_model=dict[str, int])
async def add_graph_edge(edge: GraphEdge) -> dict[str, int]:
    """Add an edge to the knowledge graph."""
    edge_id = await memory_service.add_graph_edge(edge)
    return {"edge_id": edge_id}


@router.post("/graph/neighbors", response_model=list[dict[str, Any]])
async def get_graph_neighbors(query: GraphQuery) -> list[dict[str, Any]]:
    """Get neighbors of a node."""
    return await memory_service.get_graph_neighbors(query)


@router.get("/graph/path/{source}/{target}", response_model=list[str] | None)
async def find_graph_path(source: str, target: str, max_depth: int = 5) -> list[str] | None:
    """Find path between two nodes."""
    return await memory_service.find_graph_path(source, target, max_depth)


# ========== Health ==========

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check for all memory stores."""
    return await memory_service.get_health()


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Get comprehensive statistics."""
    return memory_service.get_stats()