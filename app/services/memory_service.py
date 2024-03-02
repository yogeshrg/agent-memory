"""Main memory service coordinating vector, graph, and episodic stores."""

import logging
import uuid
from typing import Any
from datetime import datetime, timedelta

import numpy as np

from app.core.config import settings
from app.models.schemas import (
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
    Episode,
    GraphNode,
    GraphEdge,
    GraphQuery,
    ConsolidationRequest,
    HealthResponse,
)
from app.services.vector_store import vector_store
from app.services.graph_store import graph_store
from app.services.episodic_store import episodic_store

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified memory service for agents."""

    def __init__(self) -> None:
        self._initialized = False
        self._short_term: list[MemoryItem] = []

    async def initialize(self) -> None:
        """Initialize all memory stores."""
        if self._initialized:
            return

        logger.info("Initializing memory service...")

        # Initialize stores
        vector_store.initialize()
        graph_store.initialize()
        episodic_store.initialize()

        self._initialized = True
        logger.info("Memory service initialized")

    # ========== Short-term Memory (Working Memory) ==========

    def add_short_term(self, content: str, metadata: dict[str, Any] | None = None, importance: float = 0.5) -> MemoryItem:
        """Add item to short-term memory."""
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            importance=importance,
            metadata=metadata or {},
        )
        self._short_term.append(item)

        # Trim if exceeds max
        if len(self._short_term) > settings.max_short_term_items:
            # Remove oldest, least important
            self._short_term.sort(key=lambda x: (x.importance, x.created_at))
            self._short_term = self._short_term[-settings.max_short_term_items:]

        return item

    def get_short_term(self, limit: int = 10) -> list[MemoryItem]:
        """Get recent short-term memories."""
        return self._short_term[-limit:]

    def clear_short_term(self) -> int:
        """Clear short-term memory."""
        count = len(self._short_term)
        self._short_term.clear()
        return count

    # ========== Long-term Memory (Vector Store) ==========

    async def add_long_term(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryItem:
        """Add item to long-term memory (vector store)."""
        if not self._initialized:
            await self.initialize()

        item = MemoryItem(
            content=content,
            memory_type=MemoryType.LONG_TERM,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {},
        )

        vector_store.add(item)
        return item

    async def search_memory(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Search across all memory types."""
        if not self._initialized:
            await self.initialize()

        results = []

        # Generate query embedding (placeholder - would use real embedding model)
        query_embedding = np.random.rand(settings.embed_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search vector store
        memory_types = query.memory_types or [MemoryType.LONG_TERM, MemoryType.SEMANTIC]
        vector_results = vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=query.top_k,
            min_importance=query.min_importance,
            memory_types=memory_types,
        )

        for item, score in vector_results:
            results.append(MemorySearchResult(
                item=item,
                score=score,
                memory_type=item.memory_type,
            ))

        # Search short-term memory
        if MemoryType.SHORT_TERM in memory_types or not query.memory_types:
            for item in self._short_term:
                if item.importance >= query.min_importance:
                    # Simple text matching for short-term
                    if query.query.lower() in item.content.lower():
                        results.append(MemorySearchResult(
                            item=item,
                            score=0.8,  # Placeholder score
                            memory_type=MemoryType.SHORT_TERM,
                        ))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.top_k]

    async def consolidate(self, request: ConsolidationRequest) -> dict[str, int]:
        """Consolidate short-term memories into long-term storage."""
        if not self._initialized:
            await self.initialize()

        consolidated = 0
        for item in self._short_term[:]:
            if item.importance >= request.threshold:
                await self.add_long_term(
                    content=item.content,
                    importance=item.importance,
                    metadata=item.metadata,
                    embedding=item.embedding,
                )
                self._short_term.remove(item)
                consolidated += 1

                if consolidated >= request.max_items:
                    break

        logger.info(f"Consolidated {consolidated} memories")
        return {"consolidated": consolidated, "remaining_short_term": len(self._short_term)}

    # ========== Episodic Memory ==========

    async def add_episode(self, episode: Episode) -> str:
        """Add an episode to episodic memory."""
        if not self._initialized:
            await self.initialize()

        return episodic_store.add_episode(episode)

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Get an episode by ID."""
        if not self._initialized:
            await self.initialize()

        return episodic_store.get_episode(episode_id)

    async def list_episodes(self, limit: int = 50, min_importance: float = 0.0) -> list[Episode]:
        """List episodes."""
        if not self._initialized:
            await self.initialize()

        return episodic_store.list_episodes(limit=limit, min_importance=min_importance)

    async def search_episodes(self, query: str, limit: int = 10) -> list[Episode]:
        """Search episodes by text."""
        if not self._initialized:
            await self.initialize()

        return episodic_store.search_episodes(query, limit=limit)

    # ========== Graph Memory ==========

    async def add_graph_node(self, node: GraphNode) -> str:
        """Add a node to the knowledge graph."""
        if not self._initialized:
            await self.initialize()

        return graph_store.add_node(node)

    async def add_graph_edge(self, edge: GraphEdge) -> int:
        """Add an edge to the knowledge graph."""
        if not self._initialized:
            await self.initialize()

        return graph_store.add_edge(edge)

    async def get_graph_neighbors(self, query: GraphQuery) -> list[dict[str, Any]]:
        """Get graph neighbors."""
        if not self._initialized:
            await self.initialize()

        return graph_store.get_neighbors(query)

    async def find_graph_path(self, source: str, target: str, max_depth: int = 5) -> list[str] | None:
        """Find path in graph."""
        if not self._initialized:
            await self.initialize()

        return graph_store.find_path(source, target, max_depth)

    # ========== Health & Stats ==========

    async def get_health(self) -> HealthResponse:
        """Get health status of all memory stores."""
        if not self._initialized:
            await self.initialize()

        vector_stats = vector_store.get_stats()
        graph_stats = graph_store.get_stats()
        episodic_stats = episodic_store.get_stats()

        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            vector_store_ready=True,
            graph_store_ready=True,
            episodic_store_ready=True,
            short_term_count=len(self._short_term),
            long_term_count=vector_stats.get("total_items", 0),
            episodic_count=episodic_stats.get("total_episodes", 0),
            graph_nodes=graph_stats.get("nodes", 0),
            graph_edges=graph_stats.get("edges", 0),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "short_term": len(self._short_term),
            "vector": vector_store.get_stats(),
            "graph": graph_store.get_stats(),
            "episodic": episodic_store.get_stats(),
        }


# Global instance
memory_service = MemoryService()