"""Tests for Agent Memory."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
from uuid import uuid4

from app.core.config import settings
from app.models.schemas import (
    MemoryItem,
    MemoryQuery,
    MemoryType,
    Episode,
    GraphNode,
    GraphEdge,
    GraphQuery,
    HealthResponse,
)
from app.services.vector_store import VectorStore
from app.services.graph_store import GraphStore
from app.services.episodic_store import EpisodicStore
from app.services.memory_service import MemoryService


class TestConfig:
    """Test configuration loading."""

    def test_settings_defaults(self):
        """Test default settings values."""
        assert settings.app_name == "Agent Memory"
        assert settings.vector_dim == 768
        assert settings.embed_dim == 768


class TestSchemas:
    """Test Pydantic schemas."""

    def test_memory_item(self):
        """Test memory item model."""
        item = MemoryItem(content="Test memory", importance=0.8)
        assert item.content == "Test memory"
        assert item.importance == 0.8
        assert item.memory_type == MemoryType.SHORT_TERM

    def test_episode(self):
        """Test episode model."""
        episode = Episode(title="Test Episode", steps=[{"action": "test"}])
        assert episode.title == "Test Episode"
        assert len(episode.steps) == 1

    def test_graph_node(self):
        """Test graph node model."""
        node = GraphNode(label="Test Node", type="entity")
        assert node.label == "Test Node"
        assert node.type == "entity"

    def test_graph_edge(self):
        """Test graph edge model."""
        edge = GraphEdge(source="A", target="B", relationship="related_to")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.relationship == "related_to"

    def test_health_response(self):
        """Test health response model."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            vector_store_ready=True,
            graph_store_ready=True,
            episodic_store_ready=True,
            short_term_count=5,
            long_term_count=100,
            episodic_count=10,
            graph_nodes=50,
            graph_edges=100,
        )
        assert health.status == "healthy"
        assert health.long_term_count == 100


class TestVectorStore:
    """Test vector store."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create vector store with temp paths."""
        with patch.object(settings, "vector_index_path", tmp_path / "vector_index.faiss"):
            with patch.object(settings, "vector_metadata_path", tmp_path / "vector_metadata.json"):
                with patch.object(settings, "vector_dim", 128):
                    store = VectorStore()
                    yield store

    def test_initialize_creates_index(self, vector_store, tmp_path):
        """Test initialization creates FAISS index."""
        vector_store.initialize()
        assert vector_store._initialized
        assert vector_store.index is not None

    def test_add_and_search(self, vector_store, tmp_path):
        """Test adding and searching vectors."""
        vector_store.initialize()

        item = MemoryItem(
            content="Test memory content",
            memory_type=MemoryType.LONG_TERM,
            importance=0.7,
        )

        item_id = vector_store.add(item)
        assert item_id == str(item.id)

        # Search
        query_embedding = [0.1] * 128
        results = vector_store.search(query_embedding, top_k=5)
        # Note: With random embeddings, may not find exact match
        assert isinstance(results, list)

    def test_update_importance(self, vector_store, tmp_path):
        """Test updating importance."""
        vector_store.initialize()

        item = MemoryItem(content="Test", importance=0.5)
        item_id = vector_store.add(item)

        result = vector_store.update_importance(item_id, 0.9)
        assert result is True
        assert vector_store.metadata[item_id]["importance"] == 0.9

    def test_get_stats(self, vector_store, tmp_path):
        """Test getting statistics."""
        vector_store.initialize()
        stats = vector_store.get_stats()
        assert "total_vectors" in stats
        assert "total_items" in stats


class TestGraphStore:
    """Test graph store."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        """Create graph store with temp path."""
        with patch.object(settings, "graph_db_path", tmp_path / "graph.db"):
            store = GraphStore()
            yield store

    def test_initialize(self, graph_store, tmp_path):
        """Test initialization."""
        graph_store.initialize()
        assert graph_store._initialized

    def test_add_node(self, graph_store, tmp_path):
        """Test adding node."""
        graph_store.initialize()

        node = GraphNode(label="Test Entity", type="entity", importance=0.8)
        node_id = graph_store.add_node(node)

        assert node_id == node.id
        assert graph_store.graph.has_node(node_id)
        assert graph_store.graph.nodes[node_id]["label"] == "Test Entity"

    def test_add_edge(self, graph_store, tmp_path):
        """Test adding edge."""
        graph_store.initialize()

        # Add nodes first
        node1 = GraphNode(label="A", type="entity")
        node2 = GraphNode(label="B", type="entity")
        graph_store.add_node(node1)
        graph_store.add_node(node2)

        # Add edge
        edge = GraphEdge(source=node1.id, target=node2.id, relationship="related_to")
        edge_id = graph_store.add_edge(edge)

        assert edge_id > 0
        assert graph_store.graph.has_edge(node1.id, node2.id)

    def test_get_neighbors(self, graph_store, tmp_path):
        """Test getting neighbors."""
        graph_store.initialize()

        node1 = GraphNode(label="A", type="entity")
        node2 = GraphNode(label="B", type="entity")
        node3 = GraphNode(label="C", type="entity")
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        graph_store.add_node(node3)

        graph_store.add_edge(GraphEdge(source=node1.id, target=node2.id, relationship="knows"))
        graph_store.add_edge(GraphEdge(source=node2.id, target=node3.id, relationship="knows"))

        query = GraphQuery(node_id=node1.id, depth=2)
        neighbors = graph_store.get_neighbors(query)

        assert len(neighbors) >= 1
        assert neighbors[0]["target"] == node2.id

    def test_find_path(self, graph_store, tmp_path):
        """Test finding path."""
        graph_store.initialize()

        node1 = GraphNode(label="A", type="entity")
        node2 = GraphNode(label="B", type="entity")
        node3 = GraphNode(label="C", type="entity")
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        graph_store.add_node(node3)

        graph_store.add_edge(GraphEdge(source=node1.id, target=node2.id, relationship="knows"))
        graph_store.add_edge(GraphEdge(source=node2.id, target=node3.id, relationship="knows"))

        path = graph_store.find_path(node1.id, node3.id)
        assert path is not None
        assert node1.id in path
        assert node3.id in path


class TestEpisodicStore:
    """Test episodic store."""

    @pytest.fixture
    def episodic_store(self, tmp_path):
        """Create episodic store with temp path."""
        with patch.object(settings, "episodic_db_path", tmp_path / "episodic.db"):
            store = EpisodicStore()
            yield store

    def test_initialize(self, episodic_store, tmp_path):
        """Test initialization."""
        episodic_store.initialize()
        assert episodic_store._initialized

    def test_add_and_get_episode(self, episodic_store, tmp_path):
        """Test adding and getting episode."""
        episodic_store.initialize()

        episode = Episode(
            title="Test Episode",
            steps=[{"action": "step1"}, {"action": "step2"}],
            importance=0.7,
            tags=["test"],
        )

        episode_id = episodic_store.add_episode(episode)
        assert episode_id == str(episode.id)

        retrieved = episodic_store.get_episode(episode_id)
        assert retrieved is not None
        assert retrieved.title == "Test Episode"
        assert len(retrieved.steps) == 2
        assert retrieved.importance == 0.7
        assert "test" in retrieved.tags

    def test_list_episodes(self, episodic_store, tmp_path):
        """Test listing episodes."""
        episodic_store.initialize()

        ep1 = Episode(title="Episode 1", importance=0.5)
        ep2 = Episode(title="Episode 2", importance=0.8)
        episodic_store.add_episode(ep1)
        episodic_store.add_episode(ep2)

        episodes = episodic_store.list_episodes(limit=10)
        assert len(episodes) == 2
        # Should be ordered by created_at desc
        assert episodes[0].title == "Episode 2"

    def test_search_episodes(self, episodic_store, tmp_path):
        """Test searching episodes."""
        episodic_store.initialize()

        ep = Episode(title="Important Meeting", steps=[{"action": "discuss budget"}])
        episodic_store.add_episode(ep)

        results = episodic_store.search_episodes("Meeting")
        assert len(results) == 1
        assert results[0].title == "Important Meeting"

    def test_get_stats(self, episodic_store, tmp_path):
        """Test getting statistics."""
        episodic_store.initialize()

        ep1 = Episode(title="Ep1", importance=0.5, duration_seconds=100)
        ep2 = Episode(title="Ep2", importance=0.8, duration_seconds=200)
        episodic_store.add_episode(ep1)
        episodic_store.add_episode(ep2)

        stats = episodic_store.get_stats()
        assert stats["total_episodes"] == 2
        assert stats["average_importance"] == 0.65
        assert stats["total_duration_seconds"] == 300


class TestMemoryService:
    """Test unified memory service."""

    @pytest.fixture
    def memory_service(self, tmp_path):
        """Create memory service with temp paths."""
        with patch.object(settings, "vector_index_path", tmp_path / "vector_index.faiss"):
            with patch.object(settings, "vector_metadata_path", tmp_path / "vector_metadata.json"):
                with patch.object(settings, "graph_db_path", tmp_path / "graph.db"):
                    with patch.object(settings, "episodic_db_path", tmp_path / "episodic.db"):
                        with patch.object(settings, "vector_dim", 128):
                            with patch.object(settings, "embed_dim", 128):
                                with patch.object(settings, "max_short_term_items", 10):
                                    service = MemoryService()
                                    yield service

    @pytest.mark.asyncio
    async def test_initialize(self, memory_service, tmp_path):
        """Test initialization."""
        await memory_service.initialize()
        assert memory_service._initialized

    @pytest.mark.asyncio
    async def test_short_term_memory(self, memory_service, tmp_path):
        """Test short-term memory operations."""
        await memory_service.initialize()

        # Add items
        item1 = memory_service.add_short_term("Memory 1")
        item2 = memory_service.add_short_term("Memory 2", {"key": "value"})

        assert len(memory_service._short_term) == 2
        assert item1.content == "Memory 1"
        assert item2.metadata == {"key": "value"}

        # Get short term
        recent = memory_service.get_short_term(limit=1)
        assert len(recent) == 1
        assert recent[0].content == "Memory 2"

        # Clear
        cleared = memory_service.clear_short_term()
        assert cleared == 2
        assert len(memory_service._short_term) == 0

    @pytest.mark.asyncio
    async def test_long_term_memory(self, memory_service, tmp_path):
        """Test long-term memory operations."""
        await memory_service.initialize()

        item = await memory_service.add_long_term(
            content="Long term memory",
            importance=0.8,
            metadata={"source": "test"},
        )

        assert item.content == "Long term memory"
        assert item.importance == 0.8
        assert item.memory_type == MemoryType.LONG_TERM

    @pytest.mark.asyncio
    async def test_consolidation(self, memory_service, tmp_path):
        """Test memory consolidation."""
        await memory_service.initialize()

        # Add short-term items with varying importance
        memory_service.add_short_term("Low importance", importance=0.3)
        memory_service.add_short_term("High importance", importance=0.9)
        memory_service.add_short_term("Medium importance", importance=0.5)

        assert len(memory_service._short_term) == 3

        # Consolidate with threshold 0.5
        result = await memory_service.consolidate(type("Request", (), {"threshold": 0.5, "max_items": 10})())

        assert result["consolidated"] == 2  # High and Medium
        assert result["remaining_short_term"] == 1  # Low remains

    @pytest.mark.asyncio
    async def test_episodic_memory(self, memory_service, tmp_path):
        """Test episodic memory operations."""
        await memory_service.initialize()

        episode = Episode(
            title="Test Episode",
            steps=[{"action": "test"}],
            importance=0.7,
        )

        episode_id = await memory_service.add_episode(episode)
        assert episode_id == str(episode.id)

        retrieved = await memory_service.get_episode(episode_id)
        assert retrieved is not None
        assert retrieved.title == "Test Episode"

        episodes = await memory_service.list_episodes(limit=10)
        assert len(episodes) >= 1

    @pytest.mark.asyncio
    async def test_graph_memory(self, memory_service, tmp_path):
        """Test graph memory operations."""
        await memory_service.initialize()

        node = GraphNode(label="Test Entity", type="entity")
        node_id = await memory_service.add_graph_node(node)
        assert node_id == node.id

        node2 = GraphNode(label="Another Entity", type="entity")
        node2_id = await memory_service.add_graph_node(node2)

        edge = GraphEdge(source=node_id, target=node2_id, relationship="related")
        edge_id = await memory_service.add_graph_edge(edge)
        assert edge_id > 0

        query = GraphQuery(node_id=node_id, depth=1)
        neighbors = await memory_service.get_graph_neighbors(query)
        assert len(neighbors) >= 1

    @pytest.mark.asyncio
    async def test_health_check(self, memory_service, tmp_path):
        """Test health check."""
        await memory_service.initialize()
        health = await memory_service.get_health()

        assert health.status == "healthy"
        assert health.vector_store_ready is True
        assert health.graph_store_ready is True
        assert health.episodic_store_ready is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])