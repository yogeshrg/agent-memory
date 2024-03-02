"""Pydantic models for Agent Memory."""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


class MemoryType(str, Enum):
    """Types of memory."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryItem(BaseModel):
    """A single memory item."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime | None = None


class MemoryQuery(BaseModel):
    """Query for memory retrieval."""

    query: str
    memory_types: list[MemoryType] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    time_range: tuple[datetime, datetime] | None = None
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    """Search result from memory."""

    item: MemoryItem
    score: float
    memory_type: MemoryType


class Episode(BaseModel):
    """An episodic memory (trajectory)."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0


class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class GraphEdge(BaseModel):
    """An edge in the knowledge graph."""

    source: str
    target: str
    relationship: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphQuery(BaseModel):
    """Query for graph operations."""

    node_id: str | None = None
    relationship: str | None = None
    depth: int = Field(default=2, ge=1, le=5)
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)


class ConsolidationRequest(BaseModel):
    """Request for memory consolidation."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_items: int = Field(default=100, ge=1, le=1000)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    vector_store_ready: bool
    graph_store_ready: bool
    episodic_store_ready: bool
    short_term_count: int
    long_term_count: int
    episodic_count: int
    graph_nodes: int
    graph_edges: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)