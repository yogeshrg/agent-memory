"""Graph memory store using NetworkX and SQLite."""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from datetime import datetime

import networkx as nx

from app.core.config import settings
from app.models.schemas import GraphNode, GraphEdge, GraphQuery

logger = logging.getLogger(__name__)


class GraphStore:
    """Knowledge graph store using NetworkX with SQLite persistence."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._db_path = settings.graph_db_path
        self._initialized = False

    def initialize(self) -> None:
        """Initialize or load the graph from database."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    type TEXT NOT NULL,
                    properties TEXT,
                    importance REAL,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    weight REAL,
                    properties TEXT,
                    created_at TEXT,
                    FOREIGN KEY (source) REFERENCES nodes(id),
                    FOREIGN KEY (target) REFERENCES nodes(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")

        # Load from database
        self._load_from_db()
        self._initialized = True
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _load_from_db(self) -> None:
        """Load graph from SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Load nodes
            cursor = conn.execute("SELECT * FROM nodes")
            for row in cursor:
                node_id = row["id"]
                self.graph.add_node(
                    node_id,
                    label=row["label"],
                    type=row["type"],
                    properties=json.loads(row["properties"]) if row["properties"] else {},
                    importance=row["importance"],
                    created_at=row["created_at"],
                )

            # Load edges
            cursor = conn.execute("SELECT * FROM edges")
            for row in cursor:
                self.graph.add_edge(
                    row["source"],
                    row["target"],
                    key=row["id"],
                    relationship=row["relationship"],
                    weight=row["weight"],
                    properties=json.loads(row["properties"]) if row["properties"] else {},
                    created_at=row["created_at"],
                )

    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph."""
        if not self._initialized:
            self.initialize()

        node_id = node.id or str(uuid.uuid4())
        self.graph.add_node(
            node_id,
            label=node.label,
            type=node.type,
            properties=node.properties,
            importance=node.importance,
            created_at=datetime.utcnow().isoformat(),
        )

        # Persist to database
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, type, properties, importance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    node_id,
                    node.label,
                    node.type,
                    json.dumps(node.properties),
                    node.importance,
                    datetime.utcnow().isoformat(),
                ),
            )

        return node_id

    def add_edge(self, edge: GraphEdge) -> int:
        """Add an edge to the graph."""
        if not self._initialized:
            self.initialize()

        # Ensure nodes exist
        if edge.source not in self.graph:
            self.graph.add_node(edge.source)
        if edge.target not in self.graph:
            self.graph.add_node(edge.target)

        edge_key = self.graph.add_edge(
            edge.source,
            edge.target,
            relationship=edge.relationship,
            weight=edge.weight,
            properties=edge.properties,
            created_at=datetime.utcnow().isoformat(),
        )

        # Persist to database
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO edges (source, target, relationship, weight, properties, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    edge.source,
                    edge.target,
                    edge.relationship,
                    edge.weight,
                    json.dumps(edge.properties),
                    datetime.utcnow().isoformat(),
                ),
            )
            edge_id = cursor.lastrowid

        return edge_id

    def get_neighbors(self, query: GraphQuery) -> list[dict[str, Any]]:
        """Get neighbors of a node up to specified depth."""
        if not self._initialized:
            self.initialize()

        if query.node_id not in self.graph:
            return []

        results = []
        visited = set()

        def dfs(node: str, current_depth: int, path: list[str]):
            if current_depth > query.depth or node in visited:
                return
            visited.add(node)

            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data:
                    for key, data in edge_data.items():
                        if query.relationship and data.get("relationship") != query.relationship:
                            continue
                        if data.get("weight", 0) < query.min_weight:
                            continue

                        results.append({
                            "source": node,
                            "target": neighbor,
                            "relationship": data.get("relationship"),
                            "weight": data.get("weight"),
                            "properties": data.get("properties", {}),
                            "path": path + [neighbor],
                        })
                        dfs(neighbor, current_depth + 1, path + [neighbor])

        dfs(query.node_id, 1, [query.node_id])
        return results

    def find_path(self, source: str, target: str, max_depth: int = 5) -> list[str] | None:
        """Find shortest path between two nodes."""
        if not self._initialized:
            self.initialize()

        if source not in self.graph or target not in self.graph:
            return None

        try:
            path = nx.shortest_path(self.graph, source, target)
            if len(path) - 1 <= max_depth:
                return path
        except nx.NetworkXNoPath:
            pass
        return None

    def get_subgraph(self, node_ids: list[str]) -> dict[str, Any]:
        """Get subgraph induced by node IDs."""
        if not self._initialized:
            self.initialize()

        subgraph = self.graph.subgraph(node_ids)
        return {
            "nodes": [
                {
                    "id": n,
                    "label": subgraph.nodes[n].get("label", ""),
                    "type": subgraph.nodes[n].get("type", ""),
                    "importance": subgraph.nodes[n].get("importance", 0.5),
                }
                for n in subgraph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relationship": data.get("relationship", ""),
                    "weight": data.get("weight", 1.0),
                }
                for u, v, data in subgraph.edges(data=True)
            ],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        if not self._initialized:
            self.initialize()

        node_types: dict[str, int] = {}
        for node in self.graph.nodes(data=True):
            ntype = node[1].get("type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1

        rel_types: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relationship", "unknown")
            rel_types[rel] = rel_types.get(rel, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "relationship_types": rel_types,
        }


# Global instance
graph_store = GraphStore()