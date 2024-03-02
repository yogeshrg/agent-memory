"""Vector memory store using FAISS."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any
from datetime import datetime

import faiss
import numpy as np

from app.core.config import settings
from app.models.schemas import MemoryItem, MemoryType

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic memory."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: dict[str, dict[str, Any]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize or load the vector index."""
        if self._initialized:
            return

        settings.vector_index_path.parent.mkdir(parents=True, exist_ok=True)

        if settings.vector_index_path.exists() and settings.vector_metadata_path.exists():
            logger.info("Loading existing vector index...")
            self.index = faiss.read_index(str(settings.vector_index_path))
            with open(settings.vector_metadata_path) as f:
                self.metadata = json.load(f)
        else:
            logger.info("Creating new vector index...")
            self.index = faiss.IndexFlatIP(settings.vector_dim)
            self.metadata = {}

        self._initialized = True

    def add(self, item: MemoryItem) -> str:
        """Add a memory item to the vector store."""
        if not self._initialized:
            self.initialize()

        if item.embedding is None:
            # Generate random embedding for testing (in production, use real embeddings)
            embedding = np.random.rand(1, settings.vector_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
        else:
            embedding = np.array(item.embedding, dtype=np.float32)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

        # Add to FAISS index
        item_id = str(item.id)
        self.index.add(embedding)

        # Store metadata
        self.metadata[item_id] = {
            "id": item_id,
            "content": item.content,
            "memory_type": item.memory_type.value,
            "importance": item.importance,
            "metadata": item.metadata,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "access_count": item.access_count,
            "last_accessed": item.last_accessed.isoformat() if item.last_accessed else None,
        }

        self.persist()
        return item_id

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_importance: float = 0.0,
        memory_types: list[MemoryType] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Search for similar vectors."""
        if not self._initialized:
            self.initialize()

        if self.index.ntotal == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Search more than needed to filter
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue

            # Get metadata by index (FAISS doesn't maintain order perfectly, so we need mapping)
            # For simplicity, we'll iterate through metadata
            for item_id, meta in self.metadata.items():
                if meta.get("faiss_index") == idx:
                    if meta.get("importance", 0) < min_importance:
                        continue
                    if memory_types and meta.get("memory_type") not in [mt.value for mt in memory_types]:
                        continue

                    item = MemoryItem(
                        id=uuid.UUID(item_id),
                        content=meta["content"],
                        memory_type=MemoryType(meta["memory_type"]),
                        importance=meta["importance"],
                        metadata=meta["metadata"],
                        created_at=datetime.fromisoformat(meta["created_at"]),
                        updated_at=datetime.fromisoformat(meta["updated_at"]),
                        access_count=meta.get("access_count", 0),
                        last_accessed=datetime.fromisoformat(meta["last_accessed"]) if meta.get("last_accessed") else None,
                    )
                    results.append((item, float(score)))
                    break

            if len(results) >= top_k:
                break

        return results

    def update_importance(self, item_id: str, importance: float) -> bool:
        """Update importance score."""
        if item_id in self.metadata:
            self.metadata[item_id]["importance"] = importance
            self.metadata[item_id]["updated_at"] = datetime.utcnow().isoformat()
            self.persist()
            return True
        return False

    def delete(self, item_id: str) -> bool:
        """Delete an item (mark as deleted, FAISS doesn't support true deletion)."""
        if item_id in self.metadata:
            self.metadata[item_id]["deleted"] = True
            self.persist()
            return True
        return False

    def persist(self) -> None:
        """Persist index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, str(settings.vector_index_path))
        with open(settings.vector_metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if not self._initialized:
            self.initialize()

        total = len(self.metadata)
        deleted = sum(1 for m in self.metadata.values() if m.get("deleted"))
        by_type: dict[str, int] = {}
        for m in self.metadata.values():
            if not m.get("deleted"):
                mt = m.get("memory_type", "unknown")
                by_type[mt] = by_type.get(mt, 0) + 1

        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_items": total - deleted,
            "deleted_items": deleted,
            "by_type": by_type,
        }


# Global instance
vector_store = VectorStore()