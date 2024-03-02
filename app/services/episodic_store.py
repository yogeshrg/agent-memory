"""Episodic memory store using SQLite."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any
from datetime import datetime

import sqlite3

from app.core.config import settings
from app.models.schemas import Episode

logger = logging.getLogger(__name__)


class EpisodicStore:
    """Episodic memory store for agent trajectories."""

    def __init__(self) -> None:
        self._db_path = settings.episodic_db_path
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the database."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    summary TEXT,
                    importance REAL,
                    tags TEXT,
                    created_at TEXT,
                    duration_seconds REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance)")

        self._initialized = True
        logger.info("Episodic store initialized")

    def add_episode(self, episode: Episode) -> str:
        """Add an episode to the store."""
        if not self._initialized:
            self.initialize()

        episode_id = str(episode.id)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO episodes (id, title, steps, summary, importance, tags, created_at, duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    episode.title,
                    json.dumps(episode.steps),
                    episode.summary,
                    episode.importance,
                    json.dumps(episode.tags),
                    episode.created_at.isoformat(),
                    episode.duration_seconds,
                ),
            )

        logger.info(f"Added episode: {episode.title} ({episode_id})")
        return episode_id

    def get_episode(self, episode_id: str) -> Episode | None:
        """Get an episode by ID."""
        if not self._initialized:
            self.initialize()

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()

        if not row:
            return None

        return Episode(
            id=uuid.UUID(row["id"]),
            title=row["title"],
            steps=json.loads(row["steps"]),
            summary=row["summary"],
            importance=row["importance"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            duration_seconds=row["duration_seconds"],
        )

    def list_episodes(
        self,
        limit: int = 50,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[Episode]:
        """List episodes with optional filters."""
        if not self._initialized:
            self.initialize()

        query = "SELECT * FROM episodes WHERE importance >= ?"
        params: list[Any] = [min_importance]

        if tags:
            # Filter by tags (simple LIKE for each tag)
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episodes.append(Episode(
                id=uuid.UUID(row["id"]),
                title=row["title"],
                steps=json.loads(row["steps"]),
                summary=row["summary"],
                importance=row["importance"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                created_at=datetime.fromisoformat(row["created_at"]),
                duration_seconds=row["duration_seconds"],
            ))

        return episodes

    def search_episodes(self, query_text: str, limit: int = 10) -> list[Episode]:
        """Search episodes by text (simple LIKE search)."""
        if not self._initialized:
            self.initialize()

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM episodes 
                   WHERE title LIKE ? OR steps LIKE ? OR summary LIKE ?
                   ORDER BY importance DESC, created_at DESC LIMIT ?""",
                (f"%{query_text}%", f"%{query_text}%", f"%{query_text}%", limit),
            )
            rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episodes.append(Episode(
                id=uuid.UUID(row["id"]),
                title=row["title"],
                steps=json.loads(row["steps"]),
                summary=row["summary"],
                importance=row["importance"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                created_at=datetime.fromisoformat(row["created_at"]),
                duration_seconds=row["duration_seconds"],
            ))

        return episodes

    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        if not self._initialized:
            self.initialize()

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            return cursor.rowcount > 0

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if not self._initialized:
            self.initialize()

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM episodes")
            total = cursor.fetchone()[0]

            cursor = conn.execute("SELECT AVG(importance) as avg_importance FROM episodes")
            avg_importance = cursor.fetchone()[0] or 0.0

            cursor = conn.execute("SELECT SUM(duration_seconds) as total_duration FROM episodes")
            total_duration = cursor.fetchone()[0] or 0.0

        return {
            "total_episodes": total,
            "average_importance": avg_importance,
            "total_duration_seconds": total_duration,
        }


# Global instance
episodic_store = EpisodicStore()