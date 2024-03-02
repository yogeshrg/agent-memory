"""Agent Memory configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Agent Memory"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8002

    # Vector Store (FAISS)
    vector_dim: int = 768
    vector_index_path: Path = Field(default=Path("./data/vector_index.faiss"))
    vector_metadata_path: Path = Field(default=Path("./data/vector_metadata.json"))

    # Graph Store (NetworkX + SQLite)
    graph_db_path: Path = Field(default=Path("./data/graph.db"))

    # Episodic Store (SQLite)
    episodic_db_path: Path = Field(default=Path("./data/episodic.db"))

    # Memory Settings
    consolidation_interval: int = 3600  # seconds
    importance_threshold: float = 0.5
    max_short_term_items: int = 100
    forgetting_curve_half_life: int = 86400  # 24 hours in seconds

    # Embedding (would use Ollama in production)
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768


settings = Settings()