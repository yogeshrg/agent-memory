# Agent Memory

A comprehensive **agent memory system** with three complementary memory stores:

- 🧠 **Vector Memory** — Semantic long-term storage using FAISS for similarity search
- 🕸️ **Graph Memory** — Knowledge graph with NetworkX + SQLite for structured relationships
- 📼 **Episodic Memory** — Trajectory storage for agent experiences and learning

## Features

- **Multi-modal Memory** — Vector, graph, and episodic in one unified API
- **Automatic Consolidation** — Short-term → Long-term with importance scoring
- **Forgetting Curve** — Time-based decay for memory management
- **Graph Traversal** — Path finding, neighbor queries, subgraph extraction
- **Episode Replay** — Store and retrieve agent trajectories
- **Production Ready** — FastAPI, Docker, comprehensive tests

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Local Development

```bash
cd agent-memory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Run API server
python -m app.main

# Or use CLI
python -m app.cli interactive
```

### Docker Deployment

```bash
cd docker
docker-compose up -d

# API available at http://localhost:8002/docs
```

## Memory Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent                            │
└────────────────────────┬────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌───────────┐    ┌────────────┐
    │ Vector  │    │  Graph    │    │ Episodic   │
    │ Memory  │    │  Memory   │    │  Memory    │
    │ (FAISS) │    │(NetworkX) │    │  (SQLite)  │
    └────┬────┘    └─────┬─────┘    └─────┬──────┘
         │               │                │
         └───────────────┼────────────────┘
                         ▼
              ┌──────────────────┐
              │ Memory Service   │
              │ (Consolidation,  │
              │  Retrieval,      │
              │  Forgetting)     │
              └──────────────────┘
```

## API Endpoints

### Short-term Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/memory/short-term` | Add to working memory |
| GET | `/api/v1/memory/short-term` | Get recent items |
| DELETE | `/api/v1/memory/short-term` | Clear working memory |

### Long-term Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/memory/long-term` | Add to semantic memory |
| POST | `/api/v1/memory/search` | Search across memory types |
| POST | `/api/v1/memory/consolidate` | Consolidate short→long term |

### Episodic Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/episodes` | Store agent trajectory |
| GET | `/api/v1/episodes` | List episodes |
| GET | `/api/v1/episodes/{id}` | Get episode details |
| GET | `/api/v1/episodes/search/{query}` | Search episodes |

### Graph Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/graph/nodes` | Add knowledge node |
| POST | `/api/v1/graph/edges` | Add relationship |
| POST | `/api/v1/graph/neighbors` | Query neighbors |
| GET | `/api/v1/graph/path/{src}/{tgt}` | Find path |

### Health & Stats
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | System health |
| GET | `/api/v1/stats` | Comprehensive stats |

## Usage Examples

### Store and Retrieve Memory

```python
import httpx

async with httpx.AsyncClient() as client:
    # Add to short-term (working memory)
    await client.post("http://localhost:8002/api/v1/memory/short-term",
        json={"content": "User prefers dark mode", "metadata": {"user_id": "123"}})

    # Add to long-term (semantic memory)
    await client.post("http://localhost:8002/api/v1/memory/long-term",
        json={"content": "Python is a programming language", "importance": 0.9})

    # Search memories
    response = await client.post("http://localhost:8002/api/v1/memory/search",
        json={"query": "programming language", "top_k": 5})
    results = response.json()
```

### Store Agent Episode

```python
episode = {
    "title": "Code Review Session",
    "steps": [
        {"action": "analyze_code", "file": "main.py", "issues": 3},
        {"action": "suggest_fix", "issue": "unused_import", "fix": "remove import"},
        {"action": "approve_pr", "pr_number": 42}
    ],
    "importance": 0.8,
    "tags": ["code-review", "python"]
}

await client.post("http://localhost:8002/api/v1/episodes", json=episode)
```

### Build Knowledge Graph

```python
# Add entities
await client.post("/api/v1/graph/nodes", json={"label": "Python", "type": "language"})
await client.post("/api/v1/graph/nodes", json={"label": "FastAPI", "type": "framework"})

# Add relationships
await client.post("/api/v1/graph/edges", json={
    "source": "python_id",
    "target": "fastapi_id",
    "relationship": "used_by"
})

# Query
neighbors = await client.post("/api/v1/graph/neighbors",
    json={"node_id": "python_id", "depth": 2})
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DIM` | `768` | Embedding dimension |
| `MAX_SHORT_TERM_ITEMS` | `100` | Working memory capacity |
| `CONSOLIDATION_INTERVAL` | `3600` | Auto-consolidation interval (seconds) |
| `IMPORTANCE_THRESHOLD` | `0.5` | Min importance for consolidation |
| `FORGETTING_CURVE_HALF_LIFE` | `86400` | Memory decay half-life (seconds) |

## Project Structure

```
agent-memory/
├── app/
│   ├── api/routes.py          # FastAPI routes
│   ├── cli.py                 # Typer CLI
│   ├── core/config.py         # Pydantic settings
│   ├── main.py                # FastAPI app entry
│   ├── models/schemas.py      # Pydantic models
│   └── services/
│       ├── vector_store.py    # FAISS vector store
│       ├── graph_store.py     # NetworkX + SQLite graph
│       ├── episodic_store.py  # SQLite episodic store
│       └── memory_service.py  # Unified service
├── docker/
│   ├── Dockerfile             # Multi-stage build
│   └── docker-compose.yml     # Full stack
├── tests/                     # Pytest suite
├── pyproject.toml             # Project config
└── .env.example               # Environment template
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

## License

MIT License

## Author

Yogesh Gaikwad — Yogesh.r.gaikwad@gmail.com