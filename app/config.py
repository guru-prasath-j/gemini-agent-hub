"""Runtime configuration, read from environment variables.

Works with either backend the google-genai SDK supports:
  * Gemini API (AI Studio):   GOOGLE_API_KEY=...
  * Vertex AI on Google Cloud: GOOGLE_GENAI_USE_VERTEXAI=TRUE,
                               GOOGLE_CLOUD_PROJECT=..., GOOGLE_CLOUD_LOCATION=...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "gemini-agent-hub"
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "gemini-2.5-flash"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )
    # "gemini" uses the embedding API; "hashing" is an offline, dependency-free
    # fallback used by tests, CI and local demos without credentials.
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "gemini"))
    docs_dir: str = field(default_factory=lambda: os.getenv("DOCS_DIR", "data/sample_docs"))
    index_path: str = field(default_factory=lambda: os.getenv("INDEX_PATH", ".index/index.json"))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    # Session storage: empty -> in-memory; otherwise a SQLAlchemy URL
    # (e.g. Cloud SQL Postgres) handed to ADK's DatabaseSessionService.
    session_db_url: str = field(default_factory=lambda: os.getenv("SESSION_DB_URL", ""))
    redact_pii: bool = field(default_factory=lambda: _bool("REDACT_PII", True))
    block_prompt_injection: bool = field(
        default_factory=lambda: _bool("BLOCK_PROMPT_INJECTION", True)
    )
    use_vertex: bool = field(default_factory=lambda: _bool("GOOGLE_GENAI_USE_VERTEXAI", False))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
