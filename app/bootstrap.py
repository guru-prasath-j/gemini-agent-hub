"""Process start-up: load (or build) the retrieval index and pick backends."""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

from app.config import get_settings
from app.rag.embeddings import build_embedder
from app.rag.ingest import ingest_dir
from app.rag.store import HybridIndex
from app.tools import knowledge, workspace

log = logging.getLogger(__name__)
_DONE = False


def build_index(rebuild: bool = False) -> HybridIndex:
    s = get_settings()
    embedder = build_embedder(s.embedder, s.embedding_model)
    if not rebuild and Path(s.index_path).exists():
        try:
            return HybridIndex.load(s.index_path, embedder)
        except ValueError:
            log.warning("index dimension mismatch, rebuilding")
    index = HybridIndex(embedder)
    if Path(s.docs_dir).exists():
        index.add(ingest_dir(s.docs_dir, s.model, multimodal=s.embedder == "gemini"))
        index.save(s.index_path)
    return index


def bootstrap(rebuild: bool = False) -> None:
    global _DONE
    if _DONE and not rebuild:
        return
    knowledge.set_index(build_index(rebuild))
    if os.getenv("WORKSPACE_BACKEND", "demo") == "google":
        workspace.set_backend(workspace.GoogleCalendarBackend(os.getenv("CALENDAR_ID", "primary")))
    else:
        backend = workspace.DemoBackend()
        backend.seed_day(date.today().isoformat())
        workspace.set_backend(backend)
    _DONE = True


if __name__ == "__main__":  # `python -m app.bootstrap` builds the index (e.g. in a Cloud Run Job)
    bootstrap(rebuild=True)
    print(f"indexed {len(knowledge.get_index() or [])} chunks -> {get_settings().index_path}")
