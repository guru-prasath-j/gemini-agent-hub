import os

os.environ.setdefault("EMBEDDER", "hashing")
os.environ.setdefault("INDEX_PATH", ".pytest_index/index.json")

import pytest  # noqa: E402

from app.rag.embeddings import HashingEmbedder  # noqa: E402
from app.rag.ingest import ingest_dir  # noqa: E402
from app.rag.store import HybridIndex  # noqa: E402
from app.tools import knowledge, workspace  # noqa: E402


@pytest.fixture()
def index() -> HybridIndex:
    idx = HybridIndex(HashingEmbedder())
    idx.add(ingest_dir("data/sample_docs", model="unused", multimodal=False))
    knowledge.set_index(idx)
    return idx


@pytest.fixture()
def demo_backend():
    b = workspace.DemoBackend()
    b.seed_day("2026-10-26")
    workspace.set_backend(b)
    return b
