"""Knowledge-base tool exposed to the agents.

Retrieved chunks are screened for *indirect* prompt injection (instructions
planted inside documents) before they are handed to the model, and each
result carries a citation so answers can be grounded and verified.
"""

from __future__ import annotations

from typing import Any

from app.governance.audit import audit
from app.governance.injection import screen
from app.rag.store import HybridIndex

_INDEX: HybridIndex | None = None


def set_index(index: HybridIndex) -> None:
    global _INDEX
    _INDEX = index


def get_index() -> HybridIndex | None:
    return _INDEX


def search_knowledge_base(query: str, top_k: int = 4) -> dict[str, Any]:
    """Search the company knowledge base (policies, runbooks, product docs).

    Args:
        query: A focused natural-language search query.
        top_k: How many passages to return (1-8).

    Returns:
        A dict with ``status`` and ``results``; each result has ``citation``,
        ``text`` and ``score``. Cite results as [n] in the answer.
    """
    if _INDEX is None or len(_INDEX) == 0:
        return {"status": "empty", "results": [], "note": "The knowledge base has no documents yet."}
    top_k = max(1, min(int(top_k), 8))
    results, quarantined = [], 0
    for hit in _INDEX.search(query, k=top_k):
        if screen(hit.chunk.text).blocked:
            quarantined += 1
            continue
        results.append(
            {"n": len(results) + 1, "citation": hit.citation(), "text": hit.chunk.text, "score": hit.score}
        )
    if quarantined:
        audit("retrieval_quarantined", severity="WARNING", query=query, count=quarantined)
    return {"status": "ok", "results": results, "quarantined": quarantined}
