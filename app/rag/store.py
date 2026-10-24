"""Hybrid vector + keyword index.

Dense retrieval (cosine over normalized embeddings) and sparse retrieval
(BM25) are fused with Reciprocal Rank Fusion, then diversified with
Maximal Marginal Relevance. The index serializes to JSON so it can be built
at deploy time (or in a Cloud Run Job) and loaded read-only by the service.

The interface mirrors what you'd put in front of Vertex AI Vector Search or
AlloyDB/pgvector, so swapping the backend doesn't touch the agents.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from app.rag.chunker import Chunk
from app.rag.embeddings import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from how i in is it of on or our the this to we what when "
    "where which who why with you your can do does".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    def citation(self) -> str:
        return f"{self.chunk.source}" + (f" › {self.chunk.section}" if self.chunk.section else "")


class HybridIndex:
    def __init__(self, embedder: Embedder, k1: float = 1.5, b: float = 0.75) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._k1, self._b = k1, b

    # ---------- building ----------
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        for c, v in zip(chunks, vectors):
            self._append(c, v)

    def _append(self, chunk: Chunk, vector: list[float]) -> None:
        tf = Counter(_tokens(f"{chunk.section} {chunk.text}"))
        self.chunks.append(chunk)
        self.vectors.append(vector)
        self._tf.append(tf)
        self._df.update(tf.keys())

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------- scoring ----------
    def _bm25(self, query: str) -> list[float]:
        n = len(self.chunks)
        avgdl = sum(sum(tf.values()) for tf in self._tf) / max(n, 1)
        q = _tokens(query)
        scores = []
        for tf in self._tf:
            dl = sum(tf.values())
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                f = tf[term]
                s += idf * f * (self._k1 + 1) / (f + self._k1 * (1 - self._b + self._b * dl / avgdl))
            scores.append(s)
        return scores

    def search(
        self, query: str, k: int = 4, rrf_k: int = 60, mmr_lambda: float = 0.7, pool: int = 20
    ) -> list[Hit]:
        if not self.chunks:
            return []
        qv = self.embedder.embed_query(query)
        dense = [_dot(qv, v) for v in self.vectors]
        sparse = self._bm25(query)

        dense_order = sorted(range(len(dense)), key=lambda i: -dense[i])[:pool]
        sparse_order = [i for i in sorted(range(len(sparse)), key=lambda i: -sparse[i]) if sparse[i] > 0][:pool]
        fused: dict[int, float] = {}
        d_rank = {i: r for r, i in enumerate(dense_order)}
        s_rank = {i: r for r, i in enumerate(sparse_order)}
        for i in set(dense_order) | set(sparse_order):
            fused[i] = sum(1.0 / (rrf_k + rank[i] + 1) for rank in (d_rank, s_rank) if i in rank)

        candidates = sorted(fused, key=lambda i: -fused[i])
        selected: list[int] = []
        # MMR: trade relevance against redundancy with what's already picked.
        top = max(fused.values())
        while candidates and len(selected) < k:
            def mmr(i: int) -> float:
                redundancy = max((_dot(self.vectors[i], self.vectors[j]) for j in selected), default=0.0)
                return mmr_lambda * (fused[i] / top) - (1 - mmr_lambda) * redundancy

            best = max(candidates, key=mmr)
            selected.append(best)
            candidates.remove(best)
        return [
            Hit(self.chunks[i], round(fused[i], 5), d_rank.get(i), s_rank.get(i)) for i in selected
        ]

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.embedder.dim,
            "items": [{"chunk": asdict(c), "vector": v} for c, v in zip(self.chunks, self.vectors)],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> HybridIndex:
        payload = json.loads(Path(path).read_text())
        if payload["dim"] != embedder.dim:
            raise ValueError("index was built with a different embedding dimension")
        idx = cls(embedder)
        for item in payload["items"]:
            idx._append(Chunk(**item["chunk"]), item["vector"])
        return idx
