"""Embedding backends.

``GeminiEmbedder`` calls the Gemini / Vertex AI embedding endpoint with
task types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY), batching requests.
``HashingEmbedder`` is a deterministic, offline feature-hashing embedder so
the whole pipeline (and CI) runs without credentials.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbedder:
    """Bag of unigrams + bigrams hashed into ``dim`` buckets (signed)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 1) % 2 == 0 else -1.0
        return _normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbedder:
    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768, batch: int = 64):
        from google import genai  # imported lazily so tests don't need credentials

        self._client = genai.Client()
        self._model = model
        self.dim = dim
        self._batch = batch

    def _call(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = self._client.models.embed_content(
                model=self._model,
                contents=list(texts[i : i + self._batch]),
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dim
                ),
            )
            out.extend(_normalize(list(e.values)) for e in resp.embeddings)
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._call(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "RETRIEVAL_QUERY")[0]


def build_embedder(kind: str, model: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "gemini":
        return GeminiEmbedder(model=model)
    raise ValueError(f"unknown embedder: {kind}")
