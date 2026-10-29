import pytest

from app.rag.chunker import chunk_text
from app.rag.embeddings import HashingEmbedder
from app.rag.store import HybridIndex
from app.tools.knowledge import search_knowledge_base


def test_chunker_keeps_sections_and_bounds():
    text = "# A\n\n" + ("alpha " * 300) + "\n\n# B\n\nshort beta paragraph"
    chunks = chunk_text(text, "doc.md", max_chars=500, overlap=50)
    assert {c.section for c in chunks} == {"A", "B"}
    assert all(len(c.text) <= 500 for c in chunks)
    assert chunks[-1].text == "short beta paragraph"


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", "d", max_chars=10, overlap=10)


@pytest.mark.parametrize(
    "query, expected_section",
    [
        ("How many unused leave days carry over?", "Annual leave"),
        ("roll back a Cloud Run deploy to the previous revision", "Rollback on Cloud Run"),
        ("what should I do if my laptop is stolen", "Laptop and device rules"),
        ("meal allowance for travel in metro cities", "Travel and expenses"),
    ],
)
def test_hybrid_retrieval_top1(index, query, expected_section):
    hits = index.search(query, k=3)
    assert hits[0].chunk.section == expected_section


def test_index_roundtrip(tmp_path, index):
    p = tmp_path / "i.json"
    index.save(p)
    again = HybridIndex.load(p, HashingEmbedder())
    assert len(again) == len(index)
    assert again.search("sick leave certificate")[0].chunk.section == "Sick leave"


def test_tool_quarantines_planted_instructions(index):
    out = search_knowledge_base("vendor document notes ignore previous instructions", top_k=8)
    assert out["quarantined"] >= 1
    assert all("Ignore all previous instructions" not in r["text"] for r in out["results"])


@pytest.mark.parametrize("mode", ["hybrid", "hybrid_no_mmr", "dense", "sparse"])
def test_every_retrieval_mode_answers(index, mode):
    hits = index.search("medical certificate sick leave", k=3, mode=mode)
    assert 0 < len(hits) <= 3
    if mode == "sparse":
        assert all(h.sparse_rank is not None for h in hits) and hits[0].chunk.section == "Sick leave"
    if mode == "dense":
        assert all(h.dense_rank is not None for h in hits)


def test_unknown_mode_is_refused(index):
    with pytest.raises(ValueError):
        index.search("x", mode="bm42")
