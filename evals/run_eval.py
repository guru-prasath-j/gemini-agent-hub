"""Evaluation harness, used as a CI quality gate.

Retrieval (always, offline-capable):  recall@k and MRR of the gold section.
Answers   (--live, needs Gemini):     the agent's reply must cite a source and
                                      contain the gold facts; reports latency.

    python -m evals.run_eval                    # retrieval only, hashing embedder
    EMBEDDER=gemini python -m evals.run_eval --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from app.bootstrap import bootstrap
from app.tools import knowledge

GOLDEN = Path(__file__).with_name("golden.jsonl")


def load() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def retrieval(cases: list[dict], k: int) -> dict:
    index = knowledge.get_index()
    hits, rr = 0, []
    for c in cases:
        sections = [h.chunk.section for h in index.search(c["q"], k=k)]
        if c["section"] in sections:
            hits += 1
            rr.append(1 / (sections.index(c["section"]) + 1))
        else:
            rr.append(0.0)
            print(f"  miss: {c['q']!r} -> {sections}")
    return {f"recall@{k}": round(hits / len(cases), 3), "mrr": round(statistics.mean(rr), 3)}


async def answers(cases: list[dict]) -> dict:
    from app.server import AgentService, ChatRequest

    svc = AgentService()
    passed, lat = 0, []
    for c in cases:
        t0 = time.perf_counter()
        r = await svc.chat(ChatRequest(message=c["q"]))
        lat.append((time.perf_counter() - t0) * 1000)
        ok = "[1]" in r.reply and all(m.lower() in r.reply.lower() for m in c["must_contain"])
        passed += ok
        if not ok:
            print(f"  fail: {c['q']!r} -> {r.reply[:120]!r}")
    lat.sort()
    return {
        "answer_pass_rate": round(passed / len(cases), 3),
        "p50_ms": int(lat[len(lat) // 2]),
        "p95_ms": int(lat[min(len(lat) - 1, int(len(lat) * 0.95))]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--min-recall", type=float, default=0.9)
    args = ap.parse_args()

    bootstrap(rebuild=True)
    cases = load()
    report = retrieval(cases, args.k)
    if args.live:
        report |= asyncio.run(answers(cases))
    print(json.dumps(report, indent=2))
    return 0 if report[f"recall@{args.k}"] >= args.min_recall else 1


if __name__ == "__main__":
    sys.exit(main())
