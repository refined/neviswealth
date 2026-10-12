"""Spike: compare small embedding models on the seed corpus.

For each candidate model it reports:
  - retrieval quality on the golden queries (hit@1, recall@3, MRR);
  - score separation between the weakest relevant hit and the strongest
    irrelevant (negative-query) hit, which tells us whether a single cosine
    threshold can filter garbage;
  - single-thread latency (simulating a 1 vCPU box) and process memory.

Usage:
    uv run python scripts/spike_embeddings.py [model ...]
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
CORPUS = json.loads((ROOT / "db" / "seed" / "seed_corpus.json").read_text(encoding="utf-8"))
QUERIES = json.loads((ROOT / "tests" / "eval" / "queries.json").read_text(encoding="utf-8"))
CACHE_DIR = ROOT / "models_cache"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# (model name, query prefix). Some models are tried with and without the prefix.
CANDIDATES: list[tuple[str, str]] = [
    ("BAAI/bge-small-en-v1.5", ""),
    ("BAAI/bge-small-en-v1.5", BGE_QUERY_PREFIX),
    ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ("snowflake/snowflake-arctic-embed-xs", BGE_QUERY_PREFIX),
    ("ibm-granite/granite-embedding-small-english-r2", ""),
    ("minishlab/potion-retrieval-32M", ""),
    ("BAAI/bge-base-en-v1.5", ""),
    ("BAAI/bge-base-en-v1.5", BGE_QUERY_PREFIX),
]


@dataclass
class Result:
    label: str
    hit1: float
    recall3: float
    mrr: float
    min_pos: float
    max_neg: float
    load_s: float
    doc_ms: float
    query_ms: float
    rss_mb: float


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def evaluate(model_name: str, prefix: str) -> Result:
    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss

    t0 = time.perf_counter()
    model = TextEmbedding(model_name, cache_dir=str(CACHE_DIR), threads=1)
    load_s = time.perf_counter() - t0

    docs = CORPUS["documents"]
    keys = [d["key"] for d in docs]
    texts = [f"{d['title']}. {d['content']}" for d in docs]

    t0 = time.perf_counter()
    doc_vecs = _normalize(np.array(list(model.passage_embed(texts))))
    doc_ms = (time.perf_counter() - t0) * 1000 / len(texts)

    hits1, recalls3, rrs, pos_scores = [], [], [], []
    q_times = []
    for item in QUERIES["eval_queries"]:
        t0 = time.perf_counter()
        qv = _normalize(np.array(list(model.query_embed(prefix + item["q"])))[0])
        q_times.append(time.perf_counter() - t0)
        scores = doc_vecs @ qv
        order = [keys[i] for i in np.argsort(-scores)]
        expected = item["expected"]
        hits1.append(1.0 if order[0] in expected else 0.0)
        recalls3.append(len(set(order[:3]) & set(expected)) / len(expected))
        first_rank = min(order.index(e) for e in expected) + 1
        rrs.append(1.0 / first_rank)
        # Best relevant score for this query: what must survive the threshold.
        pos_scores.append(max(float(scores[keys.index(e)]) for e in expected))

    neg_scores = []
    for q in QUERIES["negative_queries"]:
        qv = _normalize(np.array(list(model.query_embed(prefix + q)))[0])
        neg_scores.append(float((doc_vecs @ qv).max()))

    rss_mb = (proc.memory_info().rss - rss_before) / 1e6
    label = model_name + (" +prefix" if prefix else "")
    del model
    return Result(
        label=label,
        hit1=float(np.mean(hits1)),
        recall3=float(np.mean(recalls3)),
        mrr=float(np.mean(rrs)),
        min_pos=min(pos_scores),
        max_neg=max(neg_scores),
        load_s=load_s,
        doc_ms=doc_ms,
        query_ms=float(np.median(q_times)) * 1000,
        rss_mb=rss_mb,
    )


def main() -> None:
    only = set(sys.argv[1:])
    candidates = [c for c in CANDIDATES if not only or c[0] in only]
    header = (
        f"{'model':<58} {'hit@1':>6} {'R@3':>6} {'MRR':>6} {'minPos':>7} {'maxNeg':>7} "
        f"{'gap':>6} {'load s':>7} {'doc ms':>7} {'q ms':>6} {'+RSS MB':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, prefix in candidates:
        try:
            r = evaluate(name, prefix)
        except Exception as exc:
            print(f"{name:<58} FAILED: {exc}")
            continue
        print(
            f"{r.label:<58} {r.hit1:>6.2f} {r.recall3:>6.2f} {r.mrr:>6.2f} {r.min_pos:>7.3f} "
            f"{r.max_neg:>7.3f} {r.min_pos - r.max_neg:>6.3f} {r.load_s:>7.1f} {r.doc_ms:>7.1f} "
            f"{r.query_ms:>6.1f} {r.rss_mb:>8.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
