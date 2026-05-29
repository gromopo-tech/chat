#!/usr/bin/env python
"""
Recall@k eval harness — in-process, sweep over k values.

Computes true per-query recall@k against objective relevant sets derived from
payload data already in Qdrant — no hand-labeling required:

  - complaint queries  (class=complaint): relevant = reviews with rating ≤ relevant_rating_max
  - topic queries      (class=topic):     relevant = reviews whose text contains any expected_keyword

recall@k = |relevant ∩ retrieved_top_k| / |relevant|, averaged per query.

Queries whose relevant set is empty are reported as N/A and excluded from the
mean — they indicate a fixture gap against the current corpus, not a retrieval
failure.

Usage (from repo root, venv active):

    # Default k sweep against the configured business_id in queries.jsonl
    python eval/run_eval.py

    # Custom k values
    python eval/run_eval.py --k-values 5,10,50,100,250,500

    # Override business_id for all queries
    python eval/run_eval.py --business-id my-tenant

    # Per-query detail table at a specific k (for spot-checking)
    python eval/run_eval.py --detail-k 50

Prereqs: Qdrant running, reviews ingested, GCP ADC available (Vertex embeddings).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before any app.* imports so Config reads VERTEX_PROJECT correctly.
# Same guard as run_ingest.py: don't let .env overwrite a real QDRANT_HOST that
# points at Docker's internal hostname when running scripts locally.
import os
from dotenv import load_dotenv
_qdrant_host_pre = os.environ.get("QDRANT_HOST")
load_dotenv(Path(__file__).parent.parent / ".env")
if _qdrant_host_pre is None:
    os.environ.pop("QDRANT_HOST", None)

from app.query_parser import parse_query_with_llm
from app.vectorstore import build_qdrant_filter, create_dense_retriever, get_qdrant

QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
DEFAULT_K_VALUES = [5, 10, 25, 50, 100, 250, 500]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def build_corpus(business_id: str) -> list[dict]:
    """Fetch every review for a tenant from Qdrant.

    Returns list of {external_id, text, rating} dicts. Paginates until all
    points are retrieved — safe for corpora of any size.
    """
    client = get_qdrant()
    corpus: list[dict] = []
    offset = None

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    tenant_filter = Filter(must=[
        FieldCondition(key="business_id", match=MatchValue(value=business_id))
    ])

    while True:
        points, next_offset = client.scroll(
            collection_name="reviews",
            scroll_filter=tenant_filter,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            corpus.append({
                "external_id": payload.get("external_id", str(p.id)),
                "text": payload.get("text", ""),
                "rating": payload.get("rating"),
            })
        if next_offset is None:
            break
        offset = next_offset

    return corpus


# ---------------------------------------------------------------------------
# Relevance labeling
# ---------------------------------------------------------------------------

def relevant_ids(query_row: dict, corpus: list[dict]) -> set[str] | None:
    """Return the set of external_ids that are relevant for this query.

    Returns None if the query class is unrecognised.
    Returns an empty set if the labeling mode matched nothing (reported as N/A).
    """
    cls = query_row.get("class", "topic")

    if cls == "complaint":
        max_rating = query_row.get("relevant_rating_max", 2)
        return {
            r["external_id"]
            for r in corpus
            if r["rating"] is not None and r["rating"] <= max_rating
        }

    # topic: keyword scan over full corpus
    keywords = [kw.lower() for kw in query_row.get("expected_keywords", [])]
    if not keywords:
        return set()
    return {
        r["external_id"]
        for r in corpus
        if any(kw in r["text"].lower() for kw in keywords)
    }


# ---------------------------------------------------------------------------
# In-process retrieval (reuses real pipeline, no HTTP)
# ---------------------------------------------------------------------------

def retrieve_ids(
    query_text: str,
    business_id: str,
    k: int,
    business_name: str = "this restaurant",
) -> tuple[list[str], bool]:
    """Run the real retrieval pipeline at a forced k.

    Returns (external_ids_ranked, off_topic).
    Uses parse_query_with_llm + build_qdrant_filter + create_dense_retriever —
    same path as production, but with k controlled by the eval.
    """
    try:
        parsed = parse_query_with_llm(query_text, business_name=business_name)
    except Exception:
        parsed = {"off_topic": False, "query_embedding_text": query_text, "filter": {}}

    if parsed.get("off_topic", False):
        return [], True

    filter_dict = parsed.get("filter") or {}
    qdrant_filter = build_qdrant_filter(filter_dict, business_id=business_id)
    retriever = create_dense_retriever(qdrant_filter=qdrant_filter, k=k)
    docs = retriever.invoke(parsed["query_embedding_text"])
    ids = [d.metadata.get("external_id", "") for d in docs]
    return ids, False


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """recall@k = |relevant ∩ top-k retrieved| / |relevant|"""
    if not relevant:
        raise ValueError("relevant set is empty — call site should guard")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_queries() -> list[dict]:
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k eval harness.")
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated k values to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--business-id",
        default=None,
        help="Override business_id for all queries (default: use per-query value)",
    )
    parser.add_argument(
        "--detail-k",
        type=int,
        default=None,
        help="If set, also print a per-query detail table at this k value.",
    )
    args = parser.parse_args()

    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    queries = load_queries()

    # Determine the business_id to use (override or first query's value)
    biz_id = args.business_id or queries[0].get("business_id", "sandys-sandies")
    if args.business_id:
        for q in queries:
            q["business_id"] = biz_id

    print(f"\nLoading corpus for tenant `{biz_id}`…", flush=True)
    t0 = time.monotonic()
    corpus = build_corpus(biz_id)
    print(f"  {len(corpus)} reviews loaded in {time.monotonic()-t0:.1f}s\n", flush=True)

    if not corpus:
        print("ERROR: no reviews found for this business_id. Ingest reviews first.")
        sys.exit(1)

    # Pre-compute relevant sets (independent of k)
    rel_sets: list[set[str] | None] = []
    for q in queries:
        rel = relevant_ids(q, corpus)
        rel_sets.append(rel)

    n_topic = sum(1 for q in queries if q.get("class", "topic") == "topic")
    n_complaint = sum(1 for q in queries if q.get("class") == "complaint")
    n_na = sum(1 for r in rel_sets if r is not None and len(r) == 0)
    print(f"Queries: {len(queries)} total  ({n_topic} topic, {n_complaint} complaint)")
    print(f"N/A (empty relevant set): {n_na}\n")

    # Retrieval sweep — retrieve once per (query, k_max), then slice for smaller k
    k_max = max(k_values)
    print(f"Retrieving at k={k_max} (max)…", flush=True)
    retrieved_all: list[list[str]] = []
    off_topic_flags: list[bool] = []

    for i, q in enumerate(queries, 1):
        print(f"  [{i:2d}/{len(queries)}] {q['query'][:60]}", end=" ", flush=True)
        t_q = time.monotonic()
        ids, off_topic = retrieve_ids(
            q["query"], biz_id, k_max, business_name="this restaurant"
        )
        retrieved_all.append(ids)
        off_topic_flags.append(off_topic)
        tag = "(off-topic)" if off_topic else f"→ {len(ids)} docs"
        print(f"{tag}  {(time.monotonic()-t_q)*1000:.0f}ms", flush=True)

    # Compute recall@k for each k in the sweep
    print("\n---\n")
    print(f"## Recall@k Sweep — `{biz_id}` ({len(corpus)} reviews)\n")
    print(f"Corpus: {len(corpus)}  |  Queries: {len(queries)}  "
          f"({n_topic} topic / {n_complaint} complaint)  |  N/A: {n_na}\n")

    header = "| k | Mean recall@k | Topic recall@k | Complaint recall@k | Relevant-set sizes |"
    sep    = "|---|:---:|:---:|:---:|---|"
    print(header)
    print(sep)

    saturation_k = None

    for k in k_values:
        scores_all: list[float] = []
        scores_topic: list[float] = []
        scores_complaint: list[float] = []
        rel_size_strs: list[str] = []

        for q, rel, retrieved, ot in zip(queries, rel_sets, retrieved_all, off_topic_flags):
            if rel is None or ot:
                continue
            if len(rel) == 0:
                rel_size_strs.append("N/A")
                continue

            r = recall_at_k(retrieved, rel, k)
            rel_size_strs.append(str(len(rel)))
            scores_all.append(r)
            cls = q.get("class", "topic")
            if cls == "topic":
                scores_topic.append(r)
            elif cls == "complaint":
                scores_complaint.append(r)

        m_all = mean(scores_all)
        m_topic = mean(scores_topic)
        m_complaint = mean(scores_complaint)

        # Detect saturation (first k where mean ≥ corpus-size-bounded ceiling)
        if saturation_k is None and m_all is not None and m_all >= 0.999:
            saturation_k = k

        sizes_preview = ", ".join(rel_size_strs[:6]) + ("…" if len(rel_size_strs) > 6 else "")
        print(f"| {k:>4} | {fmt(m_all)} | {fmt(m_topic)} | {fmt(m_complaint)} | {sizes_preview} |")

    sat_note = f"k={saturation_k}" if saturation_k else f"k>{k_max}"
    print(f"\n**Corpus size:** {len(corpus)} reviews. "
          f"**Saturation:** recall@k plateaus at ≈{sat_note}.")

    # Optional per-query detail table at --detail-k
    if args.detail_k is not None:
        dk = args.detail_k
        print(f"\n---\n\n### Per-query detail at k={dk}\n")
        print("| # | Class | Query | Recall@k | Relevant | Retrieved∩Rel |")
        print("|---|-------|-------|:---:|:---:|:---:|")
        for i, (q, rel, retrieved, ot) in enumerate(
            zip(queries, rel_sets, retrieved_all, off_topic_flags), 1
        ):
            cls = q.get("class", "topic")
            q_short = q["query"][:55] + ("…" if len(q["query"]) > 55 else "")
            if ot:
                print(f"| {i} | {cls} | {q_short} | off-topic | — | — |")
            elif rel is None or len(rel) == 0:
                print(f"| {i} | {cls} | {q_short} | N/A | 0 | — |")
            else:
                r = recall_at_k(retrieved, rel, dk)
                inter = len(set(retrieved[:dk]) & rel)
                print(f"| {i} | {cls} | {q_short} | {r:.3f} | {len(rel)} | {inter} |")


if __name__ == "__main__":
    main()
