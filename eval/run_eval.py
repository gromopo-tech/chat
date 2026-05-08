#!/usr/bin/env python
"""
Eval harness \u2014 recall@k against ground-truth keyword expectations.

Runs each query from eval/queries.jsonl through the /query endpoint,
checks whether expected keywords appear in any of the retrieved context
chunks, and prints a Markdown summary table.

Usage:
    # With the chat service running on localhost:8080
    python eval/run_eval.py

    # Against a remote deployment
    python eval/run_eval.py --base-url https://your-service-url

    # Adjust context window
    python eval/run_eval.py --k 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

QUERIES_FILE = Path(__file__).parent / "queries.jsonl"


def load_queries() -> list[dict]:
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def run_query(client: httpx.Client, base_url: str, query: str, business_id: str | None) -> dict:
    payload = {"query": query}
    if business_id:
        payload["business_id"] = business_id
    resp = client.post(f"{base_url}/query", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def keyword_hit(context_chunks: list[str], keywords: list[str]) -> bool:
    """Return True if at least one expected keyword appears in any context chunk."""
    combined = " ".join(context_chunks).lower()
    return any(kw.lower() in combined for kw in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval harness.")
    parser.add_argument("--base-url", default="http://localhost:8080",
                        help="Base URL of the chat service.")
    parser.add_argument("--k", type=int, default=None,
                        help="Override k (number of retrieved docs). Uses server default if omitted.")
    args = parser.parse_args()

    queries = load_queries()
    results = []

    with httpx.Client() as client:
        for i, q in enumerate(queries, 1):
            query_text = q["query"]
            expected_keywords = q.get("expected_keywords", [])
            business_id = q.get("business_id")

            t0 = time.monotonic()
            try:
                response = run_query(client, args.base_url, query_text, business_id)
                latency_ms = round((time.monotonic() - t0) * 1000)

                context = response.get("context", [])
                hit = keyword_hit(context, expected_keywords)

                results.append({
                    "id": i,
                    "query": query_text,
                    "hit": hit,
                    "retrieved": len(context),
                    "latency_ms": latency_ms,
                    "error": None,
                })
            except Exception as exc:
                latency_ms = round((time.monotonic() - t0) * 1000)
                results.append({
                    "id": i,
                    "query": query_text,
                    "hit": False,
                    "retrieved": 0,
                    "latency_ms": latency_ms,
                    "error": str(exc),
                })

    # Print Markdown table
    hits = sum(1 for r in results if r["hit"])
    total = len(results)
    recall = hits / total if total else 0.0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0

    print("\n## RAG Eval Results\n")
    print(f"**Recall@k (keyword match): {hits}/{total} = {recall:.1%}**")
    print(f"**Average latency: {avg_latency:.0f}ms**\n")

    # Table header
    print("| # | Query | Hit | Retrieved | Latency (ms) | Error |")
    print("|---|-------|-----|-----------|--------------|-------|")
    for r in results:
        hit_str = "✅" if r["hit"] else "❌"
        query_short = r["query"][:60] + ("…" if len(r["query"]) > 60 else "")
        error_str = r["error"] or ""
        print(f"| {r['id']} | {query_short} | {hit_str} | {r['retrieved']} | {r['latency_ms']} | {error_str} |")

    print(f"\n**Summary: {hits}/{total} queries had at least one expected keyword in retrieved context.**")


if __name__ == "__main__":
    main()
