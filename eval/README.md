# Eval Harness

Measures retrieval quality via recall@k across a range of k values. Runs entirely in-process — no HTTP server required — so it exercises the real retrieval pipeline at controlled k without the LLM response layer.

## Metric

**recall@k** = |relevant ∩ top-k retrieved| / |relevant|, averaged per query.

Relevant sets are derived from payload data already in Qdrant — no hand-labeling:

| Query class | Relevant set definition |
|---|---|
| `complaint` | All reviews with `rating ≤ relevant_rating_max` (default: ≤ 2) |
| `topic` | All reviews whose text contains any `expected_keyword` (case-insensitive) |

Queries whose relevant set is empty are reported as **N/A** and excluded from the mean — they indicate a fixture gap against the current corpus, not a retrieval failure.

## Method

The harness reuses the real production retrieval path in-process:

1. `parse_query_with_llm` extracts the query embedding text and any metadata filters (rating range, time window) — identical to production.
2. `build_qdrant_filter` applies those filters plus the `business_id` tenant scope.
3. `create_dense_retriever` runs the dense vector search at a forced k.

The LLM response step is excluded — this measures retrieval quality, not answer quality.

Retrieval runs once at `k_max`, then results are sliced for all smaller k values, so the sweep costs 20 Vertex AI embedding calls regardless of how many k values are tested.

## Results

Corpus: **556 reviews**, single tenant (`sandys-sandies`). 20 queries: 16 topic, 4 complaint.

| k | Mean recall@k | Topic recall@k | Complaint recall@k |
|---|:---:|:---:|:---:|
| 5 | 0.053 | 0.028 | 0.152 |
| 10 | 0.102 | 0.052 | 0.303 |
| 25 | 0.233 | 0.102 | 0.758 |
| **50** | **0.338** | **0.172** | **1.000** |
| 100 | 0.430 | 0.288 | 1.000 |
| 250 | 0.692 | 0.616 | 1.000 |
| **500** | **0.977** | **0.971** | **1.000** |

### Finding A — dynamic k is load-bearing at this corpus size

At k=50 (a common RAG default), topic recall is **0.172** — the retriever misses 83% of relevant docs for broad analytical queries. At k=500, topic recall reaches **0.971**. The pipeline uses dynamic k (`_get_k_value_for_query`): analytical and summary queries get k=1000, comparison queries k=100, specific-example queries k=30. The sweep quantifies what a flat k=50 would cost.

### Finding B — LLM-extracted filters protect complaint queries from k-starvation

The `rating∈[1,2]` filter emitted by the query parser for complaint queries shrinks the candidate pool to 33 docs (the 1–2 star reviews in this corpus). As a result, complaint recall reaches **1.000 at k=50** — well before topic queries saturate — and stays flat through k=500. This interaction is invisible to hand-testing: it only appears when you measure recall@k split by query class across a real corpus.

## Usage

```sh
# From repo root, venv active, Qdrant running, reviews ingested
python eval/run_eval.py

# Custom k sweep
python eval/run_eval.py --k-values 5,10,50,100,250,500

# Override tenant
python eval/run_eval.py --business-id my-tenant-id

# Per-query detail at a specific k (for spot-checking)
python eval/run_eval.py --detail-k 50
```

## Limitations

- **Keyword relevance is a proxy.** A review is labeled relevant if it contains an expected keyword, not because it was hand-judged relevant to the query. Broad keywords ("food", "great") produce large relevant sets (~400 docs) that mechanically bound recall@k near k/|relevant| at small k — this is the k-starvation effect, not a scoring artifact.
- **Single tenant, single corpus snapshot.** Numbers reflect one business's 556 reviews and don't directly transfer to other tenants. The *structural* findings should generalize — filters decouple complaint queries from k-starvation, and analytical queries require high k for large relevant sets — but the specific recall values and the k at which complaint queries saturate depend on corpus size and negative-review rate. A tenant with 50 negative reviews needs k>50 before complaint recall saturates; one with 5000 total reviews will see topic recall plateau at a much higher k.
- **Retrieval only.** The eval does not measure answer quality, faithfulness, or whether the LLM correctly uses retrieved context.
