"""
Unit tests for multi-tenant Qdrant isolation.

Verifies that a query scoped to business_id=A cannot retrieve points
belonging to business_id=B. Uses qdrant-client's in-memory mode
(QdrantClient(":memory:")) so no running Qdrant instance is needed.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

VECTOR_SIZE = 4  # tiny vectors for test speed


def _make_client() -> QdrantClient:
    return QdrantClient(":memory:")


def _create_collection(client: QdrantClient, name: str = "reviews") -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
    )


def _upsert(client: QdrantClient, point_id: int, business_id: str, text: str,
            vector: list[float], collection: str = "reviews") -> None:
    client.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector={"dense": vector},
                payload={"text": text, "business_id": business_id, "rating": 5},
            )
        ],
    )


def _search(client: QdrantClient, query_vector: list[float],
            business_id: str | None, collection: str = "reviews") -> list[dict]:
    """Run a dense search, optionally filtering by business_id."""
    qdrant_filter = None
    if business_id is not None:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="business_id", match=models.MatchValue(value=business_id)
                )
            ]
        )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        using="dense",
        query_filter=qdrant_filter,
        limit=10,
        with_payload=True,
    )
    return [r.payload for r in results.points]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultitenantFilter:
    def setup_method(self):
        self.client = _make_client()
        _create_collection(self.client)

        # Tenant A reviews
        _upsert(self.client, 1, "biz_A", "Great duck sandwich", [1.0, 0.0, 0.0, 0.0])
        _upsert(self.client, 2, "biz_A", "Love the live music", [0.9, 0.1, 0.0, 0.0])

        # Tenant B reviews
        _upsert(self.client, 3, "biz_B", "Amazing tacos here", [0.0, 1.0, 0.0, 0.0])
        _upsert(self.client, 4, "biz_B", "Slow service but good food", [0.0, 0.9, 0.1, 0.0])

    def test_biz_a_query_returns_only_biz_a(self):
        results = _search(self.client, [1.0, 0.0, 0.0, 0.0], business_id="biz_A")
        assert len(results) == 2
        for r in results:
            assert r["business_id"] == "biz_A"

    def test_biz_b_query_returns_only_biz_b(self):
        results = _search(self.client, [0.0, 1.0, 0.0, 0.0], business_id="biz_B")
        assert len(results) == 2
        for r in results:
            assert r["business_id"] == "biz_B"

    def test_biz_a_query_excludes_biz_b(self):
        results = _search(self.client, [1.0, 0.0, 0.0, 0.0], business_id="biz_A")
        business_ids = {r["business_id"] for r in results}
        assert "biz_B" not in business_ids

    def test_biz_b_query_excludes_biz_a(self):
        results = _search(self.client, [0.0, 1.0, 0.0, 0.0], business_id="biz_B")
        business_ids = {r["business_id"] for r in results}
        assert "biz_A" not in business_ids

    def test_no_filter_returns_all_tenants(self):
        """Without a business_id filter all tenants' data is visible \u2014 confirms the
        filter is what provides isolation, not a collection-level guarantee."""
        results = _search(self.client, [1.0, 0.0, 0.0, 0.0], business_id=None)
        business_ids = {r["business_id"] for r in results}
        assert "biz_A" in business_ids
        assert "biz_B" in business_ids

    def test_unknown_business_id_returns_empty(self):
        results = _search(self.client, [1.0, 0.0, 0.0, 0.0], business_id="biz_UNKNOWN")
        assert results == []

    def test_build_qdrant_filter_injects_business_id(self):
        """Verify the production filter builder always injects business_id as a must condition."""
        from app.vectorstore import build_qdrant_filter

        f = build_qdrant_filter({}, business_id="biz_A")
        assert f is not None
        keys = [cond.key for cond in f.must]
        assert "business_id" in keys

    def test_build_qdrant_filter_no_business_id_is_none(self):
        """Without business_id and no parsed filter, filter should be None (no restriction)."""
        from app.vectorstore import build_qdrant_filter

        f = build_qdrant_filter({}, business_id=None)
        assert f is None

    def test_build_qdrant_filter_combines_business_id_and_rating(self):
        from app.vectorstore import build_qdrant_filter

        f = build_qdrant_filter({"rating": {"$in": [4, 5]}}, business_id="biz_A")
        assert f is not None
        keys = [cond.key for cond in f.must]
        assert "business_id" in keys
        assert "rating" in keys
