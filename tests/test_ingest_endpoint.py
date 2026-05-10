"""
Tests for POST /ingest/google_takeout endpoint.

No Qdrant or Vertex AI calls are made — embed_and_upsert is mocked.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.ingestion.base import IngestResult
from app.main import app

SECRET = "test-secret"
AUTH_HEADER = f"Bearer {SECRET}"

VALID_REVIEW = {
    "reviewer": {"displayName": "Ada Lovelace"},
    "starRating": "FIVE",
    "comment": "Excellent place, highly recommend.",
    "createTime": "2024-03-01T10:00:00.000Z",
    "updateTime": "2024-03-01T10:00:00.000Z",
    "name": "accounts/1/locations/2/reviews/review_001",
}

NO_COMMENT_REVIEW = {
    "reviewer": {"displayName": "Silent Bob"},
    "starRating": "THREE",
    "comment": "",
    "createTime": "2024-03-02T10:00:00.000Z",
    "name": "accounts/1/locations/2/reviews/review_002",
}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("INGEST_SHARED_SECRET", SECRET)
    # Reload config so the new env var is picked up
    import importlib

    import app.config as cfg_module
    importlib.reload(cfg_module)
    # Also patch the reference held by main
    import app.main as main_module
    monkeypatch.setattr(main_module.Config, "INGEST_SHARED_SECRET", SECRET)
    return TestClient(app)


@patch("app.main.embed_and_upsert")
def test_happy_path(mock_upsert, client):
    mock_upsert.return_value = IngestResult(ingested=1, skipped=1, errors=[])

    resp = client.post(
        "/ingest/google_takeout",
        json={"business_id": "biz-a", "reviews": [VALID_REVIEW, NO_COMMENT_REVIEW]},
        headers={"Authorization": AUTH_HEADER},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ingested"] == 1
    assert body["skipped"] == 1
    assert body["errors"] == []
    mock_upsert.assert_called_once()


@patch("app.main.embed_and_upsert")
def test_business_id_propagated(mock_upsert, client):
    """Records passed to embed_and_upsert must all carry the request business_id."""
    mock_upsert.return_value = IngestResult(ingested=1, skipped=0, errors=[])

    client.post(
        "/ingest/google_takeout",
        json={"business_id": "tenant-xyz", "reviews": [VALID_REVIEW]},
        headers={"Authorization": AUTH_HEADER},
    )

    records = mock_upsert.call_args[0][0]
    assert all(r.business_id == "tenant-xyz" for r in records)


def test_missing_auth_header(client):
    resp = client.post(
        "/ingest/google_takeout",
        json={"business_id": "biz-a", "reviews": [VALID_REVIEW]},
    )
    assert resp.status_code == 422  # Header(...) makes it required → validation error


def test_wrong_secret(client):
    resp = client.post(
        "/ingest/google_takeout",
        json={"business_id": "biz-a", "reviews": [VALID_REVIEW]},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert resp.status_code == 401


@patch("app.main.embed_and_upsert")
def test_empty_reviews_list(mock_upsert, client):
    mock_upsert.return_value = IngestResult(ingested=0, skipped=0, errors=[])

    resp = client.post(
        "/ingest/google_takeout",
        json={"business_id": "biz-a", "reviews": []},
        headers={"Authorization": AUTH_HEADER},
    )

    assert resp.status_code == 200
    assert resp.json()["ingested"] == 0


@patch("app.main.embed_and_upsert")
def test_two_tenants_get_different_point_ids(mock_upsert, client):
    """Same review text + review ID, different tenants → different stable_point_id."""
    mock_upsert.return_value = IngestResult(ingested=1, skipped=0, errors=[])

    client.post(
        "/ingest/google_takeout",
        json={"business_id": "tenant-a", "reviews": [VALID_REVIEW]},
        headers={"Authorization": AUTH_HEADER},
    )
    records_a = mock_upsert.call_args[0][0]

    client.post(
        "/ingest/google_takeout",
        json={"business_id": "tenant-b", "reviews": [VALID_REVIEW]},
        headers={"Authorization": AUTH_HEADER},
    )
    records_b = mock_upsert.call_args[0][0]

    assert records_a[0].stable_point_id() != records_b[0].stable_point_id()
