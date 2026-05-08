"""
Shared embedding + upsert pipeline used by all ingestion sources.

Both GoogleTakeoutSource and OnChainSolanaSource produce ReviewRecord objects
and hand them off here for embedding and upsert into Qdrant.
"""
from __future__ import annotations

import time

import structlog
from qdrant_client import models as qdrant_models

from app.config import Config
from app.ingestion.base import IngestResult, ReviewRecord
from app.vectorstore import ensure_collection, get_qdrant
from app.vertexai_models import get_hybrid_embeddings

log = structlog.get_logger(__name__)


def embed_and_upsert(records: list[ReviewRecord]) -> IngestResult:
    """Embed each ReviewRecord and upsert into Qdrant.

    Uses stable point IDs derived from (source, business_id, external_id) so that
    repeated ingestion runs are idempotent — existing points are overwritten, not
    duplicated.

    Args:
        records: List of ReviewRecord objects from any ingestion source.

    Returns:
        IngestResult with counts of ingested, skipped, and errored records.
    """
    result = IngestResult()
    qdrant = get_qdrant()
    ensure_collection(qdrant)

    points: list[qdrant_models.PointStruct] = []

    for record in records:
        t0 = time.monotonic()
        try:
            embeddings = get_hybrid_embeddings(record.text)
            latency_ms = (time.monotonic() - t0) * 1000

            log.info(
                "embedded_record",
                source=record.source,
                business_id=record.business_id,
                external_id=record.external_id,
                latency_ms=round(latency_ms, 1),
            )

            vectors = {"dense": embeddings["dense"]}
            if embeddings.get("sparse") is not None:
                vectors["sparse"] = embeddings["sparse"]

            payload = {
                "text": record.text,
                "rating": record.rating,
                "author": record.author,
                "source": record.source,
                "business_id": record.business_id,
                "external_id": record.external_id,
                "createTime": record.timestamp,
            }
            payload.update(record.extra)

            points.append(
                qdrant_models.PointStruct(
                    id=record.stable_point_id(),
                    vector=vectors,
                    payload=payload,
                )
            )
        except Exception as exc:
            log.error(
                "embed_error",
                external_id=record.external_id,
                error=str(exc),
            )
            result.errors.append(f"{record.external_id}: {exc}")

    if points:
        t0 = time.monotonic()
        qdrant.upsert(collection_name=Config.COLLECTION_NAME, points=points)
        latency_ms = (time.monotonic() - t0) * 1000
        log.info(
            "upserted_batch",
            count=len(points),
            collection=Config.COLLECTION_NAME,
            latency_ms=round(latency_ms, 1),
        )
        result.ingested = len(points)
    else:
        log.info("no_points_to_upsert")

    result.skipped = len(records) - len(points) - len(result.errors)
    return result
