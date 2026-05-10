"""
Google Business Profile / Takeout review ingestion source.

Parses the JSON export format produced by Google Business Profile (GBP) takeout:
  {
    "reviews": [
      {
        "name": "accounts/.../locations/.../reviews/<review_id>",
        "reviewer": {"displayName": "..."},
        "starRating": "FIVE",        # ONE | TWO | THREE | FOUR | FIVE
        "comment": "...",            # optional — some reviews have no text
        "createTime": "2024-01-01T00:00:00.000Z",
        "updateTime": "2024-01-01T00:00:00.000Z"
      },
      ...
    ]
  }
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from app.ingestion.base import IngestResult, ReviewRecord, ReviewSource

logger = logging.getLogger(__name__)

STAR_MAP = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5}


def _parse_timestamp(dt_str: str | None) -> float | None:
    if not dt_str:
        return None
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(dt_str).timestamp()
    except ValueError:
        return None


class GoogleTakeoutSource(ReviewSource):
    """Ingests reviews from a Google Business Profile / Takeout JSON export file."""

    @classmethod
    def parse_records(cls, business_id: str, raw_reviews: list[dict]) -> list[ReviewRecord]:
        """Map a list of raw Google Takeout review dicts to ReviewRecord objects.

        Used by both the CLI file-based loader and the HTTP ingest endpoint.
        Reviews missing comment text, a name field, or an unrecognised star rating are skipped.
        """
        records: list[ReviewRecord] = []

        for raw in raw_reviews:
            comment = raw.get("comment", "").strip()
            if not comment:
                continue

            name = raw.get("name", "")
            external_id = name.split("/")[-1] if name else ""
            if not external_id:
                logger.warning("Review missing 'name' field; skipping: %s", raw)
                continue

            star_str = raw.get("starRating", "")
            rating = STAR_MAP.get(star_str)
            if rating is None:
                logger.warning("Unknown starRating %r for review %s; skipping", star_str, external_id)
                continue

            author = raw.get("reviewer", {}).get("displayName", "Unknown")
            timestamp = _parse_timestamp(raw.get("createTime"))

            records.append(
                ReviewRecord(
                    source="google_takeout",
                    business_id=business_id,
                    external_id=external_id,
                    author=author,
                    rating=rating,
                    text=comment,
                    timestamp=timestamp,
                    extra={
                        "update_time": raw.get("updateTime"),
                        "raw_name": name,
                    },
                )
            )

        return records

    def load(self, business_id: str, *, input_path: str | Path, **kwargs) -> list[ReviewRecord]:
        """Parse a Google Takeout JSON file and return ReviewRecords.

        Args:
            business_id: Tenant identifier (e.g. Firestore business doc ID).
            input_path: Path to the JSON export file.

        Returns:
            List of ReviewRecord objects. Reviews with no comment text are skipped.
        """
        path = Path(input_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        raw_reviews = data.get("reviews", [])
        records = self.parse_records(business_id, raw_reviews)

        logger.info(
            "Parsed %d records from %s (skipped %d with no comment)",
            len(records),
            path.name,
            len(raw_reviews) - len(records),
        )
        return records


def ingest_from_file(business_id: str, input_path: str | Path) -> IngestResult:
    """Convenience wrapper: parse file → embed → upsert. Returns IngestResult.

    Import here is deferred to avoid circular imports between ingestion and vectorstore.
    """
    from app.ingestion.pipeline import embed_and_upsert

    source = GoogleTakeoutSource()
    records = source.load(business_id, input_path=input_path)
    return embed_and_upsert(records)
