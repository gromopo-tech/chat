"""
Base abstractions for pluggable review ingestion sources.

All ingestion sources implement ReviewSource and produce ReviewRecord objects
that are embedded and upserted into the shared Qdrant collection with
per-business payload filtering for multi-tenancy.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ReviewRecord:
    """Canonical representation of a review from any source."""
    source: Literal["google_takeout", "onchain_solana"]
    business_id: str
    external_id: str          # source-specific unique ID (review_id, account pubkey, etc.)
    author: str
    rating: int               # 1–5
    text: str
    timestamp: float | None = None   # Unix timestamp; None if unavailable
    extra: dict = field(default_factory=dict)

    def stable_point_id(self) -> int:
        """Deterministic Qdrant point ID derived from (source, business_id, external_id).

        Using a stable ID means repeated ingestion runs upsert (overwrite) existing
        points rather than creating duplicates.
        """
        key = f"{self.source}:{self.business_id}:{self.external_id}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


@dataclass
class IngestResult:
    ingested: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class ReviewSource(ABC):
    """Abstract base class for a review ingestion source."""

    @abstractmethod
    def load(self, business_id: str, **kwargs) -> list[ReviewRecord]:
        """Load and parse raw records from the source.

        Args:
            business_id: Tenant identifier used to scope Qdrant payloads.
            **kwargs: Source-specific arguments (e.g. input file path, RPC URL).

        Returns:
            List of ReviewRecord objects ready for embedding.
        """
        ...
