"""
Unit tests for GoogleTakeoutSource.

These tests use purely in-memory fixtures \u2014 no file I/O, no network, no Qdrant.
They verify the parser handles the full range of Google Business Profile / Takeout
JSON schemas we can encounter in the wild.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.ingestion.google_takeout import GoogleTakeoutSource, _parse_timestamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fixture(reviews: list[dict]) -> Path:
    """Write a Takeout JSON fixture to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump({"reviews": reviews}, tmp)
    tmp.flush()
    return Path(tmp.name)


VALID_REVIEW = {
    "reviewer": {"displayName": "Darcy Myers"},
    "starRating": "FIVE",
    "comment": "Roast duck sandwich, and the Ruben are my favorite.",
    "createTime": "2016-05-24T05:40:29.418Z",
    "updateTime": "2016-05-24T05:40:49.166Z",
    "name": "accounts/123/locations/456/reviews/review_abc",
}

NON_ENGLISH_REVIEW = {
    "reviewer": {"displayName": "Carlos Méndez"},
    "starRating": "FOUR",
    "comment": "¡Excelente comida! El pato asado estaba increíble.",
    "createTime": "2024-03-01T12:00:00.000Z",
    "updateTime": "2024-03-01T12:00:00.000Z",
    "name": "accounts/123/locations/456/reviews/review_es",
}

NO_COMMENT_REVIEW = {
    "reviewer": {"displayName": "Silent Reviewer"},
    "starRating": "THREE",
    "createTime": "2024-01-01T00:00:00.000Z",
    "updateTime": "2024-01-01T00:00:00.000Z",
    "name": "accounts/123/locations/456/reviews/review_nocomment",
}

EMPTY_COMMENT_REVIEW = {
    "reviewer": {"displayName": "Whitespace Only"},
    "starRating": "TWO",
    "comment": "   ",
    "createTime": "2024-01-01T00:00:00.000Z",
    "updateTime": "2024-01-01T00:00:00.000Z",
    "name": "accounts/123/locations/456/reviews/review_empty",
}

MISSING_NAME_REVIEW = {
    "reviewer": {"displayName": "Ghost"},
    "starRating": "ONE",
    "comment": "This review has no name field.",
    "createTime": "2024-01-01T00:00:00.000Z",
}

MISSING_REVIEWER_REVIEW = {
    "starRating": "FIVE",
    "comment": "Great anonymous review.",
    "createTime": "2024-06-01T10:30:00.000Z",
    "name": "accounts/123/locations/456/reviews/review_anon",
}


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_utc_z_suffix(self):
        ts = _parse_timestamp("2016-05-24T05:40:29.418Z")
        assert ts is not None
        assert isinstance(ts, float)
        assert ts > 0

    def test_offset_aware(self):
        ts = _parse_timestamp("2024-03-01T12:00:00+05:30")
        assert ts is not None

    def test_none_input(self):
        assert _parse_timestamp(None) is None

    def test_empty_string(self):
        assert _parse_timestamp("") is None

    def test_invalid_string(self):
        assert _parse_timestamp("not-a-date") is None


# ---------------------------------------------------------------------------
# GoogleTakeoutSource.load
# ---------------------------------------------------------------------------

class TestGoogleTakeoutSource:
    def setup_method(self):
        self.source = GoogleTakeoutSource()

    def test_parses_valid_review(self):
        path = _write_fixture([VALID_REVIEW])
        records = self.source.load("biz_001", input_path=path)

        assert len(records) == 1
        r = records[0]
        assert r.source == "google_takeout"
        assert r.business_id == "biz_001"
        assert r.external_id == "review_abc"
        assert r.author == "Darcy Myers"
        assert r.rating == 5
        assert r.text == "Roast duck sandwich, and the Ruben are my favorite."
        assert r.timestamp is not None and r.timestamp > 0

    def test_skips_review_with_no_comment(self):
        path = _write_fixture([NO_COMMENT_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        assert len(records) == 0

    def test_skips_review_with_whitespace_only_comment(self):
        path = _write_fixture([EMPTY_COMMENT_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        assert len(records) == 0

    def test_skips_review_missing_name(self):
        path = _write_fixture([MISSING_NAME_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        assert len(records) == 0

    def test_handles_missing_reviewer_displayname(self):
        path = _write_fixture([MISSING_REVIEWER_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        assert len(records) == 1
        assert records[0].author == "Unknown"

    def test_handles_non_english_text(self):
        path = _write_fixture([NON_ENGLISH_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        assert len(records) == 1
        assert "Excelente" in records[0].text
        assert records[0].rating == 4

    def test_mixed_batch(self):
        path = _write_fixture([VALID_REVIEW, NO_COMMENT_REVIEW, NON_ENGLISH_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        # Only reviews with comments should be returned
        assert len(records) == 2

    def test_star_rating_mapping(self):
        for star, expected in [("ONE", 1), ("TWO", 2), ("THREE", 3), ("FOUR", 4), ("FIVE", 5)]:
            review = {**VALID_REVIEW, "starRating": star,
                      "name": f"accounts/x/reviews/r_{star}"}
            path = _write_fixture([review])
            records = self.source.load("biz_x", input_path=path)
            assert records[0].rating == expected

    def test_stable_point_id_deterministic(self):
        path = _write_fixture([VALID_REVIEW])
        r1 = self.source.load("biz_001", input_path=path)[0]
        r2 = self.source.load("biz_001", input_path=path)[0]
        assert r1.stable_point_id() == r2.stable_point_id()

    def test_stable_point_id_differs_by_tenant(self):
        path = _write_fixture([VALID_REVIEW])
        r1 = self.source.load("biz_A", input_path=path)[0]
        r2 = self.source.load("biz_B", input_path=path)[0]
        assert r1.stable_point_id() != r2.stable_point_id()

    def test_extra_fields_preserved(self):
        path = _write_fixture([VALID_REVIEW])
        records = self.source.load("biz_001", input_path=path)
        assert "raw_name" in records[0].extra
        assert "update_time" in records[0].extra

    def test_empty_reviews_list(self):
        path = _write_fixture([])
        records = self.source.load("biz_001", input_path=path)
        assert records == []
