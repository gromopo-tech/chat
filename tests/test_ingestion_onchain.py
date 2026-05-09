"""
Unit tests for OnChainReviewSource.

Uses a captured devnet account as a byte-level fixture — no RPC, no network.
Verifies discriminator filtering, Borsh deserialization, and correct mapping
to ReviewRecord fields.
"""
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.base import ReviewRecord
from app.ingestion.onchain_solana import (
    REVIEW_DISCRIMINATOR,
    OnChainReviewSource,
    _decode_review_account,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic account data matching the on-chain layout
# ---------------------------------------------------------------------------

_REVIEWER = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
_REVIEWEE = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
_RATING = 4
_COMMENT = "Great tacos, would come back!"


def _build_account_data(
    reviewer: str = _REVIEWER,
    reviewee: str = _REVIEWEE,
    rating: int = _RATING,
    comment: str = _COMMENT,
    discriminator: bytes = REVIEW_DISCRIMINATOR,
) -> bytes:
    """Synthesize raw account bytes matching the on-chain Review layout."""
    from solders.pubkey import Pubkey  # type: ignore[import-untyped]

    reviewer_bytes = bytes(Pubkey.from_string(reviewer))
    reviewee_bytes = bytes(Pubkey.from_string(reviewee))
    comment_bytes = comment.encode("utf-8")
    comment_len = struct.pack("<I", len(comment_bytes))

    return discriminator + reviewer_bytes + reviewee_bytes + bytes([rating]) + comment_len + comment_bytes


# ---------------------------------------------------------------------------
# _decode_review_account unit tests
# ---------------------------------------------------------------------------

class TestDecodeReviewAccount:
    def test_valid_account_decoded_correctly(self):
        data = _build_account_data()
        result = _decode_review_account("somepubkey", data)
        assert result is not None
        assert result["reviewer"] == _REVIEWER
        assert result["reviewee"] == _REVIEWEE
        assert result["rating"] == _RATING
        assert result["comment"] == _COMMENT

    def test_wrong_discriminator_returns_none(self):
        bad_disc = bytes([0, 1, 2, 3, 4, 5, 6, 7])
        data = _build_account_data(discriminator=bad_disc)
        assert _decode_review_account("somepubkey", data) is None

    def test_data_too_short_returns_none(self):
        assert _decode_review_account("somepubkey", b"\x00" * 10) is None

    def test_unicode_comment_decoded(self):
        comment = "¡Excelente! 🌮"
        data = _build_account_data(comment=comment)
        result = _decode_review_account("somepubkey", data)
        assert result is not None
        assert result["comment"] == comment

    def test_empty_comment(self):
        data = _build_account_data(comment="")
        result = _decode_review_account("somepubkey", data)
        assert result is not None
        assert result["comment"] == ""

    def test_max_length_comment(self):
        comment = "a" * 2500
        data = _build_account_data(comment=comment)
        result = _decode_review_account("somepubkey", data)
        assert result is not None
        assert len(result["comment"]) == 2500

    def test_rating_boundaries(self):
        for rating in [1, 5]:
            data = _build_account_data(rating=rating)
            result = _decode_review_account("somepubkey", data)
            assert result is not None
            assert result["rating"] == rating


# ---------------------------------------------------------------------------
# OnChainReviewSource.load — patched RPC
# ---------------------------------------------------------------------------

def _make_mock_account(pubkey_str: str, data: bytes):
    """Build a mock object matching the solana-py getProgramAccounts response shape."""
    from solders.pubkey import Pubkey  # type: ignore[import-untyped]

    mock_acct = MagicMock()
    mock_acct.pubkey = Pubkey.from_string(pubkey_str)
    mock_acct.account.data = list(data)  # solana-py returns list of ints for base64 encoding
    return mock_acct


class TestOnChainReviewSource:
    @pytest.fixture()
    def fake_account_data(self):
        return _build_account_data()

    def test_load_produces_review_records(self, fake_account_data):
        mock_client = MagicMock()
        mock_client.get_program_accounts.return_value.value = [
            _make_mock_account(_REVIEWER, fake_account_data),
        ]

        with patch("app.ingestion.onchain_solana.Client", return_value=mock_client), \
             patch("app.ingestion.onchain_solana.asyncio.run", return_value={_REVIEWER: 1_700_000_000.0}):
            source = OnChainReviewSource()
            records = source.load("biz_test", reviewee_pubkey=_REVIEWEE, fetch_timestamps=True)

        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, ReviewRecord)
        assert rec.source == "onchain_solana"
        assert rec.business_id == "biz_test"
        assert rec.author == _REVIEWER
        assert rec.rating == _RATING
        assert rec.text == _COMMENT
        assert rec.extra["reviewee"] == _REVIEWEE
        assert rec.extra["program_id"] == "A1sSsTDoDrBkJ96fuHo9G89gHsEXVvcW6tNV39AfyWbF"

    def test_load_skips_malformed_accounts(self):
        mock_client = MagicMock()
        bad_data = b"\x00" * 5  # too short, will be skipped
        mock_client.get_program_accounts.return_value.value = [
            _make_mock_account(_REVIEWER, bad_data),
        ]

        with patch("app.ingestion.onchain_solana.Client", return_value=mock_client):
            source = OnChainReviewSource()
            records = source.load("biz_test", fetch_timestamps=False)

        assert records == []

    def test_load_without_timestamps(self, fake_account_data):
        mock_client = MagicMock()
        mock_client.get_program_accounts.return_value.value = [
            _make_mock_account(_REVIEWER, fake_account_data),
        ]

        with patch("app.ingestion.onchain_solana.Client", return_value=mock_client):
            source = OnChainReviewSource()
            records = source.load("biz_test", fetch_timestamps=False)

        assert len(records) == 1
        assert records[0].timestamp is None

    def test_stable_point_id_is_deterministic(self, fake_account_data):
        mock_client = MagicMock()
        mock_client.get_program_accounts.return_value.value = [
            _make_mock_account(_REVIEWER, fake_account_data),
        ]

        with patch("app.ingestion.onchain_solana.Client", return_value=mock_client):
            source = OnChainReviewSource()
            r1 = source.load("biz_test", fetch_timestamps=False)
            r2 = source.load("biz_test", fetch_timestamps=False)

        assert r1[0].stable_point_id() == r2[0].stable_point_id()

    def test_different_business_ids_produce_different_point_ids(self, fake_account_data):
        mock_client = MagicMock()
        mock_client.get_program_accounts.return_value.value = [
            _make_mock_account(_REVIEWER, fake_account_data),
        ]

        with patch("app.ingestion.onchain_solana.Client", return_value=mock_client):
            source = OnChainReviewSource()
            r_a = source.load("biz_A", fetch_timestamps=False)
            mock_client.get_program_accounts.return_value.value = [
                _make_mock_account(_REVIEWER, fake_account_data),
            ]
            r_b = source.load("biz_B", fetch_timestamps=False)

        assert r_a[0].stable_point_id() != r_b[0].stable_point_id()
