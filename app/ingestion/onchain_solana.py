"""
OnChainReviewSource: Ingests reviews from the Vouched Solana Anchor program.

Design notes:
- Uses manual Borsh deserialization instead of anchorpy to avoid IDL format
  compatibility issues (anchorpy ~0.20.x predates the Anchor 0.31.x IDL spec).
- Review account layout (Anchor discriminator + Borsh-encoded struct):
    [0:8]   8 bytes  — account discriminator
    [8:40]  32 bytes — reviewer (Pubkey)
    [40:72] 32 bytes — reviewee (Pubkey)
    [72]    1 byte   — rating (u8)
    [73:77] 4 bytes  — comment length prefix (u32 little-endian)
    [77:77+N]        — comment (UTF-8)
- No timestamp field on the Review struct. We fetch the first transaction
  signature per PDA via getSignaturesForAddress and use getBlockTime on that
  slot. This is O(N) RPC calls; for production at scale you'd index slots via
  a Geyser plugin or maintain a separate timestamp store.
- Program ID: A1sSsTDoDrBkJ96fuHo9G89gHsEXVvcW6tNV39AfyWbF (devnet)
"""
from __future__ import annotations

import asyncio
import struct
import logging
from pathlib import Path

from solders.pubkey import Pubkey  # type: ignore[import-untyped]
import base58 as _base58
from solana.rpc.api import Client  # type: ignore[import-untyped]
from solana.rpc.types import MemcmpOpts  # type: ignore[import-untyped]

from app.ingestion.base import IngestResult, ReviewRecord, ReviewSource

logger = logging.getLogger(__name__)

# Vouched program deployed on Solana devnet
PROGRAM_ID = "A1sSsTDoDrBkJ96fuHo9G89gHsEXVvcW6tNV39AfyWbF"
DEFAULT_RPC_URL = "https://api.devnet.solana.com"

# Account discriminator for the Review account type (first 8 bytes of SHA256("account:Review"))
# Source: vouched/target/idl/review.json → accounts[0].discriminator
REVIEW_DISCRIMINATOR = bytes([124, 63, 203, 215, 226, 30, 222, 15])


def _decode_review_account(pubkey: str, data: bytes) -> dict | None:
    """Manually Borsh-decode a Review account from raw account data.

    Returns a dict with reviewer, reviewee, rating, comment keys, or None if
    the data is malformed or does not start with the expected discriminator.
    """
    if len(data) < 77:  # 8 + 32 + 32 + 1 + 4 = 77 minimum bytes
        logger.debug("Account %s: data too short (%d bytes), skipping", pubkey, len(data))
        return None

    if data[:8] != REVIEW_DISCRIMINATOR:
        logger.debug("Account %s: discriminator mismatch, skipping", pubkey)
        return None

    reviewer = str(Pubkey.from_bytes(data[8:40]))
    reviewee = str(Pubkey.from_bytes(data[40:72]))
    rating = data[72]

    comment_len = struct.unpack_from("<I", data, 73)[0]
    comment_end = 77 + comment_len
    if comment_end > len(data):
        logger.debug("Account %s: comment length %d exceeds data, skipping", pubkey, comment_len)
        return None

    comment = data[77:comment_end].decode("utf-8", errors="replace")

    return {"reviewer": reviewer, "reviewee": reviewee, "rating": rating, "comment": comment}


def _fetch_block_time(client: Client, account_pubkey: str) -> float | None:
    """Return the Unix timestamp of the earliest confirmed tx touching this account.

    Fetches the last page of signatures (oldest first) via getSignaturesForAddress,
    then calls getBlockTime on the slot of the first signature. Returns None on any
    RPC error so the record is still ingested without a timestamp.
    """
    try:
        resp = client.get_signatures_for_address(
            Pubkey.from_string(account_pubkey),
            limit=1,
            # 'before' omitted → most-recent signature (creation tx for a PDA
            # that is never updated; update_review would show a newer sig)
        )
        sigs = resp.value
        if not sigs:
            return None
        slot = sigs[0].slot
        block_time_resp = client.get_block_time(slot)
        return float(block_time_resp.value) if block_time_resp.value is not None else None
    except Exception as exc:
        logger.warning("Could not fetch block time for %s: %s", account_pubkey, exc)
        return None


async def _fetch_block_times_parallel(
    client: Client,
    account_pubkeys: list[str],
) -> dict[str, float | None]:
    """Fetch block times for all accounts in parallel via asyncio.gather."""

    def _fetch_one(pubkey: str) -> tuple[str, float | None]:
        return pubkey, _fetch_block_time(client, pubkey)

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _fetch_one, pk) for pk in account_pubkeys]
    results = await asyncio.gather(*tasks)
    return dict(results)


class OnChainReviewSource(ReviewSource):
    """Ingests reviews from the Vouched Solana Anchor program via RPC.

    Usage::

        source = OnChainReviewSource()
        records = source.load(
            business_id="my-restaurant",
            reviewee_pubkey="<merchant-wallet-address>",   # filter to one restaurant
            rpc_url="https://api.devnet.solana.com",       # optional
            fetch_timestamps=True,                         # optional, adds N RPC calls
        )
    """

    def load(
        self,
        business_id: str,
        *,
        reviewee_pubkey: str | None = None,
        rpc_url: str = DEFAULT_RPC_URL,
        fetch_timestamps: bool = True,
    ) -> list[ReviewRecord]:
        """Fetch and deserialize all Review accounts from the program.

        Args:
            business_id: Tenant identifier for Qdrant payload scoping.
            reviewee_pubkey: If provided, only returns reviews for this
                merchant wallet. If None, fetches all reviews in the program.
            rpc_url: Solana RPC endpoint. Defaults to devnet.
            fetch_timestamps: If True, fetches block time for each account
                (O(N) additional RPC calls). Set False for faster dry runs.
        """
        client = Client(rpc_url)
        program_pubkey = Pubkey.from_string(PROGRAM_ID)

        filters: list = [
            MemcmpOpts(offset=0, bytes=_base58.b58encode(REVIEW_DISCRIMINATOR).decode()),
        ]
        if reviewee_pubkey:
            filters.append(MemcmpOpts(offset=40, bytes=reviewee_pubkey))

        logger.info(
            "Fetching Review accounts from program %s (reviewee=%s)",
            PROGRAM_ID,
            reviewee_pubkey or "all",
        )

        resp = client.get_program_accounts(
            program_pubkey,
            filters=filters,
            encoding="base64",
        )

        accounts = resp.value
        logger.info("Found %d account(s) to process", len(accounts))

        decoded: list[tuple[str, dict]] = []
        for acct in accounts:
            pubkey_str = str(acct.pubkey)
            raw = bytes(acct.account.data)
            parsed = _decode_review_account(pubkey_str, raw)
            if parsed:
                decoded.append((pubkey_str, parsed))

        logger.info("%d account(s) decoded successfully", len(decoded))

        # Fetch block times in parallel (optional)
        block_times: dict[str, float | None] = {}
        if fetch_timestamps and decoded:
            pubkeys = [pk for pk, _ in decoded]
            block_times = asyncio.run(_fetch_block_times_parallel(client, pubkeys))

        records: list[ReviewRecord] = []
        for pubkey_str, parsed in decoded:
            records.append(
                ReviewRecord(
                    source="onchain_solana",
                    business_id=business_id,
                    external_id=pubkey_str,
                    author=parsed["reviewer"],
                    rating=parsed["rating"],
                    text=parsed["comment"],
                    timestamp=block_times.get(pubkey_str),
                    extra={
                        "reviewer": parsed["reviewer"],
                        "reviewee": parsed["reviewee"],
                        "program_id": PROGRAM_ID,
                        "rpc_url": rpc_url,
                    },
                )
            )

        return records
