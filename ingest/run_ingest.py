#!/usr/bin/env python
"""
Ingestion runner — embeds reviews from a source and upserts to Qdrant.

Usage:
    python scripts/run_ingestion.py \\
        --source google_takeout \\
        --business-id <business_id> \\
        --input <path_to_reviews.json>

    python scripts/run_ingestion.py \\
        --source onchain_solana \\
        --business-id <business_id>          # fetches from devnet RPC

Examples:
    # Ingest the bundled sample reviews
    python scripts/run_ingestion.py \\
        --source google_takeout \\
        --business-id demo \\
        --input reviews/reviews-sample.json

    # Ingest on-chain reviews (Phase C — requires anchorpy + funded RPC)
    python scripts/run_ingestion.py \\
        --source onchain_solana \\
        --business-id demo \\
        --reviewee <restaurant_wallet_pubkey>
"""
from __future__ import annotations

import argparse
import sys

import structlog

from app.logging_config import setup_logging

setup_logging()
log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed and upsert reviews into Qdrant.")
    parser.add_argument(
        "--source",
        required=True,
        choices=["google_takeout", "onchain_solana"],
        help="Which ingestion source to use.",
    )
    parser.add_argument(
        "--business-id",
        required=True,
        dest="business_id",
        help="Tenant identifier stored in Qdrant payload for multi-tenancy filtering.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to the input file (required for google_takeout).",
    )
    parser.add_argument(
        "--reviewee",
        help="Reviewee wallet public key (required for onchain_solana).",
    )
    parser.add_argument(
        "--rpc-url",
        dest="rpc_url",
        default="https://api.devnet.solana.com",
        help="Solana RPC URL (onchain_solana only). Defaults to devnet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "google_takeout":
        if not args.input_path:
            print("--input is required for --source google_takeout", file=sys.stderr)
            sys.exit(1)

        from app.ingestion.google_takeout import GoogleTakeoutSource
        from app.ingestion.pipeline import embed_and_upsert

        log.info("ingestion_start", source="google_takeout", business_id=args.business_id,
                 input=args.input_path)
        source = GoogleTakeoutSource()
        records = source.load(args.business_id, input_path=args.input_path)
        result = embed_and_upsert(records)

    elif args.source == "onchain_solana":
        from app.ingestion.onchain_solana import OnChainReviewSource
        from app.ingestion.pipeline import embed_and_upsert

        log.info("ingestion_start", source="onchain_solana", business_id=args.business_id,
                 reviewee=args.reviewee, rpc_url=args.rpc_url)
        source = OnChainReviewSource()
        records = source.load(
            args.business_id,
            reviewee_pubkey=args.reviewee,
            rpc_url=args.rpc_url,
        )
        result = embed_and_upsert(records)

    log.info(
        "ingestion_complete",
        source=args.source,
        ingested=result.ingested,
        skipped=result.skipped,
        errors=len(result.errors),
    )
    if result.errors:
        log.warning("ingestion_errors", errors=result.errors)

    print(
        f"\n✅ Ingestion complete — ingested: {result.ingested}, "
        f"skipped: {result.skipped}, errors: {len(result.errors)}"
    )


if __name__ == "__main__":
    main()
