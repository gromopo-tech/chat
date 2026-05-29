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
    # Ingest a single file
    python scripts/run_ingestion.py \\
        --source google_takeout \\
        --business-id demo \\
        --input reviews/reviews-sample.json

    # Ingest all JSONs in a directory (e.g. a Google Takeout export folder)
    # Files that don't contain a "reviews" key are automatically skipped.
    python scripts/run_ingestion.py \\
        --source google_takeout \\
        --business-id demo \\
        --input path/to/takeout-folder/

    # Ingest on-chain reviews (Phase C — requires anchorpy + funded RPC)
    python scripts/run_ingestion.py \\
        --source onchain_solana \\
        --business-id demo \\
        --reviewee <restaurant_wallet_pubkey>
"""
from __future__ import annotations

import argparse
import os
import sys

import structlog
from dotenv import load_dotenv

# Load .env before any app.* imports — app.config reads os.getenv() at import time
# and docker-compose loads .env automatically, but direct `python3 -m` invocations do not.
#
# QDRANT_HOST special case: .env contains "qdrant" (the Docker service hostname) which only
# resolves inside the Docker network. Local scripts must use the "localhost" default instead.
# We only keep the .env value if the variable was already set in the real shell environment.
_qdrant_host_pre = os.environ.get("QDRANT_HOST")
load_dotenv()
from app.logging_config import setup_logging  # noqa: E402

if _qdrant_host_pre is None:
    os.environ.pop("QDRANT_HOST", None)


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

        from pathlib import Path

        from app.ingestion.google_takeout import GoogleTakeoutSource
        from app.ingestion.pipeline import embed_and_upsert

        source = GoogleTakeoutSource()
        input_path = Path(args.input_path)

        if input_path.is_dir():
            json_files = sorted(input_path.rglob("*.json"))
            log.info("ingestion_start", source="google_takeout", business_id=args.business_id,
                     input=str(input_path), json_files_found=len(json_files))
            records = []
            for json_file in json_files:
                file_records = source.load(args.business_id, input_path=json_file)
                if file_records:
                    log.info("file_parsed", file=json_file.name, records=len(file_records))
                    records.extend(file_records)
                else:
                    log.info("file_skipped", file=json_file.name, reason="no reviews found")
        else:
            log.info("ingestion_start", source="google_takeout", business_id=args.business_id,
                     input=args.input_path)
            records = source.load(args.business_id, input_path=input_path)

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
