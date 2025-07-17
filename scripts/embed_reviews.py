import json
from pathlib import Path
from qdrant_client import models
from tqdm import tqdm
from qdrant_client.models import VectorParams, Distance
from app.models import embeddings_model
from app.vectorstore import get_qdrant, COLLECTION_NAME
from datetime import datetime

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Embed reviews and upload to Qdrant.")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=".",
        help="Path to the directory containing review files (default: current directory)",
    )
    return parser.parse_args()


VECTOR_SIZE = 3072


def iso8601_to_timestamp(dt_str):
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).timestamp()


def main():
    args = parse_args()
    reviews_dir = Path(args.dir)
    review_files = sorted(
        [
            f
            for f in reviews_dir.iterdir()
            if f.is_file() and f.name.startswith("reviews-")
        ]
    )
    if not review_files:
        print(f"No files starting with 'reviews-' found in {reviews_dir}")
        return

    all_points = []
    total_reviews = 0
    idx = 0
    for review_file in review_files:
        with open(review_file, "r") as f:
            reviews = json.load(f)["reviews"]
        print(f"🧠 Embedding {len(reviews)} reviews from {review_file.name}...")
        for review in tqdm(reviews, desc=review_file.name):
            if "comment" not in review or "name" not in review:
                continue
            text = review["comment"]
            embedding = embeddings_model.embed_query(text)
            # Map starRating (string) to int
            star_map = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5}
            rating = star_map.get(review.get("starRating", ""), None)
            create_time_str = review.get("createTime")
            create_time_ts = (
                iso8601_to_timestamp(create_time_str) if create_time_str else None
            )
            author = review.get("reviewer", {}).get("displayName", "Unknown")
            review_id = review["name"].split("/")[-1]
            all_points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "text": text,
                        "rating": rating,
                        "createTime": create_time_ts,
                        "author": author,
                        "review_id": review_id,
                    },
                )
            )
            idx += 1
        total_reviews += len(reviews)

    qdrant = get_qdrant()
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=16),
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=all_points)

    print(
        f"✅ Inserted {total_reviews} reviews from {len(review_files)} files into collection '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()
