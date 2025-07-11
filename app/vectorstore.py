from qdrant_client import QdrantClient
import os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor


COLLECTION_NAME = "reviews"


def get_qdrant():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        # Production/Cloud: use URL and API key
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        # Local/dev: use host and port
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        return QdrantClient(host=qdrant_host, port=qdrant_port)


def get_relevant_reviews(query_embedding: list[float], place_id: str, top_k=5):
    search = get_qdrant().search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        filter={"must": [{"key": "place_id", "match": {"value": place_id}}]},
    )
    return [hit.payload["text"] for hit in search]


def iso8601_to_timestamp(dt_str):
    # Handles 'Z' for UTC
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).timestamp()


def build_qdrant_filter(parsed_filter: dict) -> dict:
    """Convert parsed filter to Qdrant filter format."""
    must = []
    if not parsed_filter:
        return None
    if "rating" in parsed_filter:
        rating = parsed_filter["rating"]
        if "$in" in rating:
            must.append({"key": "rating", "match": {"any": rating["$in"]}})
        if "$gte" in rating or "$lte" in rating:
            rng = {}
            if "$gte" in rating:
                rng["gte"] = rating["$gte"]
            if "$lte" in rating:
                rng["lte"] = rating["$lte"]
            must.append({"key": "rating", "range": rng})
    if "languageCode" in parsed_filter:
        must.append(
            {"key": "languageCode", "match": {"value": parsed_filter["languageCode"]}}
        )
    if "publishTime" in parsed_filter and "$gte" in parsed_filter["publishTime"]:
        ts = iso8601_to_timestamp(parsed_filter["publishTime"]["$gte"])
        must.append({"key": "publishTime", "range": {"gte": ts}})
    return {"must": must} if must else None

#TODO: Pass stats to prompt_template
def get_review_stats_parallel(qdrant_client, collection_name, place_id):
    now = datetime.now(timezone.utc)
    periods = {
        "week": now - timedelta(days=7),
        "month": now - timedelta(days=30),
        "year": now - timedelta(days=365),
    }
    def count_for(period, rating, since):
        filter_ = {
            "must": [
                {"key": "place_id", "match": {"value": place_id}},
                {"key": "rating", "match": {"value": rating}},
                {"key": "publishTime", "range": {"gte": since.isoformat()}},
            ]
        }
        count = qdrant_client.count(
            collection_name=collection_name,
            filter=filter_
        ).count
        return (period, rating, count)
    stats = {period: {} for period in periods}
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(count_for, period, rating, since)
            for period, since in periods.items()
            for rating in range(1, 6)
        ]
        for future in futures:
            period, rating, count = future.result()
            stats[period][rating] = count
    return stats