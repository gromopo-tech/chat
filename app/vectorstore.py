from qdrant_client import QdrantClient, models
from app.config import Config
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor


def get_qdrant():
    qdrant_host = Config.QDRANT_HOST
    return QdrantClient(host=qdrant_host, prefer_grpc=True)


def iso8601_to_timestamp(dt_str):
    # Handles 'Z' for UTC
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).timestamp()


def build_qdrant_filter(parsed_filter: dict) -> models.Filter:
    """Convert parsed filter to Qdrant models.Filter format for gRPC."""
    must = []
    if not parsed_filter:
        return None
    if "rating" in parsed_filter:
        rating = parsed_filter["rating"]
        if "$in" in rating:
            must.append(
                models.FieldCondition(
                    key="rating", match=models.MatchAny(any=rating["$in"])
                )
            )
        if "$gte" in rating or "$lte" in rating:
            rng = {}
            if "$gte" in rating:
                rng["gte"] = rating["$gte"]
            if "$lte" in rating:
                rng["lte"] = rating["$lte"]
            must.append(models.FieldCondition(key="rating", range=models.Range(**rng)))
    if "createTime" in parsed_filter and "$gte" in parsed_filter["createTime"]:
        ts = iso8601_to_timestamp(parsed_filter["createTime"]["$gte"])
        must.append(models.FieldCondition(key="createTime", range=models.Range(gte=ts)))
    return models.Filter(must=must) if must else None


# TODO: Pass stats to prompt_template
def get_review_stats_parallel(qdrant_client, collection_name):
    now = datetime.now(timezone.utc)
    periods = {
        "week": now - timedelta(days=7),
        "month": now - timedelta(days=30),
        "year": now - timedelta(days=365),
    }

    def count_for(period, rating, since):
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="rating", match=models.MatchValue(value=rating)
                ),
                models.FieldCondition(
                    key="createTime",
                    range=models.Range(gte=iso8601_to_timestamp(since.isoformat())),
                ),
            ]
        )
        count = qdrant_client.count(
            collection_name=collection_name, filter=filter_
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
