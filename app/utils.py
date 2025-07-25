from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import qdrant_client


def iso8601_to_timestamp(dt_str):
    # Handles 'Z' for UTC
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).timestamp()

# TODO: Finish implementing and use in prompts
def get_review_stats_parallel(qdrant_client, collection_name, models: qdrant_client.models):
    """Get review stats for each rating (1-5) over the last week, month, and year."""
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