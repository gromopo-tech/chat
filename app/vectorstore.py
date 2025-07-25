from qdrant_client import QdrantClient, models
from app.config import Config
from app.utils import iso8601_to_timestamp


def get_qdrant():
    qdrant_host = Config.QDRANT_HOST
    return QdrantClient(host=qdrant_host, prefer_grpc=True)


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
