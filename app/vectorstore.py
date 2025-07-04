from qdrant_client import QdrantClient
import os


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
