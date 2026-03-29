from qdrant_client import QdrantClient, models
from app.config import Config
from app.utils import iso8601_to_timestamp
from app.vertexai_models import get_query_embeddings
from typing import List, Dict, Any


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


def hybrid_search(query_text: str, qdrant_filter: models.Filter = None, k: int = 20) -> List[Dict[str, Any]]:
    """Perform hybrid search using both dense and sparse vectors (if available)."""
    qdrant = get_qdrant()
    
    # Get query embeddings
    query_embeddings = get_query_embeddings(query_text)
    dense_vector = query_embeddings['dense']
    
    # For now, only use dense search since sparse embeddings may not be available
    # with the current text-embedding-004 model
    try:
        dense_results = qdrant.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=models.NamedVector(name="dense", vector=dense_vector),
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        )
        
        # Convert results to the expected format
        combined_results = []
        for result in dense_results:
            combined_results.append({
                "payload": result.payload, 
                "score": result.score
            })
        
        return combined_results
        
    except Exception as e:
        # Fallback to default vector search if named vectors don't exist yet
        print(f"Named vector search failed, trying default vector: {e}")
        try:
            dense_results = qdrant.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=dense_vector,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True,
            )
            
            combined_results = []
            for result in dense_results:
                combined_results.append({
                    "payload": result.payload, 
                    "score": result.score
                })
            
            return combined_results
        except Exception as fallback_error:
            print(f"All search methods failed: {fallback_error}")
            return []
