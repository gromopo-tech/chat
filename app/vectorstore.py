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
    sparse_vector = query_embeddings['sparse']
    
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


def _combine_search_results(search_results: List[List], k: int) -> List[Dict[str, Any]]:
    """Combine dense and sparse search results using weighted scoring."""
    dense_results = search_results[0] if len(search_results) > 0 else []
    sparse_results = search_results[1] if len(search_results) > 1 else []
    
    # Create a mapping of document ID to combined score
    combined_scores = {}
    
    # Process dense results
    for result in dense_results:
        doc_id = result.id
        combined_scores[doc_id] = {
            "payload": result.payload,
            "dense_score": result.score,
            "sparse_score": 0.0,
            "combined_score": result.score * Config.DENSE_WEIGHT
        }
    
    # Process sparse results
    for result in sparse_results:
        doc_id = result.id
        if doc_id in combined_scores:
            # Update existing entry
            combined_scores[doc_id]["sparse_score"] = result.score
            combined_scores[doc_id]["combined_score"] = (
                combined_scores[doc_id]["dense_score"] * Config.DENSE_WEIGHT +
                result.score * Config.SPARSE_WEIGHT
            )
        else:
            # New entry from sparse only
            combined_scores[doc_id] = {
                "payload": result.payload,
                "dense_score": 0.0,
                "sparse_score": result.score,
                "combined_score": result.score * Config.SPARSE_WEIGHT
            }
    
    # Sort by combined score and return top k
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )
    
    return [{"payload": result["payload"], "score": result["combined_score"]} for result in sorted_results[:k]]
