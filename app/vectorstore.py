import asyncio
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

from app.config import Config
from app.utils import iso8601_to_timestamp
from app.vertexai_models import get_query_embeddings


def get_qdrant():
    qdrant_host = Config.QDRANT_HOST
    return QdrantClient(host=qdrant_host, prefer_grpc=True)


def ensure_collection(qdrant: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist.

    Unlike embed_reviews.py (which deletes and recreates), this is safe
    to call repeatedly — it is a no-op when the collection already exists.
    """
    if qdrant.collection_exists(collection_name=Config.COLLECTION_NAME):
        return
    qdrant.create_collection(
        collection_name=Config.COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=Config.DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=16),
    )


def build_qdrant_filter(
    parsed_filter: dict,
    business_id: str | None = None,
) -> models.Filter:
    """Convert parsed filter + optional tenant filter to Qdrant models.Filter.

    When business_id is provided it is always injected as a must-match condition
    so that retrieval is strictly scoped to that tenant's data.
    """
    must = []
    if business_id:
        must.append(
            models.FieldCondition(
                key="business_id", match=models.MatchValue(value=business_id)
            )
        )
    if not parsed_filter:
        return models.Filter(must=must) if must else None
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


def build_qdrant_filter_with_business_id(
    parsed_filter: dict,
    business_id: str | None = None,
) -> models.Filter:
    """Alias kept for import clarity in chains.py."""
    return build_qdrant_filter(parsed_filter, business_id=business_id)


def hybrid_search(query_text: str, qdrant_filter: models.Filter = None, k: int = 20) -> list[dict[str, Any]]:
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


class _DenseRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by Qdrant dense-vector search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    qdrant_filter: models.Filter | None = None
    k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = hybrid_search(query, self.qdrant_filter, self.k)
        return [
            Document(page_content=r["payload"]["text"], metadata=r["payload"])
            for r in results
        ]

    async def _aget_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        results = await asyncio.to_thread(hybrid_search, query, self.qdrant_filter, self.k)
        return [
            Document(page_content=r["payload"]["text"], metadata=r["payload"])
            for r in results
        ]


def create_dense_retriever(qdrant_filter: models.Filter | None = None, k: int = 20) -> _DenseRetriever:
    """Return a LangChain retriever using dense Qdrant search."""
    return _DenseRetriever(qdrant_filter=qdrant_filter, k=k)
