import asyncio
from typing import Any

import structlog
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Direction,
    Distance,
    OrderBy,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.config import Config
from app.utils import iso8601_to_timestamp
from app.vertexai_models import get_query_embeddings

log = structlog.get_logger(__name__)


def get_qdrant():
    qdrant_host = Config.QDRANT_HOST
    return QdrantClient(host=qdrant_host, prefer_grpc=True)


def ensure_collection(qdrant: QdrantClient) -> None:
    """Create the Qdrant collection and required payload indexes if they don't exist.

    Safe to call repeatedly — collection creation is skipped if it already exists,
    and payload index creation is idempotent.
    """
    if not qdrant.collection_exists(collection_name=Config.COLLECTION_NAME):
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
    # Float index required for order_by on createTime (recency queries).
    # Idempotent — safe to call on existing collections.
    qdrant.create_payload_index(
        collection_name=Config.COLLECTION_NAME,
        field_name="createTime",
        field_schema=models.PayloadSchemaType.FLOAT,
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
        response = qdrant.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
        )
        return [{"payload": r.payload, "score": r.score} for r in response.points]

    except Exception as e:
        log.warning("named_vector_search_failed", error=str(e))
        try:
            response = qdrant.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=dense_vector,
                query_filter=qdrant_filter,
                limit=k,
                with_payload=True,
            )
            return [{"payload": r.payload, "score": r.score} for r in response.points]
        except Exception as fallback_error:
            log.error("all_search_methods_failed", error=str(fallback_error))
            return []


def recency_search(qdrant_filter: models.Filter | None = None, k: int = 5) -> list[dict[str, Any]]:
    """Return the k most recent reviews sorted by createTime descending.

    Uses scroll (no vector scoring) so the result is purely time-ordered,
    not influenced by semantic similarity to the query.
    """
    qdrant = get_qdrant()
    points, _ = qdrant.scroll(
        collection_name=Config.COLLECTION_NAME,
        scroll_filter=qdrant_filter,
        limit=k,
        order_by=OrderBy(key="createTime", direction=Direction.DESC),
        with_payload=True,
    )
    return [{"payload": p.payload, "score": None} for p in points]


class _DenseRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by Qdrant dense-vector search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    qdrant_filter: models.Filter | None = None
    k: int = 20
    sort_by_recency: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.sort_by_recency:
            results = recency_search(self.qdrant_filter, self.k)
        else:
            results = hybrid_search(query, self.qdrant_filter, self.k)
        return [
            Document(page_content=r["payload"]["text"], metadata=r["payload"])
            for r in results
        ]

    async def _aget_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        if self.sort_by_recency:
            results = await asyncio.to_thread(recency_search, self.qdrant_filter, self.k)
        else:
            results = await asyncio.to_thread(hybrid_search, query, self.qdrant_filter, self.k)
        return [
            Document(page_content=r["payload"]["text"], metadata=r["payload"])
            for r in results
        ]


def create_dense_retriever(
    qdrant_filter: models.Filter | None = None,
    k: int = 20,
    sort_by_recency: bool = False,
) -> _DenseRetriever:
    """Return a LangChain retriever using dense Qdrant search."""
    return _DenseRetriever(qdrant_filter=qdrant_filter, k=k, sort_by_recency=sort_by_recency)
