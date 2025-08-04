from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from app.vectorstore import hybrid_search
from qdrant_client import models


class HybridRetriever(BaseRetriever):
    """Custom retriever that performs hybrid search using both dense and sparse vectors."""
    
    filter: Optional[models.Filter] = None
    k: int = 20
    
    class Config:
        arbitrary_types_allowed = True  # Allow Qdrant Filter type
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        results = hybrid_search(query, self.filter, self.k)
        
        documents = []
        for result in results:
            payload = result["payload"]
            doc = Document(
                page_content=payload.get("text", ""),
                metadata={
                    "rating": payload.get("rating"),
                    "createTime": payload.get("createTime"),
                    "author": payload.get("author"),
                    "review_id": payload.get("review_id"),
                    "score": result["score"]
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async version of _get_relevant_documents."""
        return self._get_relevant_documents(query, run_manager=run_manager)


def create_hybrid_retriever(qdrant_filter: models.Filter = None, k: int = 20) -> HybridRetriever:
    """Create a hybrid retriever with the specified filter and k value."""
    return HybridRetriever(filter=qdrant_filter, k=k)
