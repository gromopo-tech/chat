import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langchain_core.runnables import RunnableMap

from app.prompts import RESPONSE_PROMPT
from app.query_parser import parse_query_with_llm
from app.vectorstore import build_qdrant_filter, create_dense_retriever
from app.vertexai_models import default_llm

log = structlog.get_logger(__name__)


def _get_k_value_for_query(user_query: str) -> int:
    """Determine optimal k value based on query type."""
    query_lower = user_query.lower()
    
    # For analytical queries that need comprehensive data - use high k
    if any(phrase in query_lower for phrase in [
        "most common", "most frequent", "trends", "patterns", 
        "all complaints", "all praise", "summary", "analyze",
        "what are the", "summarize", "overview", "how many"
    ]):
        return 1000  # Get most/all relevant reviews for comprehensive analysis
    
    # For comparison queries
    if any(phrase in query_lower for phrase in [
        "compare", "versus", "vs", "difference", "better", "worse"
    ]):
        return 100
    
    # For specific examples or particular issues
    if any(phrase in query_lower for phrase in [
        "example", "instance", "specific", "particular", "tell me about"
    ]):
        return 30  # Lower k for specific examples
    
    # Default for general queries
    return 50

def _prepare_query(user_query: str, business_id: str | None = None) -> tuple[dict, str, object]:
    """Common query preparation logic for both streaming and non-streaming RAG responses."""
    t0 = time.monotonic()
    parsed = parse_query_with_llm(user_query)
    filter_dict = parsed.get("filter")
    qdrant_filter = build_qdrant_filter(filter_dict, business_id=business_id)
    
    # Dynamic k based on query type
    k_value = _get_k_value_for_query(user_query)
    log.info("query_prepared", k=k_value, business_id=business_id, filter=filter_dict,
             parse_latency_ms=round((time.monotonic() - t0) * 1000, 1))
    
    # Create dense retriever
    retriever = create_dense_retriever(qdrant_filter=qdrant_filter, k=k_value)

    embedding_text = parsed["query_embedding_text"]
    
    return filter_dict, embedding_text, retriever

def _rag_runnable(context: list[str], filter_dict: dict, review_count: int = None) -> RunnableMap:
    return RunnableMap(
        {
            "context": lambda _: "\n\n".join(context),
            "criteria": lambda _: filter_dict,
            "review_count": lambda _: review_count,
            "question": lambda x: x["question"],
        }
    )

async def get_streaming_rag_response(user_query: str, business_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
    """Streams tokens as they're generated."""
    filter_dict, embedding_text, retriever = _prepare_query(user_query, business_id=business_id)
    parsed = parse_query_with_llm(user_query)
    if parsed.get("off_topic", False):
        yield {
            "answer": "Sorry, I can't assist you yet with that. Currently I'm only able to help you "
            "understand customer feedback for Duck and Decanter and improve business based on customer feedback. "
            "Please ask me about customer reviews, complaints, praise, business insights, or suggestions for improvements.",
            "context": [],
            "parsed_filter": None,
            "done": True
        }
        return
    t0 = time.monotonic()
    context_docs = await retriever.ainvoke(embedding_text)
    retrieval_ms = round((time.monotonic() - t0) * 1000, 1)
    context = [doc.page_content for doc in context_docs]
    review_count = len(context)
    log.info("retrieval", k=review_count, business_id=business_id, latency_ms=retrieval_ms)
    if not context or all(not c.strip() for c in context):
        yield {
            "answer": "There are no reviews matching your query.",
            "context": [],
            "parsed_filter": filter_dict,
            "done": True,  # Indicate this is the complete response
        }
        return

    # First, yield the metadata with empty answer
    yield {
        "metadata": {
            "context": context,
            "parsed_filter": filter_dict,
        }
    }

    # Now stream the answer in chunks
    llm_t0 = time.monotonic()
    streaming_rag_chain = (
        _rag_runnable(context, filter_dict, review_count)
        | RESPONSE_PROMPT
        | default_llm
    )

    buffer = ""
    async for chunk in streaming_rag_chain.astream({"question": user_query}):
        if hasattr(chunk, "content"):
            token = chunk.content
        else:
            token = str(chunk)
            
        buffer += token
        
        # Send chunks of approximately 5 characters or a single word
        while " " in buffer:
            if " " in buffer:
                # Split at the first space to send a complete word
                word, buffer = buffer.split(" ", 1)
                word += " "  # Add the space back
                yield {"chunk": word}
            else:
                # If no space but buffer is long enough, send the first 5 characters
                yield {"chunk": buffer[:5]}
                buffer = buffer[5:]

    # Send any remaining text
    if buffer:
        yield {"chunk": buffer}

    log.info("llm_call", model=default_llm.model_name,
             latency_ms=round((time.monotonic() - llm_t0) * 1000, 1))

    # Send final message indicating completion
    yield {"done": True}
