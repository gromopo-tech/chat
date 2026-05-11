import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.runnables import RunnableMap

from app.data_models import ChatMessage
from app.prompts import NO_RESULTS_PROMPT, QUESTION_REWRITER_PROMPT, RESPONSE_PROMPT
from app.query_parser import parse_query_with_llm
from app.vectorstore import build_qdrant_filter, create_dense_retriever
from app.vertexai_models import default_llm, query_parser_llm

log = structlog.get_logger(__name__)


def _format_chat_history(chat_history: list[ChatMessage] | None) -> str:
    """Render chat history as plain Human/AI lines for prompt injection."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        lines.append(f"Human: {msg.human}")
        lines.append(f"AI: {msg.ai}")
    return "\n".join(lines)


def _rewrite_question(user_query: str, chat_history: list[ChatMessage] | None) -> str:
    """Rewrite a follow-up question into a standalone form using prior turns.

    Skips the LLM call entirely on the first turn (empty history) so single-turn
    queries pay no extra latency. Falls back to the original query on any failure
    so memory issues never block the main retrieval path.
    """
    if not chat_history:
        return user_query
    t0 = time.monotonic()
    try:
        prompt = QUESTION_REWRITER_PROMPT.format(
            chat_history=_format_chat_history(chat_history),
            question=user_query,
        )
        response = query_parser_llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        rewritten = content.strip()
        log.info(
            "query_rewrite",
            original_preview=user_query[:80],
            rewritten_preview=rewritten[:80],
            history_turns=len(chat_history),
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return rewritten or user_query
    except Exception as e:
        log.warning("query_rewrite_failed", error=str(e), query=user_query[:80])
        return user_query


_RECENCY_PHRASES = (
    "most recent", "latest review", "newest review", "last review",
    "most recent review", "recent review", "latest feedback", "newest feedback",
)


def _is_recency_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _RECENCY_PHRASES)


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

def _prepare_query(user_query: str, business_id: str | None = None, business_name: str = "this restaurant") -> tuple[dict, str, object, dict]:
    """Parse the query, build the Qdrant filter, and return the retriever.

    Returns (filter_dict, embedding_text, retriever, parsed) so callers can read
    off_topic and other parsed fields without a second LLM call.
    """
    t0 = time.monotonic()
    try:
        parsed = parse_query_with_llm(user_query, business_name=business_name)
    except ValueError:
        log.warning("query_parse_failed", query=user_query[:80])
        parsed = {"off_topic": False, "query_embedding_text": user_query, "filter": {}}
    filter_dict = parsed.get("filter")
    qdrant_filter = build_qdrant_filter(filter_dict, business_id=business_id)
    sort_by_recency = _is_recency_query(user_query)
    k_value = 5 if sort_by_recency else _get_k_value_for_query(user_query)
    log.info("query_prepared", k=k_value, sort_by_recency=sort_by_recency,
             business_id=business_id, filter=filter_dict,
             parse_latency_ms=round((time.monotonic() - t0) * 1000, 1))
    retriever = create_dense_retriever(qdrant_filter=qdrant_filter, k=k_value, sort_by_recency=sort_by_recency)
    embedding_text = parsed["query_embedding_text"]
    return filter_dict, embedding_text, retriever, parsed

def _rag_runnable(
    context: list[str],
    review_count: int,
    business_name: str,
    chat_history_text: str,
) -> RunnableMap:
    return RunnableMap(
        {
            "context": lambda _: "\n\n".join(context),
            "review_count": lambda _: review_count,
            "business_name": lambda _: business_name,
            "chat_history": lambda _: chat_history_text or "(no prior turns)",
            "question": lambda x: x["question"],
        }
    )


async def _stream_llm_response(chain, inputs: dict) -> AsyncIterator[dict[str, Any]]:
    """Stream a chain's output as word-level chunks. Shared by success and no-results paths."""
    buffer = ""
    async for chunk in chain.astream(inputs):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        buffer += token
        while " " in buffer:
            word, buffer = buffer.split(" ", 1)
            yield {"chunk": word + " "}
    if buffer:
        yield {"chunk": buffer}


def _format_doc(doc) -> str:
    """Prepend ISO date to review text so the LLM can reason about recency."""
    ct = doc.metadata.get("createTime")
    if ct:
        date_str = datetime.fromtimestamp(ct, tz=UTC).strftime("%Y-%m-%d")
        return f"[{date_str}] {doc.page_content}"
    return doc.page_content


async def get_streaming_rag_response(
    user_query: str,
    business_id: str | None = None,
    business_name: str = "this restaurant",
    chat_history: list[ChatMessage] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Streams tokens as they're generated."""
    standalone_query = _rewrite_question(user_query, chat_history)
    chat_history_text = _format_chat_history(chat_history)

    filter_dict, embedding_text, retriever, parsed = _prepare_query(
        standalone_query, business_id=business_id, business_name=business_name
    )
    if parsed.get("off_topic", False):
        yield {
            "answer": f"Sorry, I can only help with questions about customer reviews for {business_name}. "
            "Please ask me about customer feedback, complaints, praise, sentiment, or suggestions for improvement.",
            "context": [],
            "parsed_filter": None,
            "done": True
        }
        return
    t0 = time.monotonic()
    context_docs = await retriever.ainvoke(embedding_text)
    retrieval_ms = round((time.monotonic() - t0) * 1000, 1)
    context = [_format_doc(doc) for doc in context_docs]
    review_count = len(context)
    log.info("retrieval", k=review_count, business_id=business_id, latency_ms=retrieval_ms)

    if not context or all(not c.strip() for c in context):
        # No matching reviews — stream a graceful, RAG-faithful explanation instead of a hardcoded string.
        yield {
            "metadata": {
                "context": [],
                "parsed_filter": filter_dict,
            }
        }
        llm_t0 = time.monotonic()
        no_results_chain = NO_RESULTS_PROMPT | default_llm
        async for out in _stream_llm_response(
            no_results_chain,
            {"business_name": business_name, "question": user_query},
        ):
            yield out
        log.info(
            "llm_call",
            model=default_llm.model_name,
            path="no_results",
            latency_ms=round((time.monotonic() - llm_t0) * 1000, 1),
        )
        yield {"done": True}
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
        _rag_runnable(context, review_count, business_name, chat_history_text)
        | RESPONSE_PROMPT
        | default_llm
    )
    async for out in _stream_llm_response(streaming_rag_chain, {"question": user_query}):
        yield out

    log.info("llm_call", model=default_llm.model_name,
             latency_ms=round((time.monotonic() - llm_t0) * 1000, 1))

    # Send final message indicating completion
    yield {"done": True}
