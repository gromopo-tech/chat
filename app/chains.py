from langchain_core.runnables import RunnableMap
from app.prompts import RESPONSE_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from app.config import Config
from app.models import embeddings_model, llm
from app.vectorstore import get_qdrant, build_qdrant_filter
from app.query_parser import parse_query_with_llm
from typing import AsyncIterator, Dict, Any, Tuple, List, Optional


vectorstore = QdrantVectorStore(
    client=get_qdrant(),
    collection_name=Config.COLLECTION_NAME,
    embedding=embeddings_model,
    content_payload_key="text",
)


def _prepare_query(user_query: str) -> Tuple[Dict, str, object]:
    """Common query preparation logic for both streaming and non-streaming RAG responses."""
    parsed = parse_query_with_llm(user_query)
    filter_dict = parsed.get("filter")
    qdrant_filter = build_qdrant_filter(filter_dict)
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": qdrant_filter, "k": 20}
    )

    intent = parsed.get("intent", "summarize reviews")
    embedding_text = parsed["query_embedding_text"]
    
    return filter_dict, intent, embedding_text, retriever

def _rag_runnable(context: List[str], intent: str, filter_dict: Dict) -> RunnableMap:
    return RunnableMap(
        {
            "context": lambda _: "\n\n".join(context),
            "intent": lambda _: intent,
            "criteria": lambda _: filter_dict,
            "question": lambda x: x["question"],
            }
        )

def get_rag_response(user_query: str):
    filter_dict, intent, embedding_text, retriever = _prepare_query(user_query)

    context_docs = retriever.invoke(embedding_text)
    context = [doc.page_content for doc in context_docs]
    if not context or all(not c.strip() for c in context):
        return {
            "answer": "There are no reviews matching your query.",
            "context": [],
            "intent": intent,
            "parsed_filter": filter_dict,
        }

    rag_chain = (
        _rag_runnable(context, intent, filter_dict)
        | RESPONSE_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({"question": embedding_text})
    return {
        "answer": answer,
        "context": context,
        "intent": intent,
        "parsed_filter": filter_dict,
    }


async def get_streaming_rag_response(user_query: str) -> AsyncIterator[Dict[str, Any]]:
    """Streaming version of get_rag_response that yields tokens as they're generated."""
    filter_dict, intent, embedding_text, retriever = _prepare_query(user_query)

    context_docs = await retriever.ainvoke(embedding_text)
    context = [doc.page_content for doc in context_docs]
    if not context or all(not c.strip() for c in context):
        yield {
            "answer": "There are no reviews matching your query.",
            "context": [],
            "intent": intent,
            "parsed_filter": filter_dict,
            "done": True,  # Indicate this is the complete response
        }
        return

    # First, yield the metadata with empty answer
    yield {
        "metadata": {
            "context": context,
            "intent": intent,
            "parsed_filter": filter_dict,
        }
    }

    # Now stream the answer in chunks
    streaming_rag_chain = (
        _rag_runnable(context, intent, filter_dict)
        | RESPONSE_PROMPT
        | llm
    )

    buffer = ""
    async for chunk in streaming_rag_chain.astream({"question": embedding_text}):
        if hasattr(chunk, "content"):
            token = chunk.content
        else:
            token = str(chunk)

        buffer += token

        # Send larger, more readable chunks
        if len(buffer) >= 5 or any(p in buffer for p in [".", "!", "?", "\n"]):
            yield {"chunk": buffer}
            buffer = ""

    # Send any remaining text
    if buffer:
        yield {"chunk": buffer}

    # Send final message indicating completion
    yield {"done": True}
