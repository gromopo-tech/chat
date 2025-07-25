from langchain_core.runnables import RunnableMap
from app.prompts import RESPONSE_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from app.config import Config
from app.models import embeddings_model, llm
from app.vectorstore import get_qdrant, build_qdrant_filter
from app.query_parser import parse_query_with_llm


vectorstore = QdrantVectorStore(
    client=get_qdrant(),
    collection_name=Config.COLLECTION_NAME,
    embedding=embeddings_model,
    content_payload_key="text",
)


def get_rag_response(user_query: str):
    parsed = parse_query_with_llm(user_query)
    filter_dict = parsed.get("filter")
    qdrant_filter = build_qdrant_filter(filter_dict)
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": qdrant_filter, "k": 20}
    )

    intent = parsed.get("intent", "summarize reviews")
    embedding_text = parsed["query_embedding_text"]

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
        RunnableMap(
            {
                "context": lambda _: context,
                "intent": lambda _: intent,
                "criteria": lambda _: filter_dict,
                "question": lambda x: x["question"],
            }
        )
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
