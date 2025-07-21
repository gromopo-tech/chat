from langchain.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from app.models import embeddings_model, llm
from app.vectorstore import get_qdrant, COLLECTION_NAME, build_qdrant_filter
from app.query_parser import parse_query_with_llm


vectorstore = QdrantVectorStore(
    client=get_qdrant(),
    collection_name=COLLECTION_NAME,
    embedding=embeddings_model,
    content_payload_key="text",
)


def get_rag_response(user_query: str):
    parsed = parse_query_with_llm(user_query)
    embedding_text = parsed["query_embedding_text"]
    filter_dict = parsed.get("filter")
    intent = parsed.get("intent", "summarize reviews")

    qdrant_filter = build_qdrant_filter(filter_dict)

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": qdrant_filter, "k": 10}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    result = qa.invoke({"query": embedding_text})
    context = [doc.page_content for doc in result["source_documents"]]

    if not context or all(not c.strip() for c in context):
        return {
            "answer": "There are no reviews matching your query.",
            "context": [],
            "intent": intent,
            "parsed_filter": filter_dict,
        }

    return {
        "answer": result["result"],
        "context": context,
        "intent": intent,
        "parsed_filter": filter_dict,
    }
