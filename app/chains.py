from langchain.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.models import embeddings_model, llm
from app.vectorstore import get_qdrant, COLLECTION_NAME


def get_rag_response(query: str, place_id: str):
    vectorstore = QdrantVectorStore(
        client=get_qdrant(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model,
        content_payload_key="text",
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": Filter(
                must=[FieldCondition(key="place_id", match=MatchValue(value=place_id))]
            )  # TODO: add "k": 5 to the search_kwargs and experiment with different values
        }
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    result = qa.invoke({"query": query})

    return {
        "answer": result["result"],
        "context": [doc.page_content for doc in result["source_documents"]],
    }
