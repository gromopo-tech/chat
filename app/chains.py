from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from app.config import Config
from app.models import embeddings_model, llm
from app.vectorstore import get_qdrant, build_qdrant_filter
from app.query_parser import parse_query_with_llm
from time import time


vectorstore = QdrantVectorStore(
    client=get_qdrant(),
    collection_name=Config.COLLECTION_NAME,
    embedding=embeddings_model,
    content_payload_key="text",
)


def get_rag_response(user_query: str):
    # Parse the user query using LLM for intent and filters
    start = time()
    parsed = parse_query_with_llm(user_query)
    print(f"Query parsing time: {time() - start:.2f} seconds")
    filter_dict = parsed.get("filter")

    # Build Qdrant filter for retrieval
    qdrant_filter = build_qdrant_filter(filter_dict)

    # Set up the retriever with filter and top-k
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": qdrant_filter, "k": 20}
    )

    # Prompt template for RAG (can be customized)
    prompt = PromptTemplate.from_template(
        """
        Answer the question based on the following context:
        {context}
        Question: {question}
        """
    )

    embedding_text = parsed["query_embedding_text"]
    intent = parsed.get("intent", "summarize reviews")

    # Retrieve context ONCE
    start = time()
    context_docs = retriever.invoke(embedding_text)
    print(f"Retrieval invocation time: {time() - start:.2f} seconds")
    context = [doc.page_content for doc in context_docs]
    if not context or all(not c.strip() for c in context):
        return {
            "answer": "There are no reviews matching your query.",
            "context": [],
            "intent": intent,
            "parsed_filter": filter_dict,
        }

    # Compose the RAG chain using Runnable, but pass context directly
    def context_lambda(x):
        return context

    rag_chain = (
        RunnableMap({
            "context": context_lambda,
            "question": lambda x: x["question"],
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time()
    answer = rag_chain.invoke({"question": embedding_text})
    print(f"RAG chain invocation time: {time() - start:.2f} seconds")
    return {
        "answer": answer,
        "context": context,
        "intent": intent,
        "parsed_filter": filter_dict,
    }
