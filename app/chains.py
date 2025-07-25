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

    intent = parsed.get("intent", "summarize reviews")

    # Prompt template for RAG (can be customized)
    specific_prompt = PromptTemplate.from_template(
        """
        {intent} based on the following reviews:
        {context}
        Question: {question}
        The reviews have already been filtered to match the user's criteria:
        {filter_dict}
        """
    )

    general_prompt = PromptTemplate.from_template(
        """
        Answer the general question based on the following reviews, which have already been filtered to match the user's criteria:
        {context}
        Question: {question}
        The reviews have already been filtered to match the user's criteria:
        {filter_dict}
        """
    )

    embedding_text = parsed["query_embedding_text"]

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

    rag_chain = (
        RunnableMap({
            "context": lambda _: context,
            "intent": lambda _: intent,
            "question": lambda x: x["question"],
            "filter_dict": lambda _: filter_dict,
        })
        | (general_prompt if intent == "general question" else specific_prompt)
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
