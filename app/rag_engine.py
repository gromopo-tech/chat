from app.vertex_client import get_vertex_ai_client, get_rag_engine_endpoint
from app.models import RagQueryRequest
from app.query_parser import parse_query_with_llm

def get_rag_response(user_query: str):
    """
    Calls the Vertex AI RAG Engine with the parsed user query and returns the answer and context.
    """
    parsed = parse_query_with_llm(user_query)
    embedding_text = parsed["query_embedding_text"]

    client = get_vertex_ai_client()
    endpoint = get_rag_engine_endpoint()

    # Build the request for the RAG Engine (see Vertex AI RAG Engine quickstart)
    request = {
        "rag_engine": endpoint,
        "queries": [
            {
                "query": embedding_text,
                # Optionally add filters/metadata if supported by your RAG Engine config
            }
        ]
    }

    response = client.retrieve_and_generate(request=request)
    result = response.responses[0]

    return {
        "answer": result.candidates[0].content if result.candidates else "No answer found.",
        "context": [doc.content for doc in result.candidates[0].citations] if result.candidates and result.candidates[0].citations else [],
        "intent": parsed.get("intent"),
        "parsed_filter": parsed.get("filter"),
    }
