import google.genai as genai
from app.models import RagQueryRequest
from app.query_parser import parse_query_with_llm
import os

def get_rag_response(user_query: str):
    """
    Calls the Gemini model using the Google Gen AI SDK and returns the answer and context.
    """
    parsed = parse_query_with_llm(user_query)
    embedding_text = parsed["query_embedding_text"]

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("")  # or your preferred model
    response = model.generate_content(embedding_text)

    # The response object structure may differ; adapt as needed
    answer = response.text if hasattr(response, "text") else str(response)
    # If citations/contexts are available, extract them; else, return empty list
    context = getattr(response, "citations", []) if hasattr(response, "citations") else []

    return {
        "answer": answer,
        "context": context,
        "intent": parsed.get("intent"),
        "parsed_filter": parsed.get("filter"),
    }
