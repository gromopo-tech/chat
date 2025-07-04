import json
from typing import Dict, Any
from app.models import llm
from datetime import datetime, timezone


current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PROMPT_TEMPLATE = """
Today's date is: {current_date}.
You are a query parser for a review retrieval system. Given a user query, extract:
- query_embedding_text: The main text to embed for semantic search.
- filter: Structured filters for rating, languageCode, and publishTime (ISO8601).
- intent: One of "summarize_reviews", "list_pros", "list_cons", "general_question".

IMPORTANT: For publishTime filters, use ISO8601 format (e.g., "2025-01-01T00:00:00Z" for January 1, 2025).

Return a JSON object matching this TypeScript type:
type ParsedQuery = {{
  query_embedding_text: string;
  filter?: {{
    rating?: {{ $in?: number[]; $gte?: number; $lte?: number }};
    languageCode?: string;
    publishTime?: {{ $gte?: string }};
  }};
  intent: "summarize_reviews" | "list_pros" | "list_cons" | "general_question";
}}

User query: "{user_query}"

Respond with ONLY the JSON object, no additional text or explanation.
"""


def parse_query_with_llm(user_query: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(user_query=user_query, current_date=current_date)
    response = llm.invoke(prompt)
    print(f"[DEBUG] Response type: {type(response)}")  # Add this for debugging
    print(f"[DEBUG] Response content: {response}")  # Add this for debugging
    # Extract content from AIMessage object
    content = response.content if hasattr(response, "content") else str(response)

    # Clean the response - remove markdown code block markers if present
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] Failed to parse JSON. Raw content: {content}")
        raise ValueError(f"Failed to parse LLM output: {content}") from e
