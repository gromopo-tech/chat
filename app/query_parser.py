import json
from typing import Dict, Any
from app.models import llm
from datetime import datetime, timezone


current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PROMPT_TEMPLATE = """
Today's date is: {current_date}.
You are a query parser for a review retrieval system for a single business called \"Duck and Decanter\" in Phoenix, AZ. 
The main user is the business owner, who wants to analyze and understand customer reviews to improve and grow their business. 

All reviews are in the following format:
- comment: The main review text (string)
- starRating: The rating as a string (\"ONE\", \"TWO\", \"THREE\", \"FOUR\", \"FIVE\")
- createTime: The publish time (ISO8601 string)
- reviewer.displayName: The reviewer's name (string)

Given a user query, extract:
- query_embedding_text: The main text to embed for semantic search.
- filter: Structured filters for rating (as integer 1-5, mapped from starRating), and createTime (ISO8601).
- intent: One of \"summarize_reviews\", \"list_pros\", \"list_cons\", \"general_question\".

IMPORTANT: 
- For createTime filters, use ISO8601 format (e.g., \"2025-01-01T00:00:00Z\" for January 1, 2025).
- For queries about complaints, negative feedback, or cons (intent \"list_cons\"), set the rating filter to include 1, 2, and 3 star reviews.
- For rating, always use integer values 1-5 (1=ONE, 2=TWO, 3=THREE, 4=FOUR, 5=FIVE).

Return a JSON object matching this TypeScript type:
type ParsedQuery = {{
  query_embedding_text: string;
  filter?: {{
    rating?: {{ $in?: number[]; $gte?: number; $lte?: number }};
    createTime?: {{ $gte?: string }};
  }};
  intent: \"summarize_reviews\" | \"list_pros\" | \"list_cons\" | \"general_question\";
}}

User query: "{user_query}"

Respond with ONLY the JSON object, no additional text or explanation.
"""


def parse_query_with_llm(user_query: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(user_query=user_query, current_date=current_date)
    response = llm.invoke(prompt)
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
        raise ValueError(f"Failed to parse LLM output: {content}") from e
