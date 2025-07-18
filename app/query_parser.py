import json
from typing import Dict, Any
from app.models import llm
from datetime import datetime, timezone


current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
PROMPT_TEMPLATE = """
You are a query parser for customer reviews of Duck and Decanter in Phoenix, AZ. The user is the business owner seeking insights to improve their business.

Reviews have these fields:
- comment: review text (string)
- starRating: "ONE" to "FIVE"
- createTime: ISO8601 string
- reviewer.displayName: name (string)

Given a user query, extract:
- query_embedding_text: main text for semantic search
- filter: rating (integer 1-5, mapped from starRating), createTime (ISO8601)
- intent: one of "summarize_reviews", "list_pros", "list_cons", "general_question"

Instructions:
- For complaints/cons (intent "list_cons"), set rating filter to [1, 2].
- For rating, always use integers 1-5 ("ONE"=1, ..., "FIVE"=5).
- For createTime, use ISO8601 format.

Return ONLY a JSON object matching:
type ParsedQuery = {{
  query_embedding_text: string;
  filter?: {{
    rating?: {{ $in?: number[]; $gte?: number; $lte?: number }};
    createTime?: {{ $gte?: string }};
  }};
  intent: "summarize_reviews" | "list_pros" | "list_cons" | "general_question";
}}

User query: "{user_query}"
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
