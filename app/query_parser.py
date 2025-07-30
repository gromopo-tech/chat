import json
from typing import Dict, Any
from app.models import query_parser_llm
from app.prompts import QUERY_PARSER_PROMPT
from datetime import datetime, timezone


def parse_query_with_llm(user_query: str) -> Dict[str, Any]:
    # to simulate streaming reviews we use the day after the most recent review (2025-05-25)
    current_date = datetime(2025, 5, 25).strftime("%Y-%m-%d")
    prompt = QUERY_PARSER_PROMPT.format(
        user_query=user_query, current_date=current_date
    )
    response = query_parser_llm.invoke(prompt)
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
