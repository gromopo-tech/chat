from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    human: str
    ai: str

class QueryRequest(BaseModel):
    query: str
    business_id: str | None = None
    session_id: str | None = None
    chat_history: list[ChatMessage] | None = []

class QueryResponse(BaseModel):
    answer: str
    context: list[str]
    parsed_filter: dict[str, Any] | None = None

# Simple in-memory storage (use Redis/DB in production)
last_contexts: dict[str, list[str]] = {}
last_filters: dict[str, dict] = {}