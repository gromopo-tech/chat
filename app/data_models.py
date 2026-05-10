from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    human: str
    ai: str

class QueryRequest(BaseModel):
    query: str
    business_id: str | None = None
    business_name: str | None = None
    session_id: str | None = None
    chat_history: list[ChatMessage] | None = []

class QueryResponse(BaseModel):
    answer: str
    context: list[str]
    parsed_filter: dict[str, Any] | None = None

class IngestGoogleTakeoutRequest(BaseModel):
    business_id: str
    reviews: list[dict]  # raw Google Takeout review entries from client-side JSON parse


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    errors: list[str]