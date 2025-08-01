from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    human: str
    ai: str

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []

class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    parsed_filter: Optional[Dict[str, Any]] = None

# Simple in-memory storage (use Redis/DB in production)
last_contexts: Dict[str, List[str]] = {}
last_filters: Dict[str, Dict] = {}