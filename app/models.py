from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class RagQueryRequest(BaseModel):
    query: str
    filter: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None

class RagQueryResponse(BaseModel):
    answer: str
    context: List[str]
    intent: Optional[str] = None
    parsed_filter: Optional[Dict[str, Any]] = None
