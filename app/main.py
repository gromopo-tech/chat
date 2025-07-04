from fastapi import FastAPI
from app.chains import get_rag_response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# -------- FastAPI app --------
app = FastAPI()


# -------- Data Models --------
class QueryRequest(BaseModel):
    query: str
    place_id: str


class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    intent: Optional[str] = None
    parsed_filter: Optional[Dict[str, Any]] = None


# -------- Routes --------
@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    try:
        result = get_rag_response(request.query, request.place_id)
        return result
    except Exception as e:
        return {"answer": "Error", "context": [str(e)], "intent": None, "parsed_filter": None}
