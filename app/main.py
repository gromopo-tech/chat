from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from app.chains import get_rag_response, get_streaming_rag_response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import json


# -------- FastAPI app --------
app = FastAPI()


# -------- Data Models --------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    intent: Optional[str] = None
    parsed_filter: Optional[Dict[str, Any]] = None


# -------- Routes --------
@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    try:
        result = get_rag_response(request.query)
        return result
    except Exception as e:
        return {
            "answer": "Error",
            "context": [str(e)],
            "intent": None,
            "parsed_filter": None,
        }


@app.post("/rag/streaming-query")
async def rag_streaming_query(request: QueryRequest):
    """Streaming endpoint that returns chunks of the answer as they're generated."""
    async def generate():
        try:
            # Send start of stream
            yield f"data: {json.dumps({'status': 'start'})}\n\n"
            
            full_answer = ""  # Collect the full answer
            
            async for chunk in get_streaming_rag_response(request.query):
                # If this contains metadata, send it separately
                if "metadata" in chunk:
                    metadata = chunk["metadata"]
                    yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
                    continue
                    
                # If this is a chunk of the answer
                if "chunk" in chunk:
                    text = chunk["chunk"]
                    full_answer += text
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
                    time.sleep(.1)
                
                # If we have a complete answer in one go
                if "answer" in chunk and chunk["answer"]:
                    full_answer = chunk["answer"]
                    yield f"data: {json.dumps({'type': 'answer', 'text': chunk['answer']})}\n\n"
                
                # If we're done
                if chunk.get("done", False):
                    yield f"data: {json.dumps({'type': 'end', 'text': full_answer})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/")
def homepage():
    return {"title": "gromopo - review based rag llm"}
