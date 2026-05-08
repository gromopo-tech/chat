import json

import structlog
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.chains import get_streaming_rag_response
from app.data_models import QueryRequest, QueryResponse
from app.logging_config import setup_logging

setup_logging()
log = structlog.get_logger(__name__)

# -------- FastAPI app --------
app = FastAPI()


# -------- Routes --------
@app.post("/rag/streaming-query")
async def rag_streaming_query(request: QueryRequest):
    """Streaming SSE endpoint — returns answer tokens as they are generated."""
    async def generate():
        try:
            yield f"data: {json.dumps({'status': 'start'})}\n\n"
            full_answer = ""

            async for chunk in get_streaming_rag_response(request.query, business_id=request.business_id):
                if "metadata" in chunk:
                    yield f"data: {json.dumps({'type': 'metadata', 'data': chunk['metadata']})}\n\n"
                    continue

                if "chunk" in chunk:
                    text = chunk["chunk"]
                    full_answer += text
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

                if "answer" in chunk and chunk["answer"]:
                    full_answer = chunk["answer"]
                    yield f"data: {json.dumps({'type': 'answer', 'text': chunk['answer']})}\n\n"

                if chunk.get("done", False):
                    yield f"data: {json.dumps({'type': 'end', 'text': full_answer})}\n\n"

        except Exception as e:
            log.error("streaming_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Synchronous query endpoint — collects the full streamed answer before returning.

    Useful for evals, curl testing, and non-streaming clients.
    """
    full_answer = ""
    context: list[str] = []
    parsed_filter = None

    async for chunk in get_streaming_rag_response(request.query, business_id=request.business_id):
        if "metadata" in chunk:
            context = chunk["metadata"].get("context", [])
            parsed_filter = chunk["metadata"].get("parsed_filter")
        if "chunk" in chunk:
            full_answer += chunk["chunk"]
        if "answer" in chunk and chunk["answer"]:
            full_answer = chunk["answer"]

    return QueryResponse(answer=full_answer, context=context, parsed_filter=parsed_filter)


@app.get("/")
def homepage():
    return {"title": "gromopo - review based rag llm"}
