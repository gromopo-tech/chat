import json

import structlog
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.chains import get_streaming_rag_response
from app.config import Config
from app.data_models import IngestGoogleTakeoutRequest, IngestResponse, QueryRequest, QueryResponse
from app.ingestion.google_takeout import GoogleTakeoutSource
from app.ingestion.pipeline import embed_and_upsert
from app.logging_config import setup_logging
from app.vectorstore import get_qdrant

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

            async for chunk in get_streaming_rag_response(request.query, business_id=request.business_id, business_name=request.business_name or "this restaurant", chat_history=request.chat_history):
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

    async for chunk in get_streaming_rag_response(request.query, business_id=request.business_id, business_name=request.business_name or "this restaurant", chat_history=request.chat_history):
        if "metadata" in chunk:
            context = chunk["metadata"].get("context", [])
            parsed_filter = chunk["metadata"].get("parsed_filter")
        if "chunk" in chunk:
            full_answer += chunk["chunk"]
        if "answer" in chunk and chunk["answer"]:
            full_answer = chunk["answer"]

    return QueryResponse(answer=full_answer, context=context, parsed_filter=parsed_filter)


@app.post("/ingest/google_takeout", response_model=IngestResponse)
def ingest_google_takeout(
    req: IngestGoogleTakeoutRequest,
    authorization: str = Header(...),
):
    """Ingest Google Business Profile / Takeout reviews into Qdrant for a given tenant.

    Accepts the raw review array parsed client-side from a Google Takeout JSON export.
    Runs synchronously — fine for up to ~200 reviews; production would queue via Cloud Tasks.

    Auth: shared secret passed as 'Authorization: Bearer <INGEST_SHARED_SECRET>'.
    """
    expected = Config.INGEST_SHARED_SECRET
    if not expected or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid ingest credentials")

    records = GoogleTakeoutSource.parse_records(req.business_id, req.reviews)
    result = embed_and_upsert(records)

    log.info(
        "ingest_complete",
        business_id=req.business_id,
        ingested=result.ingested,
        skipped=result.skipped,
        errors=len(result.errors),
    )
    return IngestResponse(
        ingested=result.ingested,
        skipped=result.skipped,
        errors=result.errors,
    )


@app.get("/health")
def health():
    """Liveness + Qdrant reachability check. Returns 503 if Qdrant is unreachable."""
    try:
        get_qdrant().get_collections()
        return {"status": "ok", "qdrant": "ok"}
    except Exception as e:
        log.error("health_check_qdrant_failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "qdrant": "unreachable"},
        )


@app.get("/")
def homepage():
    return {"title": "gromopo - review based rag llm"}
