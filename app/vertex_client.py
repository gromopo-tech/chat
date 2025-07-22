import os
from google.cloud import aiplatform

def get_vertex_ai_client():
    """Returns a Vertex AI PredictionServiceClient instance."""
    return aiplatform.gapic.PredictionServiceClient()

def get_rag_engine_endpoint():
    """Builds the full resource name for the RAG Engine endpoint."""
    project = os.getenv("VERTEX_PROJECT", os.getenv("PROJECT_ID"))
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    rag_engine_id = os.getenv("RAG_ENGINE_ID")
    return f"projects/{project}/locations/{location}/ragEngines/{rag_engine_id}"
