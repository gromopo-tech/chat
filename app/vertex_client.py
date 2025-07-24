from config import Config
from google import genai

# Only run this block for Vertex AI API


def get_vertex_ai_client():
    """Returns a Vertex AI PredictionServiceClient instance."""
    client = genai.Client(vertexai=True, 
                          project=Config.PROJECT, 
                          location=Config.LOCATION
)
    return client

def get_rag_engine_endpoint():
    """Builds the full resource name for the RAG Engine endpoint."""
    
    return f"projects/{Config.PROJECT}/locations/{Config.LOCATION}/ragEngines/{Config.RAG_ENGINE_ID}"
