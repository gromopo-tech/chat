import os
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings


llm = ChatVertexAI(
    model="gemini-2.5-flash",
    project=os.getenv("VERTEX_PROJECT", os.getenv("PROJECT_ID")),
    location=os.getenv("VERTEX_LOCATION", "us-central1"),
)

embeddings_model = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    project=os.getenv("VERTEX_PROJECT", os.getenv("PROJECT_ID")),
    location=os.getenv("VERTEX_LOCATION", "us-central1"),
)
