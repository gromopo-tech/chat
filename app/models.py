from app.config import Config
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings


llm = ChatVertexAI(
    model=Config.LLM_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)

query_parser_llm = ChatVertexAI(
    model=Config.QUERY_PARSER_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)

embeddings_model = VertexAIEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)
