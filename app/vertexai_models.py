import vertexai
from app.config import Config
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# Initialize Vertex AI
vertexai.init(project=Config.PROJECT, location=Config.LOCATION)

default_llm = ChatVertexAI(
    model=Config.DEFAULT_LLM_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)

# More powerful model for complex tasks like recommendations
thinking_llm = ChatVertexAI(
    model=Config.THINKING_LLM_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)

query_parser_llm = ChatVertexAI(
    model=Config.QUERY_PARSER_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)

def get_llm_for_query(user_query: str) -> ChatVertexAI:
    """Select the appropriate LLM based on the query type."""
    query_lower = user_query.lower()
    
    # Use pro model for complex analytical tasks
    pro_keywords = [
        "recommend", "recommendation", "recommendations",
        "suggest", "suggestion", "suggestions",
        "advice", "improve", "improvement", "improvements", 
        "strategy", "strategies", "solution", "solutions",
        "action", "actions", "plan", "planning",
        "optimize", "optimization", "enhance", "enhancement",
        "best", "better", "ideal", "perfect", "optimal"
    ]
    
    if any(keyword in query_lower for keyword in pro_keywords):
        print(f"Using Pro model for query: {user_query[:50]}...")
        return thinking_llm
    else:
        return default_llm

# Use the native Vertex AI model for hybrid embeddings
embeddings_model = TextEmbeddingModel.from_pretrained(Config.EMBEDDING_MODEL)

# Legacy embedding model for compatibility (if needed)
legacy_embeddings_model = VertexAIEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    project=Config.PROJECT,
    location=Config.LOCATION,
)

def get_hybrid_embeddings(text: str):
    """Get both dense and sparse embeddings for hybrid search."""
    try:
        # Get embeddings without task_type since it's not supported
        embeddings = embeddings_model.get_embeddings(
            [text],
            output_dimensionality=Config.DENSE_VECTOR_SIZE
        )
        
        embedding = embeddings[0]
        
        # Extract dense vector
        dense_vector = embedding.values  # Dense vector
        sparse_vector = None
        
        # Check if sparse embedding is available in the response
        # Note: text-embedding-004 may not support sparse embeddings yet
        if hasattr(embedding, 'sparse_embedding') and embedding.sparse_embedding:
            sparse_vector = embedding.sparse_embedding
        elif hasattr(embedding, 'sparse') and embedding.sparse:
            sparse_vector = embedding.sparse
        
        return {
            'dense': dense_vector,
            'sparse': sparse_vector
        }
    except Exception as e:
        print(f"Error getting hybrid embeddings: {e}")
        # Fallback to legacy embeddings model
        try:
            dense_vector = legacy_embeddings_model.embed_query(text)
            return {
                'dense': dense_vector,
                'sparse': None
            }
        except Exception as fallback_error:
            print(f"Fallback embedding also failed: {fallback_error}")
            raise e

def get_query_embeddings(text: str):
    """Get embeddings optimized for query."""
    try:
        embeddings = embeddings_model.get_embeddings(
            [text],
            output_dimensionality=Config.DENSE_VECTOR_SIZE
        )
        
        embedding = embeddings[0]
        
        # Extract dense and sparse vectors
        dense_vector = embedding.values
        sparse_vector = None
        
        if hasattr(embedding, 'sparse_embedding') and embedding.sparse_embedding:
            sparse_vector = embedding.sparse_embedding
        elif hasattr(embedding, 'sparse') and embedding.sparse:
            sparse_vector = embedding.sparse
        
        return {
            'dense': dense_vector,
            'sparse': sparse_vector
        }
    except Exception as e:
        print(f"Error getting query embeddings: {e}")
        # Fallback to legacy embeddings model
        try:
            dense_vector = legacy_embeddings_model.embed_query(text)
            return {
                'dense': dense_vector,
                'sparse': None
            }
        except Exception as fallback_error:
            print(f"Fallback embedding also failed: {fallback_error}")
            raise e
