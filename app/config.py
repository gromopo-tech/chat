import os


class Config:
    """Main configuration class for the API"""

    # --------------------------
    # Environment Configuration
    # --------------------------
    ENV: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # --------------------------
    # Vertex AI Configuration
    # --------------------------
    PROJECT: str = os.getenv("VERTEX_PROJECT", os.getenv("PROJECT_ID"))
    LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
    EMBEDDING_MODEL: str = "text-embedding-004"
    DEFAULT_LLM_MODEL: str = "gemini-2.5-flash-lite"
    THINKING_LLM_MODEL: str = "gemini-2.5-pro"  # More powerful model for complex tasks
    QUERY_PARSER_MODEL: str = "gemini-2.5-flash-lite"

    # --------------------------
    # Vector Database
    # --------------------------
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    DENSE_VECTOR_SIZE: int = int(os.getenv("DENSE_VECTOR_SIZE", "768"))  # text-embedding-004 default size
    SPARSE_VECTOR_SIZE: int = int(os.getenv("SPARSE_VECTOR_SIZE", "30522"))  # text-embedding-004 sparse size (if available)
    COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "reviews")
    
    # Hybrid search weights (for when sparse is available)
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.7"))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT", "0.3"))

    @classmethod
    def is_production(cls) -> bool:
        return cls.ENV == "production"

    @classmethod
    def validate_config(cls):
        """Validate critical configuration"""
        if cls.is_production() and cls.DEBUG:
            raise ValueError("DEBUG mode should not be enabled in production")


# Validate configuration when imported
Config.validate_config()
