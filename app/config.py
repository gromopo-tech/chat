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
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash-lite"
    QUERY_PARSER_MODEL: str = "gemini-2.5-flash-lite"
    
    # --------------------------
    # Vector Database
    # --------------------------
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "3072"))
    COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "reviews")
    
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