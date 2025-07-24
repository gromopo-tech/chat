import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Main configuration class for the API"""
    
    # --------------------------
    # Environment Configuration
    # --------------------------
    ENV: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # --------------------------
    # Gemini Models
    # --------------------------
    EMBEDDING_MODEL: str = "text-embedding-004"
    LLM_MODEL: str = "gemini-2.5-flash"

    # --------------------------
    # Vertex AI Configuration
    # --------------------------
    PROJECT: str = os.getenv("VERTEX_PROJECT", os.getenv("PROJECT_ID"))
    LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
    RAG_ENGINE_ID: str = os.getenv("RAG_ENGINE_ID")
    
    # --------------------------
    # Vector Database/Index Configuration
    # --------------------------
    INDEX_NAME: str = os.getenv("VERTEX_DEPLOYED_INDEX_NAME")
    VECTOR_DIMENSIONS: int = int(os.getenv("VECTOR_DIMENSIONS", "768"))
    
    # --------------------------
    # Authentication & Security
    # --------------------------
    
    # --------------------------
    # External Services
    # --------------------------
    
    # --------------------------
    # Rate Limiting
    # --------------------------
    
    # --------------------------
    # Constants (not environment-configurable)
    # --------------------------
    
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