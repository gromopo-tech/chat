from unittest.mock import MagicMock, patch

# Prevent google.auth from running during import of vertexai_models
patch("vertexai.init", MagicMock()).start()
patch("langchain_google_vertexai.ChatVertexAI", MagicMock()).start()
patch("langchain_google_vertexai.VertexAIEmbeddings", MagicMock()).start()
patch("vertexai.language_models.TextEmbeddingModel", MagicMock()).start()