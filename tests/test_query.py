from unittest.mock import patch, MagicMock
from app.main import app
from app.data_models import QueryRequest, ChatMessage


@patch("app.query_parser.query_parser_llm")
def test_parse_query_with_llm(mock_llm):
    """Test query parsing with LLM."""
    from app.query_parser import parse_query_with_llm
    
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.content = '''
    {
        "off_topic": false,
        "query_embedding_text": "complaints about service quality issues",
        "filter": {
            "rating": {"$in": [1, 2, 3]}, 
            "createTime": {"$gte": "2024-05-01T00:00:00"}
        }
    }
    '''
    mock_llm.invoke.return_value = mock_response
    
    user_query = "What do people dislike about the service in the last month?"
    parsed = parse_query_with_llm(user_query)
    
    assert parsed["off_topic"] == False
    assert "complaints about service" in parsed["query_embedding_text"]
    assert parsed["filter"]["rating"]["$in"] == [1, 2, 3]
    assert "2024-05-01T00:00:00" in parsed["filter"]["createTime"]["$gte"]


def test_build_qdrant_filter():
    """Test Qdrant filter building."""
    from app.vectorstore import build_qdrant_filter
    
    parsed_filter = {
        "rating": {"$in": [1, 2, 3]},
        "createTime": {"$gte": "2024-05-01T00:00:00"},
    }
    qdrant_filter = build_qdrant_filter(parsed_filter)
    
    # Check that the filter contains the correct must conditions
    assert qdrant_filter is not None
    keys = [cond.key for cond in qdrant_filter.must]
    assert "rating" in keys
    assert "createTime" in keys


def test_get_k_value_for_query():
    """Test dynamic k value selection."""
    from app.chains import _get_k_value_for_query
    
    # Test analytical queries
    assert _get_k_value_for_query("What are the most common complaints?") == 1000
    assert _get_k_value_for_query("How many reviews are there?") == 1000
    assert _get_k_value_for_query("Summarize all feedback") == 1000
    
    # Test comparison queries
    assert _get_k_value_for_query("Compare service vs food quality") == 100
    
    # Test specific queries
    assert _get_k_value_for_query("Tell me about a specific complaint") == 30
    
    # Test default
    assert _get_k_value_for_query("Random query") == 50


@patch("app.chains.create_hybrid_retriever")
@patch("app.chains.parse_query_with_llm")
def test_prepare_query(mock_parse, mock_retriever):
    """Test query preparation logic."""
    from app.chains import _prepare_query
    
    # Mock parsed query
    mock_parse.return_value = {
        "off_topic": False,
        "query_embedding_text": "test query",
        "filter": {"rating": {"$in": [1, 2, 3]}}
    }
    
    # Mock retriever
    mock_retriever_instance = MagicMock()
    mock_retriever.return_value = mock_retriever_instance
    
    filter_dict, embedding_text, retriever = _prepare_query("test query")
    
    assert filter_dict == {"rating": {"$in": [1, 2, 3]}}
    assert embedding_text == "test query"
    assert retriever == mock_retriever_instance
    
    # Check that retriever was created with correct parameters
    mock_retriever.assert_called_once()
    call_args = mock_retriever.call_args
    assert call_args[1]["k"] == 50  # Default k value for "test query"


def test_query_request_model():
    """Test the QueryRequest data model."""
    # Test basic request
    request = QueryRequest(query="test query")
    assert request.query == "test query"
    assert request.session_id is None
    assert request.chat_history == []
    
    # Test request with chat history
    chat_history = [ChatMessage(human="Hi", ai="Hello")]
    request = QueryRequest(
        query="follow up", 
        session_id="test_session",
        chat_history=chat_history
    )
    assert request.query == "follow up"
    assert request.session_id == "test_session"
    assert len(request.chat_history) == 1
    assert request.chat_history[0].human == "Hi"
    assert request.chat_history[0].ai == "Hello"
