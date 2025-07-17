from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock, patch
from app.main import app


@pytest.fixture
def mock_rag_response(mocker):
    mocker.patch(
        "app.main.get_rag_response",
        return_value={
            "answer": "Mocked LLM answer about dislikes.",
            "context": ["The soup was cold.", "The staff was rude."],
            "intent": "list_cons",
            "parsed_filter": {"rating": {"$lte": 3}},
        },
    )


def test_rag_query_with_mocked_rag(mock_rag_response):
    client = TestClient(app)
    response = client.post(
        "/rag/query",
        json={
            "query": "What do people dislike about this place?",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM answer about dislikes."
    assert "soup was cold" in str(data["context"])
    assert data["intent"] == "list_cons"
    assert data["parsed_filter"] == {"rating": {"$lte": 3}}


from app.query_parser import parse_query_with_llm
from app.chains import build_qdrant_filter


@patch("app.query_parser.llm")
def test_parse_query_with_llm(mock_llm):
    # Mock the LLM output
    mock_llm.invoke.return_value = (
        '{"query_embedding_text": "What do people dislike about this place?",'
        ' "filter": {"rating": {"$lte": 3}, "createTime": {"$gte": "2024-05-01T00:00:00Z"}},'
        ' "intent": "list_cons"}'
    )
    user_query = "What do people dislike about this place in the last month?"
    parsed = parse_query_with_llm(user_query)
    assert parsed["query_embedding_text"] == "What do people dislike about this place?"
    assert parsed["filter"]["rating"]["$lte"] == 3
    assert parsed["filter"]["createTime"]["$gte"] == "2024-05-01T00:00:00Z"
    assert parsed["intent"] == "list_cons"


def test_build_qdrant_filter():
    parsed_filter = {
        "rating": {"$lte": 3, "$gte": 1},
        "createTime": {"$gte": "2024-05-01T00:00:00Z"},
    }
    qdrant_filter = build_qdrant_filter(parsed_filter)
    # Check that the filter contains the correct must conditions
    keys = [cond.key for cond in qdrant_filter.must]
    assert "rating" in keys
    assert "createTime" in keys
