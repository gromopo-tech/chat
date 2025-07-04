from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock
from app.main import app


@pytest.fixture
def mock_rag_response(mocker):
    mocker.patch(
        "app.main.get_rag_response",
        return_value={
            "answer": "Mocked LLM answer about dislikes.",
            "context": ["The soup was cold.", "The staff was rude."],
        },
    )


def test_rag_query_with_mocked_rag(mock_rag_response):
    client = TestClient(app)
    response = client.post(
        "/rag/query",
        json={
            "place_id": "ChIJuVyExGENK4cRooPhJIUgnxk",
            "query": "What do people dislike about this place?",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM answer about dislikes."
    assert "soup was cold" in str(data["context"])
