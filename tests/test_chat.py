"""
Tests for the chat functionality.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_chat_endpoint():
    """Test the chat endpoint."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello, world!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "response" in data
    assert "timestamp" in data

def test_websocket_chat():
    """Test WebSocket chat functionality."""
    with client.websocket_connect("/api/v1/chat/ws/test_conversation") as websocket:
        websocket.send_text(json.dumps({"message": "Hello, WebSocket!"}))
        data = websocket.receive_text()
        assert isinstance(data, str)
        response = json.loads(data)
        assert "type" in response
        assert "content" in response

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main(["-v", "-s", __file__]))
