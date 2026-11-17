from fastapi.testclient import TestClient

from rag_agent.main import app


def test_backend_starts_without_running_ollama():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent_initialized": True}
