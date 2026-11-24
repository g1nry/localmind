from fastapi.testclient import TestClient

from rag_agent.core.config import Settings
from rag_agent.main import create_app


def test_backend_starts_without_running_ollama(tmp_path):
    app = create_app(Settings(DATA_DIR=tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "agent_initialized": False}
        assert app.state.services.agent is None
    assert not hasattr(app.state, "services")


def test_chat_routes_keep_success_contract_and_report_errors(tmp_path, monkeypatch):
    from rag_agent.agents.main_agent import RedTeamAgent
    from rag_agent.domain.errors import AgentError

    async def invoke(self, message, thread_id):
        if message == 'fail':
            raise AgentError('Ответ недоступен.')
        return {'response': f'{thread_id}: {message}', 'success': True}

    monkeypatch.setattr(RedTeamAgent, 'ainvoke', invoke)
    first_app = create_app(Settings(DATA_DIR=tmp_path / 'first'))
    second_app = create_app(Settings(DATA_DIR=tmp_path / 'second'))
    with TestClient(first_app) as first, TestClient(second_app) as second:
        for route in ['/api/v1/chat', '/api/agent/chat']:
            response = first.post(route, json={'message': 'hello', 'thread_id': 'test'})
            assert response.status_code == 200
            assert response.json() == {'answer': 'test: hello', 'contexts': []}
        assert first.get('/health').json()['agent_initialized']
        assert not second.get('/health').json()['agent_initialized']
        assert first_app.state.services is not second_app.state.services
        response = first.post('/api/agent/chat', json={'message': 'fail'})
        assert response.status_code == 502
        assert response.json() == {'detail': 'Ответ недоступен.'}
