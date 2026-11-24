import asyncio
import os
import subprocess
import sys

import httpx
import pytest

from rag_agent.bootstrap import open_services
from rag_agent.core.config import Settings
from rag_agent.domain.errors import ProviderError


def test_imports_create_no_resources_and_do_not_import_chat(tmp_path):
    code = '''
import importlib.abc
import sys
from unittest.mock import patch
class NoChat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'langchain_ollama', 'langgraph'}:
            raise AssertionError('Chat dependency imported: ' + fullname)
sys.meta_path.insert(0, NoChat())
with patch('httpx.AsyncClient', side_effect=AssertionError('HTTP client created')), \\
     patch('httpx.Client', side_effect=AssertionError('HTTP client created')):
    import rag_agent.services.retrieval_service
    import rag_agent.services.document_ingestion_service
    import rag_agent.tools.registry
    import rag_agent.main
    assert not hasattr(rag_agent.main.app.state, 'services')
'''
    env = {**os.environ, 'DATA_DIR': str(tmp_path / 'data'),
           'RAG_AGENT_CONFIG': str(tmp_path / 'missing.toml')}
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('fail', [False, True])
def test_search_uses_only_embeddings_and_closes_client(tmp_path, monkeypatch, fail):
    requests = []

    def handle(request):
        requests.append(request.url.path)
        if fail:
            raise httpx.ConnectError('offline', request=request)
        return httpx.Response(200, json={'embedding': [1.0, 0.0]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr('rag_agent.integrations.ollama.client.httpx.AsyncClient', lambda **kwargs: client)

    async def run():
        async with open_services(Settings(DATA_DIR=tmp_path)) as services:
            assert services.agent is None
            await services.ingestion.ingest_document('notes.md', b'Known fact')
            result = await services.retrieval.retrieve_context('question')
            assert result.contexts[0].source == 'notes.md'
            assert result.contexts[0].text == 'Known fact'
            assert services.agent is None

    if fail:
        with pytest.raises(ProviderError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert requests and set(requests) == {'/api/embeddings'}
    assert client.is_closed


def test_client_closes_if_storage_setup_fails(tmp_path, monkeypatch):
    client = httpx.AsyncClient()
    monkeypatch.setattr('rag_agent.integrations.ollama.client.httpx.AsyncClient', lambda **kwargs: client)

    def broken_store(*args):
        raise OSError('cannot create index')

    monkeypatch.setattr('rag_agent.bootstrap.JsonVectorStore', broken_store)

    async def run():
        async with open_services(Settings(DATA_DIR=tmp_path)):
            pytest.fail('setup should have failed')

    with pytest.raises(OSError):
        asyncio.run(run())
    assert client.is_closed


@pytest.mark.parametrize('fail', [False, True])
def test_lazy_agent_owns_and_closes_transports(tmp_path, monkeypatch, fail):
    from rag_agent.agents.main_agent import RedTeamAgent
    closed = []

    class SyncTransport(httpx.HTTPTransport):
        def __exit__(self, *args):
            closed.append('sync')
            return super().__exit__(*args)

    class AsyncTransport(httpx.AsyncHTTPTransport):
        async def __aexit__(self, *args):
            closed.append('async')
            return await super().__aexit__(*args)

    monkeypatch.setattr('rag_agent.bootstrap.httpx.HTTPTransport', SyncTransport)
    monkeypatch.setattr('rag_agent.bootstrap.httpx.AsyncHTTPTransport', AsyncTransport)
    if fail:
        async def broken_initialize(self, **kwargs):
            raise RuntimeError('initialization failed')
        monkeypatch.setattr(RedTeamAgent, 'initialize', broken_initialize)

    async def run():
        async with open_services(Settings(DATA_DIR=tmp_path)) as services:
            assert services.agent is None
            if fail:
                await services.get_agent()
            first, second = await asyncio.gather(services.get_agent(), services.get_agent())
            assert first is second
            assert first.initialized
            assert first.tool_registry.get_tool('rag_search').retrieval_service is services.retrieval
            assert closed == []
        assert sorted(closed) == ['async', 'sync']

    if fail:
        with pytest.raises(RuntimeError, match='initialization failed'):
            asyncio.run(run())
        assert sorted(closed) == ['async', 'sync']
    else:
        asyncio.run(run())


def test_cancellation_closes_client(tmp_path, monkeypatch):
    client = httpx.AsyncClient()
    monkeypatch.setattr('rag_agent.integrations.ollama.client.httpx.AsyncClient', lambda **kwargs: client)

    async def run():
        async with open_services(Settings(DATA_DIR=tmp_path)):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert client.is_closed
