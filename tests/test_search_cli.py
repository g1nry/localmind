import json
import sys

import pytest

from rag_agent.__main__ import main
from rag_agent.core.config import Settings
from rag_agent.domain.errors import ProviderError
from rag_agent.domain.schemas import VectorRecord
from rag_agent.domain.sources import DirectorySource
from rag_agent.storage.vector_store import JsonVectorStore


@pytest.fixture
def settings(tmp_path, monkeypatch):
    settings = Settings(DATA_DIR=tmp_path / 'data', sources={'docs': DirectorySource(root=tmp_path / 'docs')})
    monkeypatch.setattr('rag_agent.__main__.get_settings', lambda: settings)
    return settings


def invoke(monkeypatch, args):
    monkeypatch.setattr(sys, 'argv', ['localmind', *args])
    main()


def test_index_search_fetch_roundtrip(settings, tmp_path, monkeypatch, capsys):
    root = settings.sources['docs'].root
    root.mkdir()
    (root / 'notes.md').write_text('Known fact')
    calls = []

    async def embed(self, model, text):
        calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr('rag_agent.integrations.ollama.client.OllamaClient.embed', embed)
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, ['index', '--source', 'docs', '--json'])
    assert exc.value.code == 0
    capsys.readouterr()
    invoke(monkeypatch, ['search', 'fact', '--source', 'docs', '--max-chars', '5', '--json'])
    result = json.loads(capsys.readouterr().out)
    assert result['truncated']
    hit = result['contexts'][0]
    assert hit['text'] == 'Known' and hit['score'] == 1.0
    assert hit['document_id'] and hit['source_id'] and hit['version']
    (root / 'notes.md').unlink()  # Fetch использует индекс, не исходник.

    def no_client(*args, **kwargs):
        pytest.fail('fetch must not create an HTTP client')

    monkeypatch.setattr('httpx.AsyncClient', no_client)
    invoke(monkeypatch, ['fetch', hit['chunk_id'], '--version', hit['version'], '--json'])
    context = json.loads(capsys.readouterr().out)
    assert context['text'] == 'Known fact' and not context['truncated']
    assert calls == ['Known fact', 'fact']


def test_empty_search_and_missing_fetch_do_not_write(settings, monkeypatch, capsys):
    invoke(monkeypatch, ['search', 'question', '--json'])
    assert json.loads(capsys.readouterr().out) == {'contexts': [], 'truncated': False}
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, ['fetch', 'missing', '--json'])
    output = capsys.readouterr()
    assert exc.value.code == 1 and output.out == ''
    assert 'Фрагмент не найден' in output.err
    assert not settings.data_dir.exists()


@pytest.mark.parametrize('args', [
    ['search', '   '], ['search', 'a', '--top-k', '0'], ['search', 'a', '--top-k', '101'],
    ['fetch', 'a', '--max-chars', '-1'], ['search', 'a', '--max-chars', '100001'],
    ['search', 'a', '--source', 'missing'], ['--reload', 'search', 'a'],
])
def test_bad_arguments_are_usage_errors(settings, monkeypatch, args):
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, args)
    assert exc.value.code == 2


def test_provider_error_is_stderr_and_exit_one(settings, monkeypatch, capsys):
    JsonVectorStore(settings.index_path).add_many([
        VectorRecord(chunk_id='doc:0', text='fact', embedding=[1.0, 0.0])
    ])

    async def embed(*args):
        raise ProviderError('offline')

    monkeypatch.setattr('rag_agent.integrations.ollama.client.OllamaClient.embed', embed)
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, ['search', 'a', '--json'])
    output = capsys.readouterr()
    assert exc.value.code == 1 and output.out == ''
    assert 'offline' in output.err
