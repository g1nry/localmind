from pathlib import Path

from rag_agent.core.config import load_settings
from rag_agent.domain.schemas import VectorRecord
from rag_agent.domain.sources import DirectorySource
from rag_agent.storage.vector_store import JsonVectorStore


def test_source_identity_uses_root_and_relative_path(tmp_path):
    first = DirectorySource(root=tmp_path / 'first')
    second = DirectorySource(root=tmp_path / 'second')
    assert first.document_id('a/notes.md') != first.document_id('b/notes.md')
    assert first.document_id('a/notes.md') != second.document_id('a/notes.md')
    assert first.source_id == DirectorySource(root=tmp_path / 'first' / '.').source_id


def test_sources_resolve_relative_to_config(tmp_path, monkeypatch):
    config = tmp_path / 'config.toml'
    config.write_text('[sources.docs]\nroot = "./knowledge"\nexclude = ["private/**"]\n')
    monkeypatch.chdir(Path('/'))
    settings = load_settings(config)
    assert settings.sources['docs'].root == tmp_path / 'knowledge'
    assert settings.sources['docs'].exclude == ['private/**']


def test_missing_index_is_read_only_and_source_replacement_keeps_legacy(tmp_path):
    path = tmp_path / 'data' / 'index.json'
    store = JsonVectorStore(path)
    assert store.load() == []
    assert not path.parent.exists()
    legacy = VectorRecord(chunk_id='notes.md:0', text='legacy', embedding=[1.0],
                          metadata={'filename': 'notes.md'})
    source = VectorRecord(chunk_id='doc:0', text='new', embedding=[1.0],
                          metadata={'filename': 'notes.md', 'document_id': 'doc'})
    store.add_many([legacy, source])
    source.text = 'updated'
    store.replace_document('doc', [source])
    assert [item.text for item in store.load()] == ['legacy', 'updated']
