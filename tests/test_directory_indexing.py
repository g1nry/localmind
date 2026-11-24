import asyncio

from rag_agent.core.config import Settings
from rag_agent.domain.sources import DirectorySource
from rag_agent.services.directory_indexing_service import DirectoryIndexingService
from rag_agent.services.document_ingestion_service import DocumentIngestionService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore


class Embeddings:
    calls = 0
    fail = False

    async def embed(self, text):
        self.calls += 1
        if self.fail:
            from rag_agent.domain.errors import ProviderError
            raise ProviderError('unavailable')
        return [1.0, 0.0]


def make_service(tmp_path):
    settings = Settings(DATA_DIR=tmp_path / 'data')
    store = JsonVectorStore(settings.index_path)
    embeddings = Embeddings()
    ingestion = DocumentIngestionService(settings, DocumentStore(settings.documents_dir), store, embeddings)
    return DirectoryIndexingService(settings, store, ingestion), store, embeddings


def test_same_names_updates_and_failed_replacement_keep_previous(tmp_path):
    root = tmp_path / 'source'
    for folder in ['a', 'b']:
        (root / folder).mkdir(parents=True)
        (root / folder / 'notes.md').write_text(folder)
    source = DirectorySource(root=root)
    service, store, embeddings = make_service(tmp_path)
    first = asyncio.run(service.index(source))
    assert first.counts['added'] == 2
    assert len({item.metadata['document_id'] for item in store.load()}) == 2
    assert embeddings.calls == 2
    second = asyncio.run(service.index(source))
    assert second.counts['unchanged'] == 2
    assert embeddings.calls == 2
    (root / 'a' / 'notes.md').write_text('changed')
    embeddings.fail = True
    assert asyncio.run(service.index(source)).counts['failed'] == 1
    assert {item.text for item in store.load()} == {'a', 'b'}
    embeddings.fail = False
    assert asyncio.run(service.index(source)).counts['updated'] == 1
    assert {item.text for item in store.load()} == {'changed', 'b'}
    (root / 'b' / 'notes.md').unlink()
    asyncio.run(service.index(source))
    assert len(store.load()) == 2  # Удаление — отдельная будущая операция.


def test_dry_run_exclusions_symlinks_and_bad_documents(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'good.md').write_text('good')
    (root / 'bad.md').write_bytes(b'\xff')
    (root / 'empty.txt').touch()
    (root / '.env.md').write_text('secret')
    (root / 'private').mkdir()
    (root / 'private' / 'secret.md').write_text('private')
    (root / 'linked.md').symlink_to(root / 'good.md')
    (root / 'linked-directory').symlink_to(root / 'private', target_is_directory=True)
    service, store, embeddings = make_service(tmp_path)
    report = asyncio.run(service.index(DirectorySource(root=root, exclude=['**/private/**']), dry_run=True))
    assert report.counts == {'added': 1, 'updated': 0, 'unchanged': 0, 'skipped': 4, 'failed': 2}
    assert embeddings.calls == 0
    assert store.load() == []
    assert not (tmp_path / 'data').exists()


def test_read_error_does_not_stop_other_files(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'bad.md').write_text('bad')
    (root / 'good.md').write_text('good')
    service, store, _ = make_service(tmp_path)
    real_read = service._read

    def read(path, source_root):
        if path.name == 'bad.md':
            raise PermissionError('denied')
        return real_read(path, source_root)

    monkeypatch.setattr(service, '_read', read)
    report = asyncio.run(service.index(DirectorySource(root=root)))
    assert report.counts['failed'] == 1
    assert report.counts['added'] == 1
    assert store.load()[0].text == 'good'
