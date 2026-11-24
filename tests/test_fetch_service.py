import pytest

from rag_agent.domain.errors import ChunkNotFoundError, StaleChunkError
from rag_agent.domain.schemas import FetchRequest, VectorRecord
from rag_agent.services.fetch_service import FetchService
from rag_agent.storage.vector_store import JsonVectorStore


def test_fetch_reads_indexed_snapshot_and_checks_version(tmp_path):
    store = JsonVectorStore(tmp_path / 'index.json')
    record = VectorRecord(chunk_id='doc:0', text='known fact', embedding=[1.0],
                          metadata={'document_id': 'doc', 'source_id': 'source',
                                    'relative_path': 'notes.md', 'fingerprint': 'v1', 'chunk_index': 0})
    store.add_many([record])
    service = FetchService(store)
    result = service.fetch(FetchRequest(chunk_id='doc:0', max_chars=5))
    assert result.text == 'known' and result.truncated
    assert result.document_id == 'doc' and result.chunk_index == 0
    assert result.version == 'v1'
    record.metadata['fingerprint'] = 'v2'
    store.replace_document('doc', [record])
    with pytest.raises(StaleChunkError):
        service.fetch(FetchRequest(chunk_id='doc:0', version='v1'))
    with pytest.raises(ChunkNotFoundError):
        service.fetch(FetchRequest(chunk_id='doc:0', source_id='other'))


def test_fetch_legacy_chunk_and_missing_index_stay_read_only(tmp_path):
    path = tmp_path / 'data' / 'index.json'
    store = JsonVectorStore(path)
    service = FetchService(store)
    with pytest.raises(ChunkNotFoundError):
        service.fetch(FetchRequest(chunk_id='missing'))
    assert not path.parent.exists()
    store.add_many([VectorRecord(chunk_id='legacy:0', text='old', embedding=[1.0],
                                metadata={'filename': 'legacy.md'})])
    result = service.fetch(FetchRequest(chunk_id='legacy:0'))
    assert result.source == 'legacy.md' and result.version is not None
    assert result.source_id is None and result.document_id is None
