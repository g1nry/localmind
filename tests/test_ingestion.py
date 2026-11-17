import asyncio

import pytest

from rag_agent.core.config import Settings
from rag_agent.rag.chunking import split_text
from rag_agent.services.document_ingestion_service import DocumentIngestionService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore


class FakeEmbeddings:
    async def embed(self, text):
        return [1.0, 0.0]


def test_reindex_replaces_old_document_chunks(tmp_path):
    settings = Settings(DATA_DIR=tmp_path, MAX_CHUNK_SIZE=5, CHUNK_OVERLAP=2)
    store = JsonVectorStore(settings.index_path)
    service = DocumentIngestionService(
        settings, DocumentStore(settings.documents_dir), store, FakeEmbeddings()
    )

    result = asyncio.run(service.ingest_document("test.md", b"abcdefgh"))
    assert result.chunks_indexed == 2
    assert [record.text for record in store.load()] == ["abcde", "defgh"]

    asyncio.run(service.ingest_document("test.md", b"new"))
    assert [record.text for record in store.load()] == ["new"]
    assert (settings.documents_dir / "test.md").read_text() == "new"


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (5, -1), (5, 5)])
def test_invalid_chunk_settings_fail(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size, overlap)
