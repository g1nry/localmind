import asyncio
from pathlib import Path

from rag_agent.core.config import Settings
from rag_agent.services.retrieval_service import RetrievalService
from rag_agent.storage.vector_store import JsonVectorStore, VectorRecord


class FakeLLMService:
    async def embed(self, text: str) -> list[float]:
        if text == "alpha":
            return [1.0, 0.0]
        if text == "beta":
            return [0.0, 1.0]
        return [1.0, 0.0]


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_NAME="rag-agent",
        APP_HOST="127.0.0.1",
        APP_PORT=8000,
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_CHAT_MODEL="llama3.2:1b",
        OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
        DATA_DIR=tmp_path,
        DOCUMENTS_DIR=tmp_path / "documents",
        INDEX_PATH=tmp_path / "indexes" / "vector_index.json",
        MAX_CHUNK_SIZE=50,
        CHUNK_OVERLAP=10,
        DEFAULT_TOP_K=4,
        MIN_RETRIEVAL_SCORE=0.2,
    )


def test_retrieval_returns_ranked_contexts(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    vector_store = JsonVectorStore(settings.index_path)
    vector_store.add_many(
        [
            VectorRecord(
                chunk_id="doc.txt:0",
                text="alpha text",
                embedding=[1.0, 0.0],
                metadata={"filename": "doc.txt"},
            ),
            VectorRecord(
                chunk_id="doc.txt:1",
                text="beta text",
                embedding=[0.0, 1.0],
                metadata={"filename": "doc.txt"},
            ),
        ]
    )
    retrieval_service = RetrievalService(
        settings=settings,
        vector_store=vector_store,
        embedding_provider=FakeLLMService(),
    )

    result = asyncio.run(retrieval_service.retrieve_context("alpha", top_k=2))

    assert len(result.contexts) == 1
    assert result.contexts[0].chunk_id == "doc.txt:0"
    assert result.contexts[0].source == "doc.txt"
    assert result.contexts[0].text == "alpha text"


def test_empty_index_does_not_require_embedding_model(tmp_path):
    class UnavailableEmbeddings:
        async def embed(self, text):
            raise AssertionError('Empty index must not call a model')

    settings = build_settings(tmp_path)
    service = RetrievalService(settings, JsonVectorStore(settings.index_path), UnavailableEmbeddings())
    assert asyncio.run(service.retrieve_context('question')).contexts == []


def test_source_filter_applies_before_scoring_and_budget_is_global(tmp_path):
    settings = build_settings(tmp_path)
    store = JsonVectorStore(settings.index_path)
    store.add_many([
        VectorRecord(chunk_id='foreign', text='foreign', embedding=[1.0, 2.0, 3.0],
                     metadata={'source_id': 'other'}),
        VectorRecord(chunk_id='a', text='abcdef', embedding=[1.0, 0.0],
                     metadata={'source_id': 'chosen', 'fingerprint': 'v1'}),
        VectorRecord(chunk_id='b', text='ghijkl', embedding=[1.0, 0.0],
                     metadata={'source_id': 'chosen', 'fingerprint': 'v2'}),
    ])
    service = RetrievalService(settings, store, FakeLLMService())
    result = asyncio.run(service.retrieve_context('alpha', top_k=2, source_id='chosen', max_chars=8))
    assert [item.text for item in result.contexts] == ['abcdef', 'gh']
    assert result.truncated and result.contexts[1].truncated
    assert result.contexts[0].score == 1.0
    assert [item.version for item in result.contexts] == ['v1', 'v2']
    assert asyncio.run(service.retrieve_context('alpha', source_id='missing')).contexts == []


def test_blank_query_and_invalid_limits_fail_before_embeddings(tmp_path):
    import pytest
    from pydantic import ValidationError
    settings = build_settings(tmp_path)
    service = RetrievalService(settings, JsonVectorStore(settings.index_path), FakeLLMService())
    for kwargs in [{'query': '   '}, {'query': 'alpha', 'top_k': 0},
                   {'query': 'alpha', 'max_chars': 0}, {'query': 'alpha', 'top_k': 101}]:
        with pytest.raises(ValidationError):
            asyncio.run(service.retrieve_context(**kwargs))
