from rag_agent.core.config import Settings
from rag_agent.domain.contracts import EmbeddingProvider, VectorStore
from rag_agent.domain.schemas import RetrievalRequest, RetrievalResponse, RetrievalResult
from rag_agent.rag.retriever import cosine_similarity
from rag_agent.services.context import record_context


class RetrievalService:
    def __init__(self, settings: Settings, vector_store: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    async def search(self, payload: RetrievalRequest) -> RetrievalResponse:
        result = await self.retrieve_context(
            payload.query, payload.top_k, source_id=payload.source_id, max_chars=payload.max_chars
        )
        return RetrievalResponse(contexts=result.contexts, truncated=result.truncated)

    async def retrieve_context(
        self, query: str, top_k: int | None = None, *, source_id: str | None = None, max_chars: int = 4000
    ) -> RetrievalResult:
        payload = RetrievalRequest(
            query=query, top_k=top_k if top_k is not None else self._settings.default_top_k,
            source_id=source_id, max_chars=max_chars,
        )
        records = [record for record in self._vector_store.load()
                   if payload.source_id is None or record.metadata.get('source_id') == payload.source_id]
        if not records:
            return RetrievalResult(contexts=[])
        query_embedding = await self._embedding_provider.embed(payload.query)
        scored = [(cosine_similarity(query_embedding, record.embedding), record) for record in records]
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [(score, record) for score, record in scored[:payload.top_k]
                      if score >= self._settings.min_retrieval_score]
        contexts = []
        remaining = payload.max_chars
        truncated = False
        for score, record in candidates:
            if remaining <= 0:
                truncated = True
                break
            context = record_context(record, remaining, score)
            contexts.append(context)
            remaining -= len(context.text)
            truncated = truncated or context.truncated
        return RetrievalResult(contexts=contexts, truncated=truncated)
