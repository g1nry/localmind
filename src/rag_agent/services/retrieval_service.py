from rag_agent.core.config import Settings
from rag_agent.domain.schemas import ContextItem, RetrievalRequest, RetrievalResponse, RetrievalResult
from rag_agent.rag.retriever import cosine_similarity
from rag_agent.domain.contracts import EmbeddingProvider, VectorStore

class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    async def search(self, payload: RetrievalRequest) -> RetrievalResponse:
        result = await self.retrieve_context(payload.query, payload.top_k)
        return RetrievalResponse(contexts=result.contexts)

    async def retrieve_context(self, query: str, top_k: int | None = None) -> RetrievalResult:
        payload = RetrievalRequest(query=query, top_k=top_k)
        records = self._vector_store.load()
        if not records:
            return RetrievalResult(contexts=[])
        query_embedding = await self._embedding_provider.embed(payload.query)

        scored = [
            (cosine_similarity(query_embedding, record.embedding), record)
            for record in records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        limit = payload.top_k or self._settings.default_top_k
        contexts: list[ContextItem] = []

        for score, record in scored[:limit]:
            if score < self._settings.min_retrieval_score:
                continue

            contexts.append(
                ContextItem(
                    text=record.text,
                    source=(f"{record.metadata['source_id'][:12]}/{record.metadata['relative_path']}"
                            if "source_id" in record.metadata and "relative_path" in record.metadata
                            else record.metadata.get("filename", "unknown")),
                    chunk_id=record.chunk_id,
                )
            )

        return RetrievalResult(contexts=contexts)
