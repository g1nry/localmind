from rag_agent.domain.contracts import VectorStore
from rag_agent.domain.errors import ChunkNotFoundError, StaleChunkError
from rag_agent.domain.schemas import ContextItem, FetchRequest
from rag_agent.services.context import record_context


class FetchService:
    """Читать снимок фрагмента из индекса без моделей и исходных файлов."""

    def __init__(self, store: VectorStore):
        self._store = store

    def fetch(self, payload: FetchRequest) -> ContextItem:
        for record in self._store.load():
            if record.chunk_id != payload.chunk_id:
                continue
            if payload.source_id is not None and record.metadata.get('source_id') != payload.source_id:
                continue
            context = record_context(record, payload.max_chars)
            if payload.version is not None and context.version != payload.version:
                raise StaleChunkError('Фрагмент изменился после поиска; повторите поиск.')
            return context
        raise ChunkNotFoundError(f'Фрагмент не найден: {payload.chunk_id}')
