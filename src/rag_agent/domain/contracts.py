from typing import Protocol

from rag_agent.domain.schemas import VectorRecord


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def load(self) -> list[VectorRecord]: ...

    def replace_document(self, filename: str, records: list[VectorRecord]) -> None: ...
