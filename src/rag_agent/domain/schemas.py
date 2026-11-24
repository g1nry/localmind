from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional


class ContextItem(BaseModel):
    text: str
    source: str
    chunk_id: str
    document_id: str | None = None
    source_id: str | None = None
    version: str | None = None
    chunk_index: int | None = None
    score: float | None = None
    truncated: bool = False


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, gt=0, le=100)
    source_id: str | None = None
    max_chars: int = Field(default=4000, gt=0, le=100_000)

    @field_validator('query')
    @classmethod
    def nonblank_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError('Поисковый запрос не должен быть пустым.')
        return query


class RetrievalResult(BaseModel):
    contexts: List[ContextItem]
    truncated: bool = False


class RetrievalResponse(BaseModel):
    contexts: List[ContextItem]
    truncated: bool = False

class DocumentIngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class VectorRecord(BaseModel):
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class FetchRequest(BaseModel):
    chunk_id: str = Field(min_length=1)
    source_id: str | None = None
    version: str | None = None
    max_chars: int = Field(default=4000, gt=0, le=100_000)
