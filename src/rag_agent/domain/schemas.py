from pydantic import BaseModel, Field
from typing import Any, List, Optional


class ContextItem(BaseModel):
    text: str
    source: str
    chunk_id: str


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, gt=0)


class RetrievalResult(BaseModel):
    contexts: List[ContextItem]


class RetrievalResponse(BaseModel):
    contexts: List[ContextItem]

class DocumentIngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class VectorRecord(BaseModel):
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
