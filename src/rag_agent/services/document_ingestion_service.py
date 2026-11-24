from pathlib import Path

from rag_agent.core.config import Settings
from rag_agent.domain.schemas import DocumentIngestResponse, VectorRecord
from rag_agent.domain.sources import DirectorySource
from rag_agent.rag.chunking import split_text
from rag_agent.services.document_errors import (
    DocumentEncodingError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from rag_agent.domain.contracts import EmbeddingProvider, VectorStore
from rag_agent.storage.document_store import DocumentStore


class DocumentIngestionService:
    def __init__(
        self,
        settings: Settings,
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._document_store = document_store
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    async def ingest_document(self, filename: str, content: bytes) -> DocumentIngestResponse:
        text = self.validate_document(filename, content)
        stored_filename = Path(filename).name or "document.txt"
        records = await self._prepare_records(stored_filename, text, {"filename": stored_filename})
        self._document_store.save(stored_filename, text)
        self._vector_store.replace_document(stored_filename, records)
        return DocumentIngestResponse(filename=stored_filename, chunks_indexed=len(records))

    async def ingest_source_document(
        self, source: DirectorySource, relative_path: str, content: bytes, fingerprint: str
    ) -> int:
        text = self.validate_document(relative_path, content)
        document_id = source.document_id(relative_path)
        records = await self._prepare_records(document_id, text, {
            "filename": relative_path,
            "document_id": document_id,
            "source_id": source.source_id,
            "relative_path": relative_path,
            "fingerprint": fingerprint,
            "embedding_model": self._settings.ollama_embedding_model,
        })
        self._vector_store.replace_document(document_id, records)
        return len(records)

    def validate_document(self, filename: str, content: bytes) -> str:
        self._validate_file(filename, content)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentEncodingError(filename) from exc
        if not text.strip():
            raise EmptyDocumentError()
        return text

    async def _prepare_records(self, identifier: str, text: str, metadata: dict) -> list[VectorRecord]:
        chunks = split_text(
            text=text,
            chunk_size=self._settings.max_chunk_size,
            overlap=self._settings.chunk_overlap,
        )

        records: list[VectorRecord] = []
        for index, chunk in enumerate(chunks):
            embedding = await self._embedding_provider.embed(chunk)
            records.append(
                VectorRecord(
                    chunk_id=f"{identifier}:{index}",
                    text=chunk,
                    embedding=embedding,
                    metadata={**metadata, "chunk_index": index},
                )
            )

        return records

    def _validate_file(self, filename: str, content: bytes) -> None:
        if not content:
            raise EmptyDocumentError()

        if len(content) > self._settings.max_upload_size_bytes:
            raise DocumentTooLargeError(self._settings.max_upload_size_bytes)

        extension = Path(filename).suffix.lower()
        if extension not in self._settings.allowed_document_extensions:
            raise UnsupportedDocumentTypeError(filename)
