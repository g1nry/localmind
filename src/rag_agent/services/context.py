from hashlib import sha256

from rag_agent.domain.schemas import ContextItem, VectorRecord


def record_context(record: VectorRecord, max_chars: int, score: float | None = None) -> ContextItem:
    metadata = record.metadata
    source_id = metadata.get('source_id')
    relative_path = metadata.get('relative_path')
    source = (f'{source_id[:12]}/{relative_path}' if source_id and relative_path
              else metadata.get('filename', 'unknown'))
    return ContextItem(
        text=record.text[:max_chars],
        source=source,
        chunk_id=record.chunk_id,
        document_id=metadata.get('document_id'),
        source_id=source_id,
        version=metadata.get('fingerprint') or sha256(record.text.encode()).hexdigest(),
        chunk_index=metadata.get('chunk_index'),
        score=score,
        truncated=len(record.text) > max_chars,
    )
