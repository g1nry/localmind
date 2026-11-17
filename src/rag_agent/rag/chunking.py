def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Разбить текст на перекрывающиеся фрагменты по числу символов."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Expected chunk_size > 0 and 0 <= overlap < chunk_size.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks
