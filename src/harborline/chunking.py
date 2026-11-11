"""Character chunker with overlap. Short policy pages stay one chunk."""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than chunk size")
    body = text.strip()
    if not body:
        return []
    if len(body) <= size:
        return [body]
    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + size, len(body))
        chunks.append(body[start:end])
        if end == len(body):
            break
        start = end - overlap
    return chunks
