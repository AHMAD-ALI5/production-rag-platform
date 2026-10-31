
def chunk_pages(
    document_id: str,
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    """Split extracted PDF pages into overlapping, page-aware chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []

    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "").strip()

        if not text:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer a word boundary, but don't create tiny chunks.
            if end < len(text):
                boundary = text.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": (
                        f"{document_id}:p{page_number}:c{chunk_index}"
                    ),
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                })
                chunk_index += 1

            if end >= len(text):
                break

            # Move forward while retaining overlap with the previous chunk.
            start = max(start + 1, end - overlap)

    return chunks
