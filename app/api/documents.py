
import json
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.services.chunking_service import chunk_pages
from app.services.embedding_service import generate_embeddings
from app.services.pdf_service import (
    MAX_PDF_SIZE,
    UPLOAD_DIR,
    extract_pdf_pages,
    save_processed_pdf,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


class PageSummary(BaseModel):
    page_number: int
    character_count: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    extracted_character_count: int
    needs_ocr: bool
    pages: list[PageSummary]


class ProcessDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    embedding_dimension: int
    embedding_model: str
    chunks_preview: list[dict]


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, extract page text, and save document metadata."""

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    # Read only up to the configured limit plus one byte.
    pdf_bytes = await file.read(MAX_PDF_SIZE + 1)
    await file.close()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The PDF exceeds the 20 MB upload limit.",
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=415,
            detail="The uploaded file does not have a valid PDF signature.",
        )

    try:
        pages = extract_pdf_pages(pdf_bytes)
        metadata = save_processed_pdf(pdf_bytes, filename, pages)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except OSError as exc:
        logger.exception("Failed to save uploaded PDF.")
        raise HTTPException(
            status_code=500,
            detail="The document could not be saved.",
        ) from exc

    return DocumentUploadResponse(
        document_id=metadata["document_id"],
        filename=metadata["original_filename"],
        page_count=metadata["page_count"],
        extracted_character_count=sum(
            page["character_count"] for page in pages
        ),
        needs_ocr=metadata["needs_ocr"],
        pages=[
            PageSummary(
                page_number=page["page_number"],
                character_count=page["character_count"],
            )
            for page in pages
        ],
    )


@router.post(
    "/{document_id}/process",
    response_model=ProcessDocumentResponse,
)
def process_document(document_id: str):
    """Chunk a stored PDF's extracted text and generate embeddings."""

    # Validate the UUID before constructing file paths.
    try:
        normalized_id = str(UUID(document_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        ) from exc

    metadata_path = UPLOAD_DIR / f"{normalized_id}.json"
    output_path = UPLOAD_DIR / f"{normalized_id}_chunks.json"

    if not metadata_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload the PDF first.",
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Failed to read document metadata.")
        raise HTTPException(
            status_code=500,
            detail="Could not read the stored document metadata.",
        ) from exc

    if metadata.get("document_id") != normalized_id:
        raise HTTPException(
            status_code=422,
            detail="Stored document metadata is inconsistent.",
        )

    try:
        chunks = chunk_pages(
            document_id=normalized_id,
            pages=metadata.get("pages", []),
            chunk_size=500,
            overlap=80,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.exception("Failed to chunk document.")
        raise HTTPException(
            status_code=422,
            detail="The stored document text could not be chunked.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text was found. "
                "This PDF may require OCR."
            ),
        )

    try:
        embeddings, dimension = generate_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings does not match the chunk count."
            )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        output = {
            "document_id": normalized_id,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": dimension,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

        output_path.write_text(
            json.dumps(output, ensure_ascii=False),
            encoding="utf-8",
        )

    except Exception as exc:
        logger.exception("Document embedding or storage failed.")
        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check the server logs.",
        ) from exc

    return ProcessDocumentResponse(
        document_id=normalized_id,
        chunk_count=len(chunks),
        embedding_dimension=dimension,
        embedding_model=settings.embedding_model,
        chunks_preview=[
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "character_count": chunk["character_count"],
                "text": chunk["text"][:160],
            }
            for chunk in chunks[:5]
        ],
    )
