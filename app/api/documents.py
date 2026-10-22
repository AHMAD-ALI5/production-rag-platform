
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.pdf_service import (
    MAX_PDF_SIZE,
    extract_pdf_pages,
    save_processed_pdf,
)

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


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    # Read at most one byte beyond the limit to detect oversized uploads.
    pdf_bytes = await file.read(MAX_PDF_SIZE + 1)
    await file.close()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

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
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
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
