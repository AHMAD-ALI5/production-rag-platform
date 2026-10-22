
import json
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_PDF_PAGES = 200


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract text while preserving one-based PDF page numbers."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=True)
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid readable PDF.") from exc

    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported yet.")

    if len(reader.pages) == 0:
        raise ValueError("The PDF contains no pages.")

    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(
            f"The PDF exceeds the limit of {MAX_PDF_PAGES} pages."
        )

    pages = []

    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            pages.append({
                "page_number": page_number,
                "text": text,
                "character_count": len(text),
            })
    except Exception as exc:
        raise ValueError("Text extraction failed for this PDF.") from exc

    return pages


def save_processed_pdf(
    pdf_bytes: bytes,
    original_filename: str,
    pages: list[dict],
) -> dict:
    """Store the original PDF and extracted page data using a unique ID."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid4())
    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(original_filename).name,
    ).strip("._")

    if not safe_name:
        safe_name = "document.pdf"

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    json_path = UPLOAD_DIR / f"{document_id}.json"

    metadata = {
        "document_id": document_id,
        "original_filename": safe_name,
        "page_count": len(pages),
        "needs_ocr": not any(page["text"] for page in pages),
        "pages": pages,
    }

    try:
        pdf_path.write_bytes(pdf_bytes)
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pdf_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise

    return metadata
