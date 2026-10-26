
from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app
import json
from uuid import uuid4

from app.services.chunking_service import chunk_pages
client = TestClient(app)


def make_blank_pdf() -> bytes:
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    return buffer.getvalue()


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_invalid_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "broken.pdf",
                b"%PDF-this-is-not-a-valid-pdf",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422


def test_upload_accepts_blank_pdf_and_flags_ocr(tmp_path, monkeypatch):
    import app.services.pdf_service as pdf_service

    monkeypatch.setattr(pdf_service, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "blank.pdf",
                make_blank_pdf(),
                "application/pdf",
            )
        },
    )




def test_chunking_preserves_page_numbers():
    pages = [
        {"page_number": 1, "text": "alpha " * 180},
        {"page_number": 2, "text": "beta " * 180},
    ]

    chunks = chunk_pages(
        document_id="test-document",
        pages=pages,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 2
    assert {chunk["page_number"] for chunk in chunks} == {1, 2}
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(chunk["document_id"] == "test-document" for chunk in chunks)


def test_process_document_saves_embeddings(tmp_path, monkeypatch):
    import app.api.documents as documents_api

    document_id = str(uuid4())

    metadata = {
        "document_id": document_id,
        "original_filename": "sample.pdf",
        "page_count": 2,
        "needs_ocr": False,
        "pages": [
            {
                "page_number": 1,
                "text": "Artificial intelligence helps computers learn. " * 20,
                "character_count": 960,
            },
            {
                "page_number": 2,
                "text": "Retrieval systems find relevant documents. " * 20,
                "character_count": 900,
            },
        ],
    }

    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)

    (tmp_path / f"{document_id}.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    def fake_embeddings(texts):
        return [[0.1] * 384 for _ in texts], 384

    monkeypatch.setattr(
        documents_api,
        "generate_embeddings",
        fake_embeddings,
    )

    response = client.post(f"/documents/{document_id}/process")

    assert response.status_code == 200

    result = response.json()
    assert result["chunk_count"] > 0
    assert result["embedding_dimension"] == 384

    saved = json.loads(
        (tmp_path / f"{document_id}_chunks.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(saved["chunks"]) == result["chunk_count"]
    assert all(len(c["embedding"]) == 384 for c in saved["chunks"])
    assert all(c["page_number"] in (1, 2) for c in saved["chunks"])

    assert response.status_code == 200

    data = response.json()
    assert data["page_count"] == 1
    assert data["needs_ocr"] is True
    assert data["pages"][0]["page_number"] == 1
    assert (tmp_path / f'{data["document_id"]}.pdf').exists()
    assert (tmp_path / f'{data["document_id"]}.json').exists()
