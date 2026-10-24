
from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app

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

    assert response.status_code == 200

    data = response.json()
    assert data["page_count"] == 1
    assert data["needs_ocr"] is True
    assert data["pages"][0]["page_number"] == 1
    assert (tmp_path / f'{data["document_id"]}.pdf').exists()
    assert (tmp_path / f'{data["document_id"]}.json').exists()
