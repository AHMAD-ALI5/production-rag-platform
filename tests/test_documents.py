def test_upload_accepts_blank_pdf_and_flags_ocr(tmp_path, monkeypatch):
    import app.api.documents as documents_api
    import app.services.pdf_service as pdf_service

    # Patch both module-level upload directories.
    monkeypatch.setattr(documents_api, "UPLOAD_DIR", tmp_path)
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

    # Verify that both files were saved in the temporary directory.
    assert (tmp_path / f'{data["document_id"]}.pdf').exists()
    assert (tmp_path / f'{data["document_id"]}.json').exists()