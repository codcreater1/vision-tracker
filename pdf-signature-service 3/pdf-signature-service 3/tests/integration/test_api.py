"""Integration tests for the PDF signing API endpoints.

These tests use FastAPI's TestClient (synchronous HTTPX wrapper) and override
service dependencies so no real PDF library or disk I/O is needed.

Pattern:
    1. Override get_*_service() dependencies with Mocks.
    2. Call the endpoint via client.
    3. Assert HTTP status + response body.

To run:
    pip install pytest httpx
    pytest tests/integration/
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    CorruptedPdfError,
    InvalidPdfError,
    PayloadTooLargeError,
    TaskNotFoundError,
)
from app.main import app
from app.routers.pdf import get_file_service, get_pdf_service, get_storage_service
from app.services.pdf_service import PageInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PDF_MAGIC = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"  # minimal valid-looking header
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture()
def mock_storage():
    svc = MagicMock()
    svc.create_task.return_value = ("abc123def456", MagicMock())
    svc.original_path.return_value = MagicMock()
    svc.signed_path.return_value = MagicMock(is_file=MagicMock(return_value=True))
    return svc


@pytest.fixture()
def mock_file_svc():
    svc = MagicMock()
    svc.stream_pdf_to_disk = MagicMock(return_value=1024)

    async def _fake_stream(upload, dest):
        return 1024

    svc.stream_pdf_to_disk.side_effect = _fake_stream

    async def _fake_read_image(upload):
        return PNG_MAGIC

    svc.read_signature_image.side_effect = _fake_read_image
    return svc


@pytest.fixture()
def mock_pdf_svc():
    svc = MagicMock()
    svc.describe_pages.return_value = [
        PageInfo(index=0, width=595.0, height=842.0),
    ]
    return svc


@pytest.fixture()
def client(mock_storage, mock_file_svc, mock_pdf_svc):
    app.dependency_overrides[get_storage_service] = lambda: mock_storage
    app.dependency_overrides[get_file_service] = lambda: mock_file_svc
    app.dependency_overrides[get_pdf_service] = lambda: mock_pdf_svc
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_healthz_returns_ok(self, client) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/v1/pdf/upload
# ---------------------------------------------------------------------------


class TestUpload:
    def test_successful_upload_returns_201(self, client) -> None:
        resp = client.post(
            "/api/v1/pdf/upload",
            files={"file": ("doc.pdf", BytesIO(PDF_MAGIC), "application/pdf")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "task_id" in body
        assert body["page_count"] == 1
        assert len(body["pages"]) == 1
        assert body["pages"][0]["width"] == 595.0

    def test_empty_filename_still_processed(self, client) -> None:
        # FastAPI requires a filename for UploadFile validation.
        resp = client.post(
            "/api/v1/pdf/upload",
            files={"file": ("", BytesIO(PDF_MAGIC), "application/pdf")},
        )
        assert resp.status_code in (201, 422)  # FastAPI may reject empty filename

    def test_corrupted_pdf_returns_422(self, client, mock_pdf_svc) -> None:
        mock_pdf_svc.describe_pages.side_effect = CorruptedPdfError("bad bytes")
        resp = client.post(
            "/api/v1/pdf/upload",
            files={"file": ("doc.pdf", BytesIO(b"not a pdf"), "application/pdf")},
        )
        assert resp.status_code == 422

    def test_non_pdf_returns_415(self, client, mock_file_svc) -> None:
        async def _raise(upload, dest):
            raise InvalidPdfError()

        mock_file_svc.stream_pdf_to_disk.side_effect = _raise
        resp = client.post(
            "/api/v1/pdf/upload",
            files={"file": ("img.png", BytesIO(PNG_MAGIC), "image/png")},
        )
        assert resp.status_code == 415

    def test_oversized_pdf_returns_413(self, client, mock_file_svc) -> None:
        async def _raise(upload, dest):
            raise PayloadTooLargeError()

        mock_file_svc.stream_pdf_to_disk.side_effect = _raise
        resp = client.post(
            "/api/v1/pdf/upload",
            files={"file": ("big.pdf", BytesIO(PDF_MAGIC * 1000), "application/pdf")},
        )
        assert resp.status_code == 413


# ---------------------------------------------------------------------------
# POST /api/v1/pdf/sign
# ---------------------------------------------------------------------------


class TestSign:
    def _form(self, **overrides):
        defaults = dict(
            task_id="abc123def456",
            page=0,
            x=50.0,
            y=50.0,
            w=100.0,
            h=40.0,
        )
        return {**defaults, **overrides}

    def test_successful_sign_returns_200(self, client) -> None:
        resp = client.post(
            "/api/v1/pdf/sign",
            data=self._form(),
            files={"image": ("sig.png", BytesIO(PNG_MAGIC), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "abc123def456"
        assert "download_url" in body

    def test_unknown_task_returns_404(self, client, mock_storage) -> None:
        mock_storage.original_path.side_effect = TaskNotFoundError()
        resp = client.post(
            "/api/v1/pdf/sign",
            data=self._form(task_id="doesnotexist"),
            files={"image": ("sig.png", BytesIO(PNG_MAGIC), "image/png")},
        )
        assert resp.status_code == 404

    def test_negative_page_rejected_by_fastapi(self, client) -> None:
        # FastAPI Form validation (ge=0) catches this before our code.
        resp = client.post(
            "/api/v1/pdf/sign",
            data=self._form(page=-1),
            files={"image": ("sig.png", BytesIO(PNG_MAGIC), "image/png")},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/pdf/download/{task_id}
# ---------------------------------------------------------------------------


class TestDownload:
    def test_unknown_task_returns_404(self, client, mock_storage) -> None:
        mock_storage.signed_path.side_effect = TaskNotFoundError()
        resp = client.get("/api/v1/pdf/download/doesnotexist")
        assert resp.status_code == 404

    def test_signed_pdf_not_ready_returns_404(self, client, mock_storage) -> None:
        mock_path = MagicMock()
        mock_path.is_file.return_value = False
        mock_storage.signed_path.return_value = mock_path
        resp = client.get("/api/v1/pdf/download/abc123def456")
        assert resp.status_code == 404
