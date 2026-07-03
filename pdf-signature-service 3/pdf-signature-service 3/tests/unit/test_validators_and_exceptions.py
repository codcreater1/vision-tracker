"""Unit tests for core validators and exception hierarchy.

These tests have zero I/O and zero network calls — they run in milliseconds
and can be executed without any installed PDF or image libraries.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    AppError,
    BadRequestError,
    CorruptedPdfError,
    InvalidCoordinateError,
    NotFoundError,
    PayloadTooLargeError,
    TaskNotFoundError,
)
from app.core.validators import ImageKind, detect_image_kind, human_readable_size, is_pdf


# ---------------------------------------------------------------------------
# validators.is_pdf
# ---------------------------------------------------------------------------


class TestIsPdf:
    def test_valid_pdf_magic(self) -> None:
        assert is_pdf(b"%PDF-1.7\n") is True

    def test_valid_pdf_magic_exact(self) -> None:
        assert is_pdf(b"%PDF-") is True

    def test_empty_bytes(self) -> None:
        assert is_pdf(b"") is False

    def test_png_is_not_pdf(self) -> None:
        assert is_pdf(b"\x89PNG\r\n\x1a\n") is False

    def test_html_is_not_pdf(self) -> None:
        assert is_pdf(b"<!DOCTYPE html>") is False

    def test_pdf_with_bom_prefix_rejected(self) -> None:
        # We require strict %PDF- at byte 0 — liberal parsers allow BOM prefix.
        assert is_pdf(b"\xef\xbb\xbf%PDF-1.4") is False


# ---------------------------------------------------------------------------
# validators.detect_image_kind
# ---------------------------------------------------------------------------


class TestDetectImageKind:
    _PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    _JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 12
    _JPEG_JFIF = b"\xff\xd8\xff\xe1" + b"\x00" * 12  # EXIF variant

    def test_png_detected(self) -> None:
        assert detect_image_kind(self._PNG_HEADER) is ImageKind.PNG

    def test_jpeg_e0_detected(self) -> None:
        assert detect_image_kind(self._JPEG_HEADER) is ImageKind.JPEG

    def test_jpeg_e1_detected(self) -> None:
        assert detect_image_kind(self._JPEG_JFIF) is ImageKind.JPEG

    def test_pdf_bytes_not_image(self) -> None:
        assert detect_image_kind(b"%PDF-1.7") is None

    def test_empty_bytes(self) -> None:
        assert detect_image_kind(b"") is None

    def test_random_bytes(self) -> None:
        assert detect_image_kind(b"\x00\x01\x02\x03") is None


# ---------------------------------------------------------------------------
# validators.human_readable_size
# ---------------------------------------------------------------------------


class TestHumanReadableSize:
    def test_bytes(self) -> None:
        assert "B" in human_readable_size(512)

    def test_kilobytes(self) -> None:
        result = human_readable_size(2048)
        assert "KB" in result

    def test_megabytes(self) -> None:
        result = human_readable_size(15 * 1024 * 1024)
        assert "MB" in result


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_all_app_errors_have_http_status(self) -> None:
        for cls in [
            BadRequestError,
            NotFoundError,
            PayloadTooLargeError,
            CorruptedPdfError,
            InvalidCoordinateError,
            TaskNotFoundError,
        ]:
            instance = cls()
            assert isinstance(instance.http_status, int)
            assert 400 <= instance.http_status < 600

    def test_custom_detail_overrides_default(self) -> None:
        err = TaskNotFoundError("custom message")
        assert err.detail == "custom message"

    def test_default_detail_used_when_none(self) -> None:
        err = TaskNotFoundError()
        assert err.detail == TaskNotFoundError.default_detail

    def test_task_not_found_is_app_error(self) -> None:
        assert isinstance(TaskNotFoundError(), AppError)

    def test_exception_message_equals_detail(self) -> None:
        err = CorruptedPdfError("bad bytes")
        assert str(err) == "bad bytes"
