"""Domain exception hierarchy.

Rules:
- Every exception has a `http_status` so routers can map it without importing FastAPI.
- Exceptions carry a `detail` string safe to expose to clients (no tracebacks, no paths).
- Raise the most specific subclass available; catch the base class at the boundary.
"""

from __future__ import annotations

from http import HTTPStatus


class AppError(Exception):
    """Base for all domain errors. Never raise this directly."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    default_detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


# ---------------------------------------------------------------------------
# 400 Bad Request
# ---------------------------------------------------------------------------


class BadRequestError(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    default_detail = "Bad request."


class EmptyUploadError(BadRequestError):
    default_detail = "Uploaded file is empty."


class InvalidPageError(BadRequestError):
    default_detail = "Page index is out of range."


class InvalidCoordinateError(BadRequestError):
    default_detail = "Signature bounding box is outside the page bounds."


# ---------------------------------------------------------------------------
# 404 Not Found
# ---------------------------------------------------------------------------


class NotFoundError(AppError):
    http_status = HTTPStatus.NOT_FOUND
    default_detail = "Resource not found."


class TaskNotFoundError(NotFoundError):
    default_detail = "task_id not found or already expired."


class SignedPdfNotReadyError(NotFoundError):
    default_detail = "Signed PDF is not ready — call POST /sign first."


# ---------------------------------------------------------------------------
# 413 Payload Too Large
# ---------------------------------------------------------------------------


class PayloadTooLargeError(AppError):
    http_status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    default_detail = "Payload exceeds the allowed size limit."


# ---------------------------------------------------------------------------
# 415 Unsupported Media Type
# ---------------------------------------------------------------------------


class UnsupportedMediaTypeError(AppError):
    http_status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    default_detail = "File type is not supported."


class InvalidPdfError(UnsupportedMediaTypeError):
    default_detail = "Uploaded file is not a valid PDF."


class InvalidSignatureImageError(UnsupportedMediaTypeError):
    default_detail = "Signature image must be a valid PNG or JPEG."


# ---------------------------------------------------------------------------
# 422 Unprocessable Entity
# ---------------------------------------------------------------------------


class UnprocessableError(AppError):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    default_detail = "File could not be processed."


class CorruptedPdfError(UnprocessableError):
    default_detail = "PDF is corrupted or cannot be parsed."


class EncryptedPdfError(UnprocessableError):
    default_detail = "Password-protected PDFs are not supported."


class ImageNormalizationError(UnprocessableError):
    default_detail = "Signature image could not be decoded or converted."


# ---------------------------------------------------------------------------
# 500 Internal Server Error
# ---------------------------------------------------------------------------


class StorageError(AppError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    default_detail = "A storage operation failed."


class PdfSaveError(AppError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    default_detail = "Failed to save the signed PDF."
