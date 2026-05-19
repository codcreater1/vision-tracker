"""Pydantic schemas for all API request inputs and response bodies.

Design notes:
- Request bodies for multipart endpoints are NOT modeled here; FastAPI
  parses those via Form()/File() parameters directly in the route function.
- Response models are stable contracts — adding fields is backward-compatible,
  removing or renaming fields is a breaking change requiring a version bump.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / primitive
# ---------------------------------------------------------------------------


class PageGeometry(BaseModel):
    """Dimensions of a single PDF page, in PDF points (1 pt = 1/72 inch)."""

    index: int = Field(..., ge=0, description="0-indexed page number.")
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Returned by POST /pdf/upload."""

    task_id: str = Field(
        ...,
        description="Opaque server-issued handle. Pass to /sign and /download.",
    )
    page_count: int = Field(..., ge=1)
    pages: list[PageGeometry] = Field(
        ...,
        description=(
            "Per-page dimensions the front-end needs to translate "
            "click coordinates into PDF-space coordinates."
        ),
    )
    upload_size_bytes: int = Field(..., ge=1, description="Bytes accepted by the server.")


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


class SignResponse(BaseModel):
    """Returned by POST /pdf/sign."""

    task_id: str
    page: int = Field(..., ge=0)
    download_url: str = Field(..., description="Relative URL ready for GET /download/{task_id}.")


# ---------------------------------------------------------------------------
# Download — no JSON body; endpoint returns FileResponse directly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all 4xx / 5xx responses."""

    detail: str
    error_code: str | None = Field(
        default=None,
        description="Machine-readable code for programmatic error handling.",
    )
