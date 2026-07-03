"""FileService — async streaming upload with validation.

Responsibilities:
- Stream UploadFile to disk in fixed-size chunks (backpressure-friendly).
- Enforce size limits without buffering the entire file in memory.
- Validate file identity via magic-byte check on the first chunk.
- Guarantee atomic failure: partially-written files are removed on error.

This service has zero knowledge of FastAPI routing or HTTP status codes.
It raises domain exceptions from app.core.exceptions; the router translates
those into HTTPException.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import (
    EmptyUploadError,
    InvalidPdfError,
    InvalidSignatureImageError,
    PayloadTooLargeError,
    StorageError,
)
from app.core.validators import detect_image_kind, human_readable_size, is_pdf

logger = logging.getLogger(__name__)


class FileService:
    """Stateless helper — instantiate once and reuse (e.g. via FastAPI DI)."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_pdf_to_disk(
        self,
        upload: UploadFile,
        destination: Path,
    ) -> int:
        """Stream *upload* to *destination*, validating it is a PDF.

        Returns the total number of bytes written.

        Raises:
            EmptyUploadError: if the upload has zero bytes.
            PayloadTooLargeError: if the upload exceeds settings.max_pdf_bytes.
            InvalidPdfError: if the first chunk fails the PDF magic-byte check.
            StorageError: if the disk write fails.
        """
        return await self._stream_to_disk(
            upload=upload,
            destination=destination,
            max_bytes=settings.max_pdf_bytes,
            magic_ok=is_pdf,
            bad_magic_exc=InvalidPdfError(
                f"Uploaded file is not a valid PDF (magic-byte check failed). "
                f"Maximum size: {human_readable_size(settings.max_pdf_bytes)}."
            ),
        )

    async def read_signature_image(self, upload: UploadFile) -> bytes:
        """Read the signature image into memory, with size and magic checks.

        Buffers the whole image because PyMuPDF needs random access. The
        5 MB cap keeps this safe even on constrained servers.

        Returns the raw image bytes.

        Raises:
            EmptyUploadError: if the upload has zero bytes.
            PayloadTooLargeError: if the upload exceeds settings.max_image_bytes.
            InvalidSignatureImageError: if the bytes are not PNG or JPEG.
        """
        # Fast path: reject before reading if Content-Length is already too large.
        if upload.size is not None and upload.size > settings.max_image_bytes:
            raise PayloadTooLargeError(
                f"Signature image exceeds the {human_readable_size(settings.max_image_bytes)} limit."
            )

        # Read one byte more than the limit so we can detect overrun without
        # allocating the full oversized buffer.
        raw = await upload.read(settings.max_image_bytes + 1)
        await upload.close()

        if not raw:
            raise EmptyUploadError()

        if len(raw) > settings.max_image_bytes:
            raise PayloadTooLargeError(
                f"Signature image exceeds the {human_readable_size(settings.max_image_bytes)} limit."
            )

        if detect_image_kind(raw[:16]) is None:
            raise InvalidSignatureImageError(
                "Signature image must be a PNG or JPEG file. "
                "Magic-byte check failed — check the file format."
            )

        return raw

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _stream_to_disk(
        upload: UploadFile,
        destination: Path,
        *,
        max_bytes: int,
        magic_ok,
        bad_magic_exc: Exception,
    ) -> int:
        """Generic streaming helper shared by PDF and (future) other types."""
        total = 0
        first_chunk = True

        try:
            with destination.open("wb") as fh:
                while True:
                    chunk = await upload.read(settings.read_chunk_bytes)
                    if not chunk:
                        break

                    if first_chunk:
                        if not magic_ok(chunk):
                            raise bad_magic_exc
                        first_chunk = False

                    total += len(chunk)
                    if total > max_bytes:
                        raise PayloadTooLargeError(
                            f"Payload exceeds the {human_readable_size(max_bytes)} limit."
                        )

                    fh.write(chunk)

        except (EmptyUploadError, PayloadTooLargeError, type(bad_magic_exc)):
            destination.unlink(missing_ok=True)
            raise

        except OSError as exc:
            destination.unlink(missing_ok=True)
            logger.exception("Disk write failed for upload to %s", destination)
            raise StorageError(f"Failed to persist upload: {exc}") from exc

        finally:
            await upload.close()

        if first_chunk:
            destination.unlink(missing_ok=True)
            raise EmptyUploadError()

        logger.debug("Streamed %d bytes → %s", total, destination)
        return total


# ---------------------------------------------------------------------------
# Module-level singleton for use with FastAPI Depends()
# ---------------------------------------------------------------------------
file_service = FileService()
