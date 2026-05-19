"""PdfService — PDF inspection and coordinate-based signature embedding.

This module is intentionally framework-agnostic: it imports nothing from
FastAPI. All errors are domain exceptions from app.core.exceptions.

# Coordinate system contract
PDF user-space has origin (0,0) at the page bottom-left, y growing upward.
PyMuPDF's page.rect, however, uses a top-left origin with y growing downward
(the same convention used by HTML5 Canvas, CSS, and most UI frameworks). The
public API of this module expects coordinates in PyMuPDF / UI convention so
callers do not need to perform a flip. If you ever switch PDF backends, the
_to_fitz_rect() function is the single place to update.

# Image normalisation
Signature images are re-encoded to RGBA PNG before embedding. This:
  - Strips EXIF / ICC metadata (reduces file size, removes exif-based exploits)
  - Defangs polyglot files (the normalised bytes came through PIL's decode)
  - Preserves alpha so the signature does not paint a white box over the document
  - Guarantees PyMuPDF receives a well-formed stream regardless of input format
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as _e:
    fitz = None  # type: ignore[assignment]  # tests run without PyMuPDF
from PIL import Image, UnidentifiedImageError

from app.core.exceptions import (
    CorruptedPdfError,
    EncryptedPdfError,
    ImageNormalizationError,
    InvalidCoordinateError,
    InvalidPageError,
    InvalidSignatureImageError,
    PdfSaveError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageInfo:
    """Immutable page geometry returned to callers."""

    index: int
    width: float   # PDF points
    height: float  # PDF points


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_pdf(path: Path) -> fitz.Document:
    """Open and validate a PDF; raises domain exceptions on bad input."""
    try:
        doc = fitz.open(str(path))
    except (fitz.FileDataError, RuntimeError) as exc:
        raise CorruptedPdfError(f"PDF could not be parsed: {exc}") from exc

    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise EncryptedPdfError()

    if doc.page_count < 1:
        doc.close()
        raise CorruptedPdfError("PDF contains no pages.")

    return doc


def _normalize_image(raw: bytes) -> bytes:
    """Decode *raw*, verify its integrity, and re-encode as RGBA PNG.

    Two passes are intentional:
      Pass 1 (verify): structural check that raises on truncated/corrupt data.
      Pass 2 (load + convert): decodes pixels; img.verify() rewinds the stream
        but leaves it in a partially-consumed state, so we must reopen.

    Raises:
        InvalidSignatureImageError: if PIL cannot decode the file at all.
        ImageNormalizationError: if re-encoding to RGBA PNG fails.
    """
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidSignatureImageError(
            f"Signature image could not be decoded: {exc}"
        ) from exc

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except (OSError, ValueError) as exc:
        raise ImageNormalizationError(
            f"Signature image could not be converted to RGBA PNG: {exc}"
        ) from exc


def _to_fitz_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    page_rect: fitz.Rect,
) -> fitz.Rect:
    """Convert a UI-convention bounding box to a fitz.Rect with bounds checking.

    Args:
        x, y:  Top-left corner of the signature box in PDF points.
               Origin is page top-left, y grows downward (UI / PyMuPDF convention).
        w, h:  Width and height in PDF points. Must both be positive.
        page_rect: The target page's bounding rect (from page.rect).

    Returns:
        fitz.Rect(x0, y0, x1, y1) ready for page.insert_image().

    Raises:
        InvalidCoordinateError: dimension ≤ 0, or box is outside the page.
    """
    if not (w > 0 and h > 0):
        raise InvalidCoordinateError(
            f"Signature box dimensions must be positive; got w={w:.2f}, h={h:.2f}."
        )

    x0, y0 = float(x), float(y)
    x1, y1 = x0 + float(w), y0 + float(h)
    pw, ph = page_rect.width, page_rect.height

    # Allow 0.5 pt of floating-point slack without admitting real overflow.
    eps = 0.5
    if x0 < -eps or y0 < -eps or x1 > pw + eps or y1 > ph + eps:
        raise InvalidCoordinateError(
            f"Signature box ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f}) "
            f"is outside the page bounds (0,0)-({pw:.2f},{ph:.2f}). "
            f"All coordinates must be in PDF points with origin at the page top-left."
        )

    # Clamp within [0, page] after the slack check.
    return fitz.Rect(
        max(0.0, min(x0, pw)),
        max(0.0, min(y0, ph)),
        max(0.0, min(x1, pw)),
        max(0.0, min(y1, ph)),
    )


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------


class PdfService:
    """Stateless PDF processing service — safe to reuse across requests."""

    def describe_pages(self, path: Path) -> list[PageInfo]:
        """Return per-page dimensions for the PDF at *path*.

        The front-end needs these to translate pixel click coordinates into
        PDF-space points before calling /sign.

        Raises:
            CorruptedPdfError, EncryptedPdfError: see _open_pdf().
        """
        doc = _open_pdf(path)
        try:
            return [
                PageInfo(index=i, width=doc[i].rect.width, height=doc[i].rect.height)
                for i in range(doc.page_count)
            ]
        finally:
            doc.close()

    def embed_signature(
        self,
        source_pdf: Path,
        output_pdf: Path,
        *,
        image_bytes: bytes,
        page_index: int,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        """Embed *image_bytes* on *page_index* of *source_pdf*, save to *output_pdf*.

        Coordinate contract:
            x, y — top-left of the signature bounding box, in PDF points,
                    using page top-left as origin with y growing downward.
            w, h — box width and height in PDF points (both must be > 0).

        Save options (garbage=3, deflate=True, clean=True):
            - Drop orphan objects and duplicate streams (garbage=3).
            - zlib-compress all streams (deflate=True).
            - Normalise PDF syntax (clean=True).
            The document's existing bookmarks, layers, and annotations on
            untouched pages are preserved.

        Atomicity:
            Written to a sibling .part file and renamed on success so a crash
            mid-save cannot leave a partially-written signed.pdf.

        Raises:
            InvalidSignatureImageError: PIL structural check failed.
            ImageNormalizationError: RGBA conversion failed.
            InvalidPageError: page_index out of range.
            InvalidCoordinateError: box outside page bounds.
            PdfSaveError: OS-level write failure.
            CorruptedPdfError, EncryptedPdfError: bad source PDF.
        """
        safe_image = _normalize_image(image_bytes)
        doc = _open_pdf(source_pdf)

        part_path = output_pdf.with_suffix(output_pdf.suffix + ".part")

        try:
            if not (0 <= page_index < doc.page_count):
                raise InvalidPageError(
                    f"Page index {page_index} is out of range. "
                    f"This PDF has {doc.page_count} page(s), indexed 0–{doc.page_count - 1}."
                )

            page = doc[page_index]
            rect = _to_fitz_rect(x, y, w, h, page.rect)

            try:
                page.insert_image(
                    rect,
                    stream=safe_image,
                    keep_proportion=True,  # preserve aspect ratio inside the box
                    overlay=True,          # render above existing content
                )
            except (ValueError, RuntimeError) as exc:
                raise ImageNormalizationError(
                    f"PyMuPDF rejected the signature image: {exc}"
                ) from exc

            try:
                doc.save(
                    str(part_path),
                    garbage=3,
                    deflate=True,
                    clean=True,
                )
            except (RuntimeError, OSError) as exc:
                raise PdfSaveError(f"Failed to save signed PDF: {exc}") from exc

        finally:
            doc.close()

        # Atomic publish — os.replace() is atomic on POSIX and on Windows for
        # files not open in another process.
        part_path.replace(output_pdf)
        logger.info(
            "Signed PDF written: task=%s page=%d box=(%.1f,%.1f,%.1f,%.1f)",
            output_pdf.parent.name,
            page_index,
            x, y, w, h,
        )


# Module-level singleton.
pdf_service = PdfService()
