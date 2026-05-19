"""Magic-byte file type validators.

Filename extensions and Content-Type headers are attacker-controlled and
must not be trusted for security decisions. This module identifies file
types exclusively from their leading bytes (magic numbers).

All functions are pure: bytes in → bool/enum out. No I/O, no side effects.
"""

from __future__ import annotations

from enum import Enum


class ImageKind(str, Enum):
    PNG = "png"
    JPEG = "jpeg"


# ---------------------------------------------------------------------------
# Magic signatures
# ---------------------------------------------------------------------------

# PDF: "%PDF-" — we require strict conformance; some liberal parsers allow
# up to ~1KB of garbage before the marker but we reject those to keep our
# parser attack surface narrow.
_PDF_MAGIC = b"%PDF-"

# PNG: 8-byte signature
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# JPEG: SOI marker FF D8, followed by FF (any APP marker variant)
_JPEG_PREFIX = b"\xff\xd8\xff"


def is_pdf(head: bytes) -> bool:
    """Return True if *head* starts with the PDF magic number."""
    return head[:5] == _PDF_MAGIC


def detect_image_kind(head: bytes) -> ImageKind | None:
    """Return the ImageKind for *head*, or None if unrecognised.

    Only PNG and JPEG are accepted; PyMuPDF supports more formats, but
    restricting the surface reduces risk from obscure format parsers.
    """
    if head[:8] == _PNG_MAGIC:
        return ImageKind.PNG
    if head[:3] == _JPEG_PREFIX:
        return ImageKind.JPEG
    return None


def human_readable_size(n_bytes: int) -> str:
    """Format *n_bytes* as a human-readable string (e.g. '14.3 MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"
