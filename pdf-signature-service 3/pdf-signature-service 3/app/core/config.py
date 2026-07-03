"""Runtime configuration — every tunable lives here, sourced from environment.

Override any value via the PDFSIGN_ prefix:
    PDFSIGN_MAX_PDF_BYTES=10485760 uvicorn app.main:app

Or via a .env file (gitignored) for local development.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PDFSIGN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    storage_root: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent / "tmp",
        description="Root directory for per-task working directories.",
    )

    # ------------------------------------------------------------------ #
    # Upload limits
    # ------------------------------------------------------------------ #
    max_pdf_bytes: int = Field(
        default=15 * 1024 * 1024,  # 15 MB
        description="Maximum accepted PDF upload size in bytes.",
    )

    max_image_bytes: int = Field(
        default=5 * 1024 * 1024,  # 5 MB
        description="Maximum accepted signature image size in bytes.",
    )

    read_chunk_bytes: int = Field(
        default=256 * 1024,  # 256 KB
        description="Chunk size for streaming uploads to disk.",
    )

    # ------------------------------------------------------------------ #
    # Task lifecycle
    # ------------------------------------------------------------------ #
    task_ttl_seconds: int = Field(
        default=3600,  # 1 hour
        description="Age in seconds after which orphaned task dirs are purged at startup.",
    )

    # ------------------------------------------------------------------ #
    # API surface
    # ------------------------------------------------------------------ #
    api_prefix: str = Field(default="/api/v1")

    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Tighten in production.",
    )

    app_title: str = "PDF Signature Service"
    app_version: str = "2.0.0"

    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    api_secret_key: str = Field(
        default="",
        description="If non-empty, every request must include "
                    "'Authorization: Bearer <key>'.",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("storage_root", mode="before")
    @classmethod
    def _resolve_storage_root(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @field_validator(
        "max_pdf_bytes",
        "max_image_bytes",
        "read_chunk_bytes",
        "task_ttl_seconds",
        mode="before",
    )
    @classmethod
    def _positive_int(cls, v) -> int:
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            raise ValueError("must be a positive integer")

        if value <= 0:
            raise ValueError("must be a positive integer")

        return value


# Module-level singleton — import this everywhere.
settings = Settings()