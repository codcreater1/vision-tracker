"""FastAPI application factory.

Responsibilities of this module:
- Create the FastAPI app and attach middleware.
- Register the global (last-resort) exception handler.
- Define the lifespan hook (startup cleanup, future shutdown hooks).
- Mount routers.

Nothing in this file contains business logic — that lives in services/.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import AppError
from app.core.models import ErrorResponse
from app.routers import pdf as pdf_router
from app.services.storage_service import storage_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pdfsign")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks.

    Startup: purge task directories older than TTL (recovers from crashes).
    Shutdown: intentionally a no-op — live task dirs belong to in-flight
    downloads; let the OS handle them on next startup via the purge.
    """
    removed = storage_service.purge_stale(settings.task_ttl_seconds)
    if removed:
        logger.info("Startup: purged %d stale task director(ies).", removed)
    else:
        logger.info("Startup: no stale task directories found.")

    yield  # app runs here

    logger.info("Shutdown: exiting cleanly.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "Upload a PDF, place a visual signature at precise coordinates, "
            "and download the signed file — all via a clean REST API."
        ),
        lifespan=lifespan,
        # Disable default /docs redirect; access Swagger at /docs directly.
        redirect_slashes=False,
    )

    # ------------------------------------------------------------------ #
    # Middleware
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["Content-Disposition"],
    )

    # ------------------------------------------------------------------ #
    # Global exception handlers
    # ------------------------------------------------------------------ #

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Translate domain AppError subclasses into structured JSON responses.

        Domain exceptions should not reach here in normal operation — the router
        catches them via _raise_http(). This handler is a safety net for any
        AppError that escapes the router (e.g. raised inside middleware).
        """
        logger.warning(
            "Domain error on %s %s: [%s] %s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(detail=exc.detail).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Last-resort handler — catches bugs, never leaks tracebacks to clients."""
        logger.exception(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(detail="Internal server error.").model_dump(),
        )

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #
    app.include_router(pdf_router.router, prefix=settings.api_prefix)

    # ------------------------------------------------------------------ #
    # Meta endpoints
    # ------------------------------------------------------------------ #

    @app.get("/healthz", tags=["meta"], summary="Liveness probe.")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": settings.app_version}

    @app.get("/readyz", tags=["meta"], summary="Readiness probe.")
    async def readyz() -> dict[str, str]:
        """Verify that the storage root is writable before accepting traffic."""
        try:
            probe = settings.storage_root / ".readyz"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": str(exc)},
            )
        return {"status": "ok", "storage": str(settings.storage_root)}

    return app


# Module-level instance used by uvicorn.
app = create_app()
