"""PDF signing API — three endpoints.

POST   /pdf/upload             — stream + validate PDF, return task_id
POST   /pdf/sign               — embed signature image at given coordinates
GET    /pdf/download/{task_id} — stream signed PDF, schedule cleanup

The router's sole responsibility is HTTP: parameter extraction, service calls,
and translating domain exceptions (AppError subclasses) into HTTPException.
All business logic lives in the service layer.
"""

from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.exceptions import AppError, SignedPdfNotReadyError, TaskNotFoundError
from app.core.models import (
    ErrorResponse,
    PageGeometry,
    SignResponse,
    UploadResponse,
)
from app.services.file_service import FileService, file_service
from app.services.pdf_service import PdfService, pdf_service
from app.services.storage_service import StorageService, storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["pdf"])

# ---------------------------------------------------------------------------
# Reusable OpenAPI error examples (avoids repetition in every decorator)
# ---------------------------------------------------------------------------
_ERR: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "Bad request."},
    404: {"model": ErrorResponse, "description": "Not found."},
    413: {"model": ErrorResponse, "description": "Payload too large."},
    415: {"model": ErrorResponse, "description": "Unsupported media type."},
    422: {"model": ErrorResponse, "description": "Unprocessable entity."},
    500: {"model": ErrorResponse, "description": "Internal server error."},
}


# ---------------------------------------------------------------------------
# Dependency injection helpers
# ---------------------------------------------------------------------------
# Using Depends() for services makes unit-testing trivial: override the
# dependency in the test client to inject a mock.


def get_file_service() -> FileService:
    return file_service


def get_pdf_service() -> PdfService:
    return pdf_service


def get_storage_service() -> StorageService:
    return storage_service


# ---------------------------------------------------------------------------
# Error translation helper
# ---------------------------------------------------------------------------


def _raise_http(err: AppError) -> None:
    """Translate a domain AppError into an HTTPException with its status."""
    raise HTTPException(status_code=err.http_status, detail=err.detail)


# ---------------------------------------------------------------------------
# POST /pdf/upload
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERR,
    summary="Upload a PDF and receive a task_id with page geometry.",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to sign. Max 15 MB."),
    fs: FileService = Depends(get_file_service),
    ps: PdfService = Depends(get_pdf_service),
    ss: StorageService = Depends(get_storage_service),
) -> UploadResponse:
    """
    Stream the PDF to a temporary task directory, validate its format,
    and return metadata the front-end needs to render the signature placement UI.

    - **task_id**: opaque handle; pass it to `/sign` and `/download`.
    - **pages**: per-page widths and heights in PDF points (72 pt = 1 inch).
      The front-end should use these to convert pixel coordinates from the
      canvas into PDF-space points before calling `/sign`.
    """
    # Fast reject before reading: Content-Length already too large.
    if file.size is not None and file.size > settings.max_pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds the {settings.max_pdf_bytes} byte limit.",
        )

    task_id, task_dir = ss.create_task()
    pdf_path = task_dir / "original.pdf"
    succeeded = False

    try:
        size = await fs.stream_pdf_to_disk(file, pdf_path)

        try:
            pages = ps.describe_pages(pdf_path)
        except AppError as err:
            _raise_http(err)

        succeeded = True

    except AppError as err:
        _raise_http(err)

    finally:
        if not succeeded:
            ss.remove_task(task_id)

    logger.info("Uploaded PDF: task=%s pages=%d size=%d", task_id, len(pages), size)

    return UploadResponse(
        task_id=task_id,
        page_count=len(pages),
        pages=[PageGeometry(index=p.index, width=p.width, height=p.height) for p in pages],
        upload_size_bytes=size,
    )


# ---------------------------------------------------------------------------
# POST /pdf/sign
# ---------------------------------------------------------------------------


@router.post(
    "/sign",
    response_model=SignResponse,
    responses=_ERR,
    summary="Embed a signature image at specific coordinates on a PDF page.",
)
async def sign_pdf(
    task_id: str = Form(..., min_length=8, max_length=64, description="From POST /upload."),
    image: UploadFile = File(..., description="PNG or JPEG signature image."),
    page: int = Form(..., ge=0, description="0-indexed target page number."),
    x: float = Form(..., description="Top-left X of the bounding box, in PDF points."),
    y: float = Form(..., description="Top-left Y of the bounding box, in PDF points."),
    w: float = Form(..., gt=0, description="Bounding box width in PDF points."),
    h: float = Form(..., gt=0, description="Bounding box height in PDF points."),
    fs: FileService = Depends(get_file_service),
    ps: PdfService = Depends(get_pdf_service),
    ss: StorageService = Depends(get_storage_service),
) -> SignResponse:
    """
    Place a signature image on the specified page of a previously uploaded PDF.

    **Coordinate system**: origin is the page *top-left*, x grows right, y grows
    down — the same convention as HTML5 Canvas and CSS. Units are PDF points
    (1 point = 1/72 inch). The `/upload` response returns per-page dimensions
    you can use to convert pixel coordinates:

    ```
    x_pt = click_x_px / canvas_width_px  * page.width
    y_pt = click_y_px / canvas_height_px * page.height
    ```

    The image is fitted inside the (w, h) box while preserving its aspect ratio.
    """
    # Resolve source PDF — raises TaskNotFoundError if task_id is unknown.
    try:
        source = ss.original_path(task_id)
    except TaskNotFoundError as err:
        _raise_http(err)

    output = source.parent / "signed.pdf"

    try:
        image_bytes = await fs.read_signature_image(image)
    except AppError as err:
        _raise_http(err)

    try:
        ps.embed_signature(
            source,
            output,
            image_bytes=image_bytes,
            page_index=page,
            x=x, y=y, w=w, h=h,
        )
    except AppError as err:
        _raise_http(err)
    except Exception:
        logger.exception("Unexpected error while signing task %s", task_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sign PDF.",
        ) from None

    download_url = f"{settings.api_prefix}/pdf/download/{task_id}"
    logger.info("Signed: task=%s page=%d box=(%.1f,%.1f,%.1f,%.1f)", task_id, page, x, y, w, h)

    return SignResponse(task_id=task_id, page=page, download_url=download_url)


# ---------------------------------------------------------------------------
# GET /pdf/download/{task_id}
# ---------------------------------------------------------------------------


@router.get(
    "/download/{task_id}",
    response_class=FileResponse,
    responses=_ERR,
    summary="Stream the signed PDF and schedule task cleanup.",
)
async def download_signed(
    task_id: str,
    background: BackgroundTasks,
    ss: StorageService = Depends(get_storage_service),
) -> FileResponse:
    """
    Stream the signed PDF to the client, then schedule the task directory for
    deletion in a background task.

    **Single-use semantics**: once downloaded, the task and all associated files
    are discarded. Call `/sign` again on a new upload if you need a second copy.
    """
    try:
        path = ss.signed_path(task_id)
    except TaskNotFoundError as err:
        _raise_http(err)

    if not path.is_file():
        _raise_http(SignedPdfNotReadyError())

    # Schedule cleanup *after* the response is fully streamed.
    background.add_task(ss.remove_task, task_id)

    logger.info("Downloading signed PDF: task=%s", task_id)

    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename="signed.pdf",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
