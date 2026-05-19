# PDF Signature Service — v2.0 (Production-Ready)

Upload a PDF → place a visual signature at precise coordinates → download the signed file.

## Project structure

```
pdf-signature-service/
├── app/
│   ├── main.py                  # FastAPI factory, middleware, global exc. handlers
│   ├── core/
│   │   ├── config.py            # pydantic-settings (PDFSIGN_* env vars / .env)
│   │   ├── exceptions.py        # domain exception hierarchy (AppError subclasses)
│   │   ├── models.py            # Pydantic request/response schemas
│   │   └── validators.py        # magic-byte file-type predicates (pure functions)
│   ├── services/
│   │   ├── file_service.py      # async streaming upload, size + magic validation
│   │   ├── pdf_service.py       # PyMuPDF: page inspection, signature embedding
│   │   └── storage_service.py   # task-scoped temp directory lifecycle
│   └── routers/
│       └── pdf.py               # HTTP: /upload, /sign, /download/{task_id}
├── tests/
│   ├── unit/
│   │   └── test_validators_and_exceptions.py
│   └── integration/
│       └── test_api.py          # FastAPI TestClient, services mocked via DI
├── tmp/                         # gitignored working directory for task files
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Architecture decisions

| Layer | Responsibility | What it does NOT do |
|---|---|---|
| `routers/` | HTTP in → HTTP out | No business logic |
| `services/` | Business logic | No FastAPI imports |
| `core/` | Shared contracts | No I/O |

**Domain exceptions** live in `core/exceptions.py`. Every `AppError` subclass
carries `http_status` so the router can translate without knowing HTTP details.
The router calls `_raise_http(err)` which produces a clean `HTTPException`.

**Dependency injection** via FastAPI `Depends()` makes every service
swappable in tests — override `get_pdf_service()` with a mock and the
router tests run without PyMuPDF installed.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

## Configuration

All settings are overridable via environment variables prefixed `PDFSIGN_`:

| Variable | Default | Description |
|---|---|---|
| `PDFSIGN_MAX_PDF_BYTES` | `15728640` (15 MB) | Max PDF upload size |
| `PDFSIGN_MAX_IMAGE_BYTES` | `5242880` (5 MB) | Max signature image size |
| `PDFSIGN_STORAGE_ROOT` | `./tmp` | Working directory for tasks |
| `PDFSIGN_TASK_TTL_SECONDS` | `3600` | Orphan cleanup age (startup) |
| `PDFSIGN_CORS_ORIGINS` | `["*"]` | Tighten in production |
| `PDFSIGN_API_SECRET_KEY` | `""` | Bearer token (empty = disabled) |

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
pytest --cov=app --cov-report=term-missing   # with coverage
```

## API reference

### POST /api/v1/pdf/upload
Accepts a multipart PDF. Returns `task_id` + page geometry.

```json
{
  "task_id": "3f8a2b...",
  "page_count": 2,
  "pages": [
    {"index": 0, "width": 595.0, "height": 842.0},
    {"index": 1, "width": 595.0, "height": 842.0}
  ],
  "upload_size_bytes": 204800
}
```

### POST /api/v1/pdf/sign
Embeds a signature image. All coordinates in PDF points (1 pt = 1/72 in).
Origin is page **top-left**; x right, y down.

**Converting pixel clicks to PDF points:**
```js
const x_pt = click_x / canvas_width  * page.width;
const y_pt = click_y / canvas_height * page.height;
```

Form fields: `task_id`, `page` (0-indexed), `x`, `y`, `w`, `h`, `image` (PNG/JPEG).

### GET /api/v1/pdf/download/{task_id}
Streams `signed.pdf`. Schedules task cleanup after the response is sent
(single-use semantics).

## Migrating from v1 prototype

The wire format is backward-compatible with one addition: `/upload` now also
returns `upload_size_bytes`. No endpoint paths changed. The internal module
layout changed significantly:

| v1 file | v2 equivalent |
|---|---|
| `app/processor.py` | `app/services/pdf_service.py` |
| `app/storage.py` | `app/services/storage_service.py` |
| `app/validators.py` | `app/core/validators.py` |
| `app/config.py` | `app/core/config.py` |
| `app/models.py` | `app/core/models.py` |
| `app/routers/pdf.py` | `app/routers/pdf.py` (rewritten) |
