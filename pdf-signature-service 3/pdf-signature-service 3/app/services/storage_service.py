"""StorageService — task-scoped temporary directory management.

Each upload creates an isolated directory:

    <storage_root>/<task_id>/
        original.pdf       # written by /upload
        signed.pdf         # written by /sign (atomic rename)

The task_id is a UUID4 hex string — opaque, unguessable, and URL-safe.

All path resolution is centralised here. No other module constructs paths
from user-supplied strings directly, which prevents path-traversal attacks.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Final

from app.core.config import settings
from app.core.exceptions import StorageError, TaskNotFoundError

logger = logging.getLogger(__name__)

ORIGINAL_NAME: Final[str] = "original.pdf"
SIGNED_NAME: Final[str] = "signed.pdf"


class StorageService:
    """Manages per-task working directories under settings.storage_root."""

    # ------------------------------------------------------------------
    # Directory resolution
    # ------------------------------------------------------------------

    def _root(self) -> Path:
        root = settings.storage_root
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _resolve_task_dir(self, task_id: str, *, must_exist: bool) -> Path:
        """Return the directory path for *task_id* with full path-traversal guard.

        Even though task_id is server-issued, the /sign and /download endpoints
        accept it from clients. A malicious client could supply ".." or an
        absolute path. We defend by:
          1. Rejecting ids that contain path separators or ".." literally.
          2. Resolving the candidate path and verifying it is inside storage_root.
        """
        if not task_id or "/" in task_id or "\\" in task_id or ".." in task_id:
            raise TaskNotFoundError()

        root = self._root()
        candidate = (root / task_id).resolve()

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise TaskNotFoundError() from exc

        if must_exist and not candidate.is_dir():
            raise TaskNotFoundError(
                f"task_id '{task_id}' not found or already expired."
            )

        return candidate

    # ------------------------------------------------------------------
    # Public API — task lifecycle
    # ------------------------------------------------------------------

    def create_task(self) -> tuple[str, Path]:
        """Allocate a new task directory.

        Returns:
            (task_id, dir_path) — the caller writes files into dir_path.

        Raises:
            StorageError: if the directory cannot be created.
        """
        task_id = uuid.uuid4().hex
        task_dir = self._root() / task_id
        try:
            task_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise StorageError(f"Could not create task directory: {exc}") from exc
        logger.debug("Created task %s at %s", task_id, task_dir)
        return task_id, task_dir

    def remove_task(self, task_id: str) -> None:
        """Best-effort recursive cleanup — never raises.

        Safe to call from BackgroundTasks even after a crash because we
        catch all exceptions.
        """
        try:
            d = self._resolve_task_dir(task_id, must_exist=False)
            shutil.rmtree(d, ignore_errors=True)
            logger.debug("Removed task directory: %s", task_id)
        except Exception:  # noqa: BLE001
            logger.warning("remove_task(%s) failed silently.", task_id, exc_info=True)

    def purge_stale(self, ttl_seconds: int) -> int:
        """Delete task directories older than *ttl_seconds*.

        Called at application startup to recover disk space lost to crashes
        mid-request.

        Returns the number of directories removed.
        """
        root = self._root()
        cutoff = time.time() - ttl_seconds
        removed = 0

        for child in root.iterdir():
            if child.name.startswith(".") or not child.is_dir():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError as exc:
                logger.warning("purge_stale: could not stat %s: %s", child, exc)

        return removed

    # ------------------------------------------------------------------
    # Public API — path accessors
    # ------------------------------------------------------------------

    def original_path(self, task_id: str) -> Path:
        """Return the path to the uploaded PDF, asserting the task exists."""
        return self._resolve_task_dir(task_id, must_exist=True) / ORIGINAL_NAME

    def signed_path(self, task_id: str) -> Path:
        """Return the path to the signed PDF, asserting the task exists."""
        return self._resolve_task_dir(task_id, must_exist=True) / SIGNED_NAME


# Module-level singleton.
storage_service = StorageService()
