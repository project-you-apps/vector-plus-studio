"""Image Builder client (Day 2).

Thin loopback client around Image Builder's `POST /ocr` endpoint on
`127.0.0.1:7879`. Called from the paired Desktop Cart Builder exe's
upload/build path when a queued file is an image or a scanned PDF.

Auth: shared bearer token at `~/.vector-plus/token` (see
image-builder/auth.py's TOKEN_PATH — same file both Builders read/write).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import httpx

_log = logging.getLogger(__name__)

# Endpoint constants — mirror image-builder/main.py defaults. If the exe
# falls back to a higher port (7880-7888) because 7879 was busy, we'd need
# to probe /health across the range; MVP: single port.
IMAGE_BUILDER_HOST = "127.0.0.1"
IMAGE_BUILDER_PORT_DEFAULT = 7879
IMAGE_BUILDER_ORIGIN_DEFAULT = f"http://{IMAGE_BUILDER_HOST}:{IMAGE_BUILDER_PORT_DEFAULT}"

# Docling sync response. Budget sized for CPU-only laptops (no CUDA):
# EasyOCR runs 10-30x slower on CPU, and force_full_page_ocr rasterizes
# every page of a PDF, so a dense receipt or multi-page deck can easily
# push past 3 min on integrated graphics. 600s covers realistic worst-
# case laptop OCR while still failing loud if the pipeline actually hangs.
# GPU machines finish in seconds and never approach this ceiling.
# Bumped from 180s after integrated-graphics laptops timed out on
# multi-page PDF OCR.
OCR_TIMEOUT_SEC = 600.0

# Token file — same location image-builder/auth.py writes to. Cart Builder
# reads the file at delegation time rather than caching in memory so a
# token rotation between builds doesn't require a helper restart.
TOKEN_PATH = Path.home() / ".vector-plus" / "token"


class ImageBuilderNotRunningError(RuntimeError):
    """POST /ocr received a connection error — no exe on 127.0.0.1:7879."""


class ImageBuilderAuthError(RuntimeError):
    """POST /ocr got 401 — token missing or mismatched."""


class ImageBuilderFailedError(RuntimeError):
    """POST /ocr returned 4xx/5xx that wasn't 401 — Docling refused the input."""


def load_shared_token() -> Optional[str]:
    """Read the shared bearer token from disk. Returns None when absent."""
    if not TOKEN_PATH.exists():
        return None
    try:
        val = TOKEN_PATH.read_text(encoding="utf-8").strip()
        return val or None
    except OSError:
        return None


def health_check(origin: str = IMAGE_BUILDER_ORIGIN_DEFAULT, timeout: float = 1.0) -> bool:
    """Fast HEAD-like check on /health (no auth required). True when the exe
    is reachable and returns 200; False on any error or non-200."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{origin}/health")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


def ocr_file(
    file_bytes: bytes,
    filename: str,
    *,
    mime: Optional[str] = None,
    include_graphics: bool = True,
    include_tables: bool = True,
    max_pages: int = 100,
    origin: str = IMAGE_BUILDER_ORIGIN_DEFAULT,
    token: Optional[str] = None,
) -> dict:
    """POST a file to Image Builder /ocr and return the parsed JSON response.

    Args:
        file_bytes: raw file contents (image or PDF).
        filename: original filename — Image Builder uses the extension to
            pick between image + PDF pipelines.
        mime: optional MIME type; defaults to application/octet-stream when
            we don't know it.
        include_graphics / include_tables / max_pages: /ocr options.
        origin: base URL (default: http://127.0.0.1:7879).
        token: bearer token; falls back to load_shared_token() when None.

    Returns:
        dict with keys `markdown`, `graphics`, `tables`, `source_type`,
        `page_count`, `elapsed_sec` (see image-builder/models.py OcrResponse).

    Raises:
        ImageBuilderNotRunningError: connection refused / DNS / timeout.
        ImageBuilderAuthError: 401 from /ocr (missing/wrong token).
        ImageBuilderFailedError: any other non-2xx response.
    """
    if token is None:
        token = load_shared_token()
    if not token:
        raise ImageBuilderAuthError("No shared bearer token found at ~/.vector-plus/token")

    files = {"file": (filename, file_bytes, mime or "application/octet-stream")}
    data = {
        "options.include_graphics": "1" if include_graphics else "0",
        "options.include_tables": "1" if include_tables else "0",
        "options.max_pages": str(max_pages),
    }
    headers = {"Authorization": f"Bearer {token}"}

    try:
        with httpx.Client(timeout=OCR_TIMEOUT_SEC) as client:
            resp = client.post(f"{origin}/ocr", files=files, data=data, headers=headers)
    except httpx.ConnectError as e:
        raise ImageBuilderNotRunningError(f"Image Builder not reachable on {origin}: {e}") from e
    except httpx.TimeoutException as e:
        raise ImageBuilderFailedError(f"Image Builder OCR timed out after {OCR_TIMEOUT_SEC}s") from e

    if resp.status_code == 401:
        raise ImageBuilderAuthError("Image Builder rejected token (401)")
    if not resp.is_success:
        # Docling errors surface as {"error": "...", "detail": "..."} —
        # bubble the detail up so callers can surface something meaningful
        # to the user instead of a bare status code.
        detail = ""
        try:
            body = resp.json()
            detail = body.get("detail") or body.get("error") or ""
        except Exception:
            detail = resp.text[:200]
        raise ImageBuilderFailedError(f"Image Builder /ocr {resp.status_code}: {detail}")

    return resp.json()


# MIME lookup — mirrors the isImageFile check on the frontend so the
# on-server routing decision stays consistent with what the browser did.
_IMAGE_EXTENSIONS_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".heic": "image/heic",
    ".heif": "image/heic",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def guess_mime(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return "application/pdf"
    return _IMAGE_EXTENSIONS_MIMES.get(ext, "application/octet-stream")
