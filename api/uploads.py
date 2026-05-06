"""
api/uploads.py — Public-demo upload endpoint for client-side cartridge files.

Andy 2026-05-06: when the server is in read-only mode (public droplet demo),
users can't open the server's filesystem via the native file picker, but they
DO want to evaluate VPS against their own carts. Solution: client uploads
a `.cart.npz` or `.pkl` to a sandboxed temp dir; backend forces a read-only
permissions sidecar; user mounts via the existing /api/cartridges/mount path.

Sandboxing layers:
  • Per-file UUID prefix so concurrent uploads don't collide.
  • Filename sanitized — only alphanumeric / dash / dot / underscore in the
    user-visible portion, rest stripped.
  • Size cap (default 250MB; configurable via VPS_UPLOAD_MAX_MB env var).
  • Magic-byte check on the first 4 bytes — NPZ must be PK zip, PKL must be
    a pickle protocol marker. Naive renamed files are rejected.
  • Forced `default: r` permissions sidecar — even if the cart's own sidecar
    inside the NPZ said `rw`, the public demo enforces read-only.
  • TTL cleanup: a background task runs every UPLOAD_CLEANUP_INTERVAL_SEC
    and evicts files older than UPLOAD_TTL_SEC. Files currently mounted
    (engine.mounted_path matches) are skipped.

The endpoint is exempt from _enforce_writable() in main.py — uploads write
ONLY to the sandbox dir, never to the canonical cartridges/ catalog. Mount
still goes through the standard mount path (which doesn't gate on writes).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_DIR = Path(__file__).resolve().parent
_VPS_ROOT = _API_DIR.parent
SANDBOX_DIR = _VPS_ROOT / "cartridges" / "_session_uploads"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("VPS_UPLOAD_MAX_MB", "250"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_TTL_SEC = int(os.environ.get("VPS_UPLOAD_TTL_SEC", "3600"))           # 1 hour
UPLOAD_CLEANUP_INTERVAL_SEC = int(os.environ.get("VPS_UPLOAD_CLEANUP_SEC", "600"))  # 10 min

ALLOWED_EXTS = (".cart.npz", ".npz", ".pkl")

# Magic-byte signatures — first 4 bytes of a valid file.
NPZ_MAGIC = b"PK\x03\x04"   # zip archive (NPZ is a zip)
PKL_MAGICS = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")  # pickle protocol 2-5

# Filename sanitization — keep alphanumerics, dot, dash, underscore. Anything
# else (spaces, slashes, unicode, etc.) is replaced with `_`. This runs on the
# user-visible portion only; the UUID prefix isolates collisions.
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "")  # strip any path traversal attempt
    name = _SAFE_FILENAME.sub("_", name).strip("._-")
    return name or "upload.cart.npz"


def _has_valid_magic(data: bytes, ext: str) -> bool:
    """Validate first bytes match the declared extension."""
    if ext in (".cart.npz", ".npz"):
        return data.startswith(NPZ_MAGIC)
    if ext == ".pkl":
        return any(data.startswith(m) for m in PKL_MAGICS)
    return False


def _ext_for(filename: str) -> str | None:
    fname_lower = filename.lower()
    for ext in ALLOWED_EXTS:
        if fname_lower.endswith(ext):
            return ext
    return None


# ---------------------------------------------------------------------------
# TTL cleanup
# ---------------------------------------------------------------------------

_cleanup_task: asyncio.Task | None = None


def _evict_stale(skip_paths: set[str]) -> int:
    """Synchronous sweep — evict files older than TTL. Returns count evicted."""
    now = time.time()
    evicted = 0
    if not SANDBOX_DIR.exists():
        return 0
    for entry in SANDBOX_DIR.iterdir():
        if not entry.is_file():
            continue
        if str(entry.resolve()) in skip_paths:
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age > UPLOAD_TTL_SEC:
            try:
                entry.unlink()
                evicted += 1
            except OSError as e:
                print(f"[uploads] eviction failed for {entry}: {e}")
    return evicted


async def cleanup_loop(skip_path_provider) -> None:
    """Background task — runs forever, sweeping every UPLOAD_CLEANUP_INTERVAL_SEC."""
    while True:
        try:
            await asyncio.sleep(UPLOAD_CLEANUP_INTERVAL_SEC)
            skip = {str(Path(p).resolve())} if (p := skip_path_provider()) else set()
            n = await asyncio.to_thread(_evict_stale, skip)
            if n > 0:
                print(f"[uploads] TTL evicted {n} file(s) from {SANDBOX_DIR}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[uploads] cleanup loop error: {e}")


def start_cleanup(skip_path_provider) -> None:
    """Launch the cleanup task. Idempotent."""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        return
    loop = asyncio.get_event_loop()
    _cleanup_task = loop.create_task(cleanup_loop(skip_path_provider))


def stop_cleanup() -> None:
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
    _cleanup_task = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/cartridges", tags=["uploads"])


@router.post("/upload")
async def upload_cartridge(file: UploadFile = File(...)):
    """Upload a cart file to the sandbox dir for temporary use.

    The endpoint is intentionally exempt from _enforce_writable() because
    uploads land in a sandbox sub-tree, NEVER in the canonical cartridges/
    catalog. The forced read-only permissions sidecar ensures the cart can't
    be written to once mounted, regardless of any sidecar inside the cart.

    Response shape (success):
      { success: true, message: str, cart_path: str (absolute), size_mb: float,
        ttl_sec: int }
    """
    raw_name = file.filename or "upload"
    ext = _ext_for(raw_name)
    if ext is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTS)}",
        )

    # Read with size cap. Reject early if too large.
    chunks: list[bytes] = []
    total = 0
    head_bytes: bytes | None = None
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        if head_bytes is None:
            head_bytes = chunk[:8]
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large. Max {MAX_UPLOAD_MB}MB.",
            )

    if total == 0 or head_bytes is None:
        raise HTTPException(status_code=400, detail="Empty file")

    if not _has_valid_magic(head_bytes, ext):
        raise HTTPException(
            status_code=400,
            detail=f"File magic bytes don't match extension {ext}. Expected NPZ (zip) or PKL.",
        )

    # Write to sandbox under a unique prefix
    safe_name = _sanitize_filename(raw_name)
    upload_id = uuid.uuid4().hex[:12]
    target = SANDBOX_DIR / f"{upload_id}_{safe_name}"
    try:
        with open(target, "wb") as f:
            for c in chunks:
                f.write(c)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}") from e

    # Force a read-only permissions sidecar — even if the cart claims `rw`,
    # the demo refuses writes against this upload at mount.
    perms_path = SANDBOX_DIR / f"{upload_id}_{safe_name.replace('.cart.npz', '').replace('.npz', '').replace('.pkl', '')}.permissions.json"
    try:
        with open(perms_path, "w", encoding="utf-8") as f:
            json.dump({
                "default": "r",
                "version": "1.0",
                "description": "Sandboxed user upload (TTL temp)",
            }, f, indent=2)
    except OSError as e:
        # Sidecar write failure is non-fatal — engine.cart_permissions will
        # be None and engine.read_only stays True from the global flag anyway.
        print(f"[uploads] permissions sidecar write failed: {e}")

    size_mb = total / (1024 * 1024)
    return {
        "success": True,
        "message": f"Uploaded {safe_name} ({size_mb:.1f} MB) to sandbox",
        "cart_path": str(target.resolve()),
        "size_mb": round(size_mb, 2),
        "ttl_sec": UPLOAD_TTL_SEC,
    }
