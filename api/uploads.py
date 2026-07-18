"""
api/uploads.py — Public-demo upload endpoint for client-side cartridge files.

When the server is in read-only mode (public droplet demo), users can't
open the server's filesystem via the native file picker, but they DO
want to evaluate VPS against their own carts. Solution: client uploads
a `.cart.npz` to a sandboxed temp dir; backend forces a read-only
permissions sidecar; user mounts via the existing /api/cartridges/mount
path.

Sandboxing layers:
  • Per-file UUID prefix so concurrent uploads don't collide.
  • Filename sanitized — only alphanumeric / dash / dot / underscore in the
    user-visible portion, rest stripped.
  • Size cap (default 250MB; configurable via VPS_UPLOAD_MAX_MB env var).
  • Magic-byte check on the first 4 bytes — NPZ must be PK zip. Naive
    renamed files are rejected.
  • Deep structural NPZ validation: testzip() integrity
    check + zip-slip defense (no `..`, no absolute paths, no backslashes
    in entry names) + zip-bomb defense (per-entry compression ratio cap +
    total uncompressed size cap) + entry-type allowlist (`.npy` only).
  • `.pkl` uploads INTENTIONALLY DROPPED from the public demo. Legacy pkl
    carts unpickle at mount time which is RCE-on-mount territory for
    untrusted uploads. Private deploys can re-enable by adding `.pkl` to
    ALLOWED_EXTS, but the public droplet keeps it off.
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
import io
import json
import os
import re
import time
import uuid
import zipfile
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

# Public-demo upload allowlist. .pkl is intentionally OFF here — pickle
# deserialization at mount time is an RCE vector when content is untrusted.
# Private deploys can re-enable by extending this tuple.
ALLOWED_EXTS = (".cart.npz", ".npz")

# Magic-byte signatures — first 4 bytes of a valid file.
NPZ_MAGIC = b"PK\x03\x04"   # zip archive (NPZ is a zip)

# Deep NPZ validation tunables.
# Per-entry compression ratio cap — anything above this is treated as a
# zip-bomb attempt. 200x is generous for legitimate text content; numpy
# arrays compress at 2-10x typically.
NPZ_MAX_COMPRESSION_RATIO = 200
# Total-uncompressed-size cap — refuse NPZs that expand to more than this.
# Ratio'd against the upload size cap so a 250MB upload can't expand into
# multi-GB on disk.
NPZ_MAX_UNCOMPRESSED_BYTES = MAX_UPLOAD_BYTES * 8

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
    return False


def _ext_for(filename: str) -> str | None:
    fname_lower = filename.lower()
    for ext in ALLOWED_EXTS:
        if fname_lower.endswith(ext):
            return ext
    return None


def _validate_npz_structure(source) -> None:
    """Deep structural validation of an NPZ container.

    `source` accepts:
      • bytes — validates an in-memory blob (legacy callers)
      • Path / str — validates a file on disk via streaming reads (preferred;
        keeps memory footprint low when validating large uploads)

    Defends against:
      • Corrupt zips (testzip catches CRC mismatches).
      • Zip-slip (entry names with `..`, absolute paths, or backslashes —
        could escape the extraction dir if anyone ever extracts the zip).
      • Zip-bombs (per-entry compression ratio + total uncompressed size).
      • Smuggled non-numpy content (entries that aren't `.npy`).

    Raises HTTPException(400 or 413) on any violation. Caller is responsible
    for cleaning up any disk artifact on failure.
    """
    zip_target = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with zipfile.ZipFile(zip_target, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Corrupt zip entry: {bad!r}",
                )

            total_uncompressed = 0
            for info in zf.infolist():
                name = info.filename

                # zip-slip defense — reject anything that could escape an
                # extraction root if the zip were ever extracted to disk.
                if name.startswith("/") or name.startswith("\\"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Absolute path in zip entry: {name!r}",
                    )
                if "\\" in name:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Backslash in zip entry: {name!r}",
                    )
                parts = [p for p in name.split("/") if p]
                if any(p == ".." for p in parts):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Path traversal in zip entry: {name!r}",
                    )

                # entry-type allowlist — NPZ files should ONLY contain .npy
                # entries (numpy.savez_compressed semantics). Anything else
                # is suspicious — reject rather than try to whitelist edge
                # cases (executable smuggling, web-shells, etc.).
                if not name.lower().endswith(".npy"):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unexpected entry type in NPZ: {name!r}. "
                            f"Only .npy entries are allowed."
                        ),
                    )

                # zip-bomb defense — per-entry ratio + running total.
                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > NPZ_MAX_COMPRESSION_RATIO:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Zip entry {name!r} compression ratio {ratio:.0f}x "
                                f"exceeds cap of {NPZ_MAX_COMPRESSION_RATIO}x."
                            ),
                        )
                total_uncompressed += info.file_size
                if total_uncompressed > NPZ_MAX_UNCOMPRESSED_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Total uncompressed size exceeds "
                            f"{NPZ_MAX_UNCOMPRESSED_BYTES // (1024 * 1024)}MB cap."
                        ),
                    )
    except zipfile.BadZipFile as e:
        raise HTTPException(
            status_code=400,
            detail=f"File is not a valid ZIP archive: {e}",
        ) from e


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


@router.delete("/eject")
async def eject_cartridge(cart_path: str):
    """Immediately delete a sandboxed upload + its permissions sidecar.

    Privacy/control feature: users who uploaded a sensitive
    cart shouldn't have to wait up to 1h for TTL eviction. This endpoint
    deletes the file on demand.

    Safety:
      • Only files inside SANDBOX_DIR can be ejected (path-resolution check).
        Attempts on the canonical cartridges/ catalog or any other path 403.
      • Refuses if the file is currently mounted — caller must unmount first
        (avoids OS-specific mounted-file delete confusion).

    Response: { success: true, ejected: <absolute path> }
    """
    target = Path(cart_path).resolve()
    sandbox_resolved = SANDBOX_DIR.resolve()
    try:
        target.relative_to(sandbox_resolved)
    except ValueError as e:
        raise HTTPException(
            status_code=403,
            detail="Eject is only allowed for sandboxed uploads.",
        ) from e

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file.")

    # Refuse if currently mounted — caller must unmount first.
    # We import lazily to avoid a circular import with main.py.
    try:
        from .engine import engine  # type: ignore
        mounted = getattr(engine, "mounted_path", None)
        if mounted and Path(mounted).resolve() == target:
            raise HTTPException(
                status_code=409,
                detail="Cart is currently mounted. Unmount it first, then eject.",
            )
    except ImportError:
        # engine not importable in this context (tests etc.) — skip the check.
        pass

    # Compute the sidecar's predicted path. Mirrors the naming used in
    # upload_cartridge() above. If the sidecar isn't where we expect, swallow
    # the failure — the cart file delete is the primary operation.
    base = target.name
    for suffix in (".cart.npz", ".npz"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    perms_path = target.with_name(f"{base}.permissions.json")

    try:
        target.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Eject failed: {e}") from e

    if perms_path.exists():
        try:
            perms_path.unlink()
        except OSError as e:
            print(f"[uploads] eject: sidecar unlink failed {perms_path}: {e}")

    return {"success": True, "ejected": str(target)}


@router.post("/upload")
async def upload_cartridge(file: UploadFile = File(...)):
    """Upload a cart file to the sandbox dir for temporary use.

    The endpoint is intentionally exempt from _enforce_writable() because
    uploads land in a sandbox sub-tree, NEVER in the canonical cartridges/
    catalog. The forced read-only permissions sidecar ensures the cart can't
    be written to once mounted, regardless of any sidecar inside the cart.

    Streaming write architecture: reads in 64KB chunks
    directly to disk rather than buffering the full upload in memory. Memory
    footprint per concurrent upload drops from MAX_UPLOAD_MB (250 MB) to
    64 KB, so the droplet survives N concurrent demo-day uploads without
    OOM. Deep structural validation runs on the disk file via zipfile's
    streaming reads (also low-memory). Failed validation = unlink + raise
    so we never leave a malformed cart in the sandbox after a rejection.

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

    # Pre-allocate target path so we can stream-write directly.
    safe_name = _sanitize_filename(raw_name)
    upload_id = uuid.uuid4().hex[:12]
    target = SANDBOX_DIR / f"{upload_id}_{safe_name}"

    # Stream to disk in 64 KB chunks. Track size + grab the magic-byte head
    # for the format check we'll do after streaming completes.
    total = 0
    head_bytes: bytes | None = None
    cap_exceeded = False
    write_error: str | None = None
    try:
        with open(target, "wb") as f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                if head_bytes is None:
                    head_bytes = chunk[:8]
                f.write(chunk)
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    cap_exceeded = True
                    break
    except OSError as e:
        write_error = str(e)

    if cap_exceeded:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large. Max {MAX_UPLOAD_MB}MB.",
        )
    if write_error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Write failed: {write_error}")
    if total == 0 or head_bytes is None:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file")

    if not _has_valid_magic(head_bytes, ext):
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File magic bytes don't match extension {ext}. Expected NPZ (zip).",
        )

    # Deep structural validation — streams from disk via zipfile (low
    # memory). Unlink on failure so a malformed cart never persists in the
    # sandbox after a rejection.
    try:
        _validate_npz_structure(target)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    # Force a read-only permissions sidecar — even if the cart claims `rw`,
    # the demo refuses writes against this upload at mount.
    perms_path = SANDBOX_DIR / f"{upload_id}_{safe_name.replace('.cart.npz', '').replace('.npz', '')}.permissions.json"
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
