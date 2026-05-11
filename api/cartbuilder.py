"""
api/cartbuilder.py — Cart Builder routes ported from project-you-apps/cart-builder Flask app.

This is Phase 1 of the Cart Builder port plan (docs/CARTBUILDER-PORT-PLAN.md).
Mirrors the Flask app's backend routes as FastAPI routes under /api/cartbuilder/*,
delegating to cart-builder's existing parsers / builder / cartridge_builder modules.

Scope reduced 2026-05-04 evening per CRUD-as-own-screen architecture decision:
the /remove, /restore, /replace, /delete routes are NOT ported here — they live
in the future CRUD screen. This module is purely cart-as-package operations
(upload, files, metadata, ingest, pattern0, build, build/status, carts,
cart_folders, browse, load_cart, clear_workspace, has_changes).

The cart-builder source modules don't collide with any VPS local module names,
so we add cart-builder/cart-builder/ to sys.path here once at import time.
The membot importlib pattern in main.py is unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# sys.path bootstrap so cart-builder's parsers.py / builder.py / cartridge_builder.py
# resolve their sibling imports correctly.
# ---------------------------------------------------------------------------

_CART_BUILDER_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "cart-builder", "cart-builder"
))
if os.path.isdir(_CART_BUILDER_SRC) and _CART_BUILDER_SRC not in sys.path:
    sys.path.append(_CART_BUILDER_SRC)

# Late imports — only after sys.path is patched
try:
    from parsers import parse_file, chunk_texts  # type: ignore
    from builder import build_cart_async, get_state  # type: ignore
    _CART_BUILDER_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _CART_BUILDER_AVAILABLE = False
    _IMPORT_ERROR = str(_e)
    print(f"[cartbuilder] WARNING: cart-builder modules unavailable: {_e}")


# ---------------------------------------------------------------------------
# State — workspace registry, dirs, settings
# ---------------------------------------------------------------------------

# Workspace: in-memory file registry. Mirror of cart-builder's files_db.
files_db: dict[str, dict] = {}

# Directories: store under VPS api dir to keep deployment self-contained.
# Cart-builder's defaults were under cart-builder/cart-builder/{uploads,built_carts}.
# We use a workspace dir alongside the VPS cartridges/ folder so VPS users can
# discover their built carts in the existing cartridge picker.
_API_DIR = Path(__file__).resolve().parent
_VPS_ROOT = _API_DIR.parent
UPLOAD_DIR = _VPS_ROOT / "cartridges" / "_cartbuilder_uploads"
BUILD_DIR = _VPS_ROOT / "cartridges"  # Built carts land in cartridges/ for discovery
SETTINGS_FILE = _VPS_ROOT / "cartridges" / "_cartbuilder_settings.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BUILD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers (ports of cart-builder's app.py helpers)
# ---------------------------------------------------------------------------

def _public_file_info(info: dict) -> dict:
    """Strip server-internal fields from a file_db entry before returning to client."""
    return {k: v for k, v in info.items() if k not in ("path", "parsed_chunks")}


def _active_files() -> dict[str, dict]:
    """Return files that haven't been soft-removed."""
    return {fid: info for fid, info in files_db.items() if not info.get("_removed")}


def load_cart_folders() -> list[str]:
    """Return the user's saved cart folders, exactly as configured.

    The saved-folders list is bookmark semantics — the user curates which
    directories they care about for cart browsing. Empty list = empty list;
    we don't re-inject a default folder when the user has removed everything.
    Operators of hosted instances can pre-seed SETTINGS_FILE with whatever
    curated-cart folder they want pinned (along with VPS_READ_ONLY=1 to
    prevent visitors from removing it). Andy 2026-05-10.
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return list(json.load(f).get("cart_folders", []))
        except Exception:
            pass
    return []


def save_cart_folders(folders: list[str]) -> None:
    settings: dict = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                settings = json.load(f)
        except Exception:
            pass
    settings["cart_folders"] = folders
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def _metadata_sidecar_path() -> str:
    """Path for the per-cart metadata sidecar file."""
    if not _CART_BUILDER_AVAILABLE:
        return str(BUILD_DIR / "_workspace_meta.json")
    state = get_state()
    cart_path = state.get("cart_path")
    if cart_path:
        base = os.path.splitext(cart_path)[0]
        if base.endswith(".cart"):
            base = base[:-5]
        return base + ".meta.json"
    return str(BUILD_DIR / "_workspace_meta.json")


def _save_metadata_sidecar() -> None:
    meta: dict[str, dict] = {}
    for info in files_db.values():
        if info.get("owner") or info.get("description") or info.get("tags"):
            meta[info["name"]] = {
                "owner": info.get("owner", ""),
                "description": info.get("description", ""),
                "tags": info.get("tags", []),
            }
    if meta:
        try:
            with open(_metadata_sidecar_path(), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[cartbuilder] Failed to save metadata sidecar: {e}")


def _load_metadata_sidecar(cart_path: str) -> dict:
    base = os.path.splitext(cart_path)[0]
    if base.endswith(".cart"):
        base = base[:-5]
    meta_path = base + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _check_available() -> None:
    """Raise 503 if cart-builder modules failed to import."""
    if not _CART_BUILDER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Cart Builder modules not available: {_IMPORT_ERROR}. "
                   f"Ensure cart-builder/cart-builder/ exists and its dependencies "
                   f"are installed (pip install pymupdf python-docx openpyxl striprtf)."
        )


def _check_writable() -> None:
    """Raise 403 if the server is in global read-only mode (VPS_READ_ONLY env var).
    Mirrors main._enforce_writable but kept local to avoid a circular import."""
    if os.environ.get("VPS_READ_ONLY", "").lower() in ("1", "true", "yes", "on"):
        raise HTTPException(
            status_code=403,
            detail="Server is in read-only mode. Cart Builder writes disabled for the public demo.",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/cartbuilder", tags=["cartbuilder"])


@router.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Multipart file upload. Saves to UPLOAD_DIR with a hashed prefix, parses, registers in files_db."""
    _check_writable()
    _check_available()
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for f in files:
        file_id = uuid.uuid4().hex[:8]
        safe_name = f.filename or "unknown"
        save_path = UPLOAD_DIR / f"{file_id}_{safe_name}"
        contents = await f.read()
        save_path.write_bytes(contents)

        sections = parse_file(save_path)
        chunks = chunk_texts(sections)
        preview = chunks[0]["text"][:200] if chunks else ""

        info = {
            "id": file_id,
            "name": safe_name,
            "type": save_path.suffix.lstrip(".").lower(),
            "size": save_path.stat().st_size,
            "chunks": len(chunks),
            "chars": sum(len(c["text"]) for c in chunks),
            "preview": preview,
            "owner": "",
            "description": "",
            "tags": [],
            "path": str(save_path),
            "parsed_chunks": chunks,
        }
        files_db[file_id] = info
        results.append(_public_file_info(info))

    return {"files": results}


@router.get("/files")
async def list_files():
    """Return current workspace files (active + soft-removed)."""
    return {"files": [_public_file_info(info) for info in files_db.values()]}


@router.post("/metadata")
async def set_metadata(payload: dict):
    """Set owner/description/tags on a workspace file. Persists to .meta.json sidecar."""
    _check_writable()
    file_id = payload.get("file_id")
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="File not found")
    if "owner" in payload:
        files_db[file_id]["owner"] = payload["owner"]
    if "description" in payload:
        files_db[file_id]["description"] = payload["description"]
    if "tags" in payload:
        files_db[file_id]["tags"] = payload["tags"]
    _save_metadata_sidecar()
    return {"ok": True}


@router.post("/ingest")
async def ingest_file(payload: dict):
    """Add a document to the workspace by server-side path (from the folder browser)."""
    _check_writable()
    _check_available()
    file_path = payload.get("path", "")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    filepath = Path(file_path)
    file_id = uuid.uuid4().hex[:8]
    sections = parse_file(filepath)
    chunks = chunk_texts(sections)
    preview = chunks[0]["text"][:200] if chunks else ""

    info = {
        "id": file_id,
        "name": filepath.name,
        "type": filepath.suffix.lstrip(".").lower(),
        "size": filepath.stat().st_size,
        "chunks": len(chunks),
        "chars": sum(len(c["text"]) for c in chunks),
        "preview": preview,
        "owner": "",
        "description": "",
        "tags": [],
        "path": str(filepath),
        "parsed_chunks": chunks,
    }
    files_db[file_id] = info
    return {"file": _public_file_info(info)}


@router.get("/pattern0")
async def pattern0(name: str = "hackathon-cart"):
    """Manifest preview for the current workspace."""
    active = _active_files()
    all_chunks: list = []
    for info in active.values():
        all_chunks.extend(info["parsed_chunks"])
    return {
        "cart_name": name,
        "file_count": len(active),
        "total_chunks": len(all_chunks),
        "files": [
            {"name": info["name"], "chunks": info["chunks"], "owner": info["owner"]}
            for info in active.values()
        ],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@router.post("/build")
async def build(payload: dict):
    """Kick off async cart build from current workspace state."""
    _check_writable()
    _check_available()
    cart_name = payload.get("cart_name", "hackathon-cart")
    cart_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in cart_name)

    active = _active_files()
    all_chunks: list = []
    for info in active.values():
        all_chunks.extend(info["parsed_chunks"])

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No files uploaded")

    file_meta = {}
    for info in active.values():
        file_meta[info["name"]] = {
            "owner": info.get("owner", ""),
            "description": info.get("description", ""),
            "tags": info.get("tags", []),
        }

    # If editing an existing cart, save back to its source dir; else BUILD_DIR.
    state = get_state()
    existing_path = state.get("cart_path")
    if existing_path and os.path.exists(existing_path):
        output_dir = os.path.dirname(existing_path)
    else:
        output_dir = str(BUILD_DIR)

    build_cart_async(all_chunks, cart_name, output_dir, file_metadata=file_meta)
    return {"status": "building", "cart_name": cart_name, "chunks": len(all_chunks)}


@router.get("/build/status")
async def build_status():
    """Long-poll target — returns cart-builder's internal build_state dict."""
    _check_available()
    return get_state()


@router.get("/carts")
async def list_carts(path: str = ""):
    """List carts and subdirectories for a given path (or saved-folders root).

    SECURITY (Andy 2026-05-06): the no-path variant returns carts from the
    SAVED folders only — that's bounded to whatever the operator configured.
    The with-path variant walks the filesystem and is gated by read-only mode
    on public deploys, otherwise it leaks directory structure to anyone who
    knows the URL. Sandbox uploads have their own dir; users don't need
    arbitrary path traversal in a hosted demo.
    """
    if path:
        _check_writable()  # blocks arbitrary filesystem reads on the public droplet
    folders = load_cart_folders()
    DOC_EXTS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".txt", ".rtf"}

    if path:
        browse = Path(path)
        if not browse.is_dir():
            return {
                "carts": [], "subdirs": [], "docs": [],
                "folders": folders, "current_path": path,
            }

        carts = []
        for f in sorted(browse.glob("*.cart.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
            stat = f.stat()
            size_mb = round(stat.st_size / (1024 * 1024), 2)
            manifest_path = f.with_name(f.stem.replace(".cart", "") + ".cart_manifest.json")
            count: object = "?"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as mf:
                        count = json.load(mf).get("count", "?")
                except Exception:
                    pass
            carts.append({
                "name": f.stem.replace(".cart", ""),
                "filename": f.name,
                "size_mb": size_mb,
                "passages": count,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "path": str(f),
            })

        subdirs = []
        try:
            for entry in sorted(os.scandir(str(browse)), key=lambda e: e.name.lower()):
                if entry.is_dir() and not entry.name.startswith("."):
                    subdirs.append({"name": entry.name, "path": str(Path(entry.path).resolve())})
        except PermissionError:
            pass

        docs = []
        try:
            for entry in sorted(os.scandir(str(browse)), key=lambda e: e.name.lower()):
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in DOC_EXTS:
                    stat = entry.stat()
                    docs.append({
                        "name": entry.name,
                        "path": str(Path(entry.path).resolve()),
                        "size": stat.st_size,
                        "type": os.path.splitext(entry.name)[1].lstrip(".").lower(),
                    })
        except PermissionError:
            pass

        return {
            "carts": carts, "subdirs": subdirs, "docs": docs,
            "folders": folders, "current_path": str(browse.resolve()),
        }

    # No path specified — show carts from all saved folders
    carts = []
    seen: set[str] = set()
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
        for f in folder_path.glob("*.cart.npz"):
            if str(f) in seen:
                continue
            seen.add(str(f))
            stat = f.stat()
            size_mb = round(stat.st_size / (1024 * 1024), 2)
            manifest_path = f.with_name(f.stem.replace(".cart", "") + ".cart_manifest.json")
            count: object = "?"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as mf:
                        count = json.load(mf).get("count", "?")
                except Exception:
                    pass
            carts.append({
                "name": f.stem.replace(".cart", ""),
                "filename": f.name,
                "size_mb": size_mb,
                "passages": count,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "path": str(f),
                "folder": folder,
            })
    carts.sort(key=lambda c: c["modified"], reverse=True)
    return {"carts": carts, "subdirs": [], "folders": folders, "current_path": ""}


@router.get("/cart_folders")
async def get_cart_folders():
    return {"folders": load_cart_folders()}


@router.post("/cart_folders")
async def add_cart_folder(payload: dict):
    _check_writable()
    folder = (payload.get("folder") or "").strip()
    if not folder:
        raise HTTPException(status_code=400, detail="No folder provided")
    if not Path(folder).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid folder: {folder}")
    folders = load_cart_folders()
    if folder not in folders:
        folders.append(folder)
        save_cart_folders(folders)
    return {"folders": folders}


@router.delete("/cart_folders")
async def remove_cart_folder(payload: dict):
    _check_writable()
    folder = (payload.get("folder") or "").strip()
    folders = load_cart_folders()
    if folder in folders:
        folders.remove(folder)
        save_cart_folders(folders)
    return {"folders": folders}


@router.post("/build-to-folder")
async def build_to_folder(
    cart: UploadFile = File(...),
    manifest: UploadFile = File(...),
    permissions: UploadFile = File(...),
    folder: str = Form(...),
    cart_name: str = Form(...),
    replace: str = Form("false"),
):
    """Write a browser-built cart bundle to a server-side folder.

    Used by the Edit Carts "New Cart" flow when the user picks a destination
    folder via the server-side FolderPickerModal — we already know the
    absolute path, so the browser POSTs the cart + manifest + permissions
    blobs alongside it, and the server writes all three files in place. The
    user's own permissions sidecar is HONORED here (no forced read-only),
    because we're on a writable instance (gated by _check_writable) and the
    cart is going to a folder the user explicitly picked.

    Distinct from /api/cartridges/upload (sandbox upload that forces
    read-only for the public-demo case). This endpoint is intentionally
    locked off on the droplet via VPS_READ_ONLY=1; the New Cart flow is
    a writable-instance feature.

    Andy 2026-05-10. Returns {cart_path, mounted_filename, folder} so the
    caller can mount + switch to Open Cart mode.
    """
    _check_writable()

    target_dir = Path(folder).resolve()
    if not target_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Destination folder does not exist: {folder}",
        )

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in cart_name)
    if not safe_name:
        safe_name = "new-cart"

    cart_filename = f"{safe_name}.cart.npz"
    manifest_filename = f"{safe_name}.cart_manifest.json"
    permissions_filename = f"{safe_name}.permissions.json"

    cart_path = target_dir / cart_filename
    manifest_path = target_dir / manifest_filename
    permissions_path = target_dir / permissions_filename

    # Prevent silent clobber. Andy 2026-05-10 QA: built a cart with the same
    # name as an existing one in the destination folder and the server
    # overwrote without warning. Frontend handles 409 by prompting the user
    # and retrying with replace=true if they confirm.
    if cart_path.exists() and replace.lower() not in ("true", "1", "yes", "on"):
        raise HTTPException(
            status_code=409,
            detail=f"A cart named '{cart_filename}' already exists in {target_dir}. Set replace=true to overwrite.",
        )

    try:
        cart_bytes = await cart.read()
        cart_path.write_bytes(cart_bytes)

        manifest_bytes = await manifest.read()
        manifest_path.write_bytes(manifest_bytes)

        permissions_bytes = await permissions.read()
        permissions_path.write_bytes(permissions_bytes)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write cart bundle: {e}",
        )

    return {
        "cart_path": str(cart_path),
        "mounted_filename": cart_filename,
        "folder": str(target_dir),
    }


@router.get("/browse")
async def browse_folders(path: str = ""):
    """List subdirectories at a path. Empty path → drive roots on Windows, / on Unix.

    SECURITY (Andy 2026-05-06): always gated by read-only mode. There's no
    legitimate use for a public-demo user to walk the server's filesystem;
    the only purpose was the local-deploy folder picker. On the droplet this
    becomes a reconnaissance vector (list /etc/, /opt/, /root/, etc.). 403.
    """
    _check_writable()
    if not path:
        import platform
        if platform.system() == "Windows":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
            return {"path": "", "dirs": drives, "is_root": True}
        path = "/"

    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Not a directory")

    dirs = []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(entry.name)
    except PermissionError:
        pass

    parent = os.path.dirname(path) if path != os.path.dirname(path) else None
    return {"path": path, "dirs": dirs, "parent": parent, "is_root": False}


@router.post("/load_cart")
async def load_cart(payload: dict):
    """Open an existing cart for editing — populate workspace from cart contents.

    Large carts (>100 sources) get truncated file-card display but remain searchable.
    Restores per-file metadata from .meta.json sidecar if present.
    """
    _check_writable()  # Mutates server-side workspace state (files_db.clear())
    _check_available()
    cart_path = payload.get("cart_path", "")
    if not cart_path or not os.path.exists(cart_path):
        raise HTTPException(status_code=404, detail="Cart not found")

    import numpy as np
    try:
        cart_data = np.load(cart_path, allow_pickle=True)
    except Exception:
        import pickle
        with open(cart_path, "rb") as f:
            pkl = pickle.load(f)

        class FakeNpz:
            def __getitem__(self, key):
                if key == "passages":
                    return np.array(pkl.get("passages", pkl.get("texts", [])), dtype=object)
                if key == "embeddings":
                    return pkl.get("embeddings", np.array([]))
                return pkl.get(key, np.array([]))
        cart_data = FakeNpz()

    passages = list(cart_data["passages"])
    total_passages = len(passages)
    MAX_FILE_CARDS = 100

    source_groups: dict[str, list[dict]] = {}
    for i, passage in enumerate(passages):
        text = str(passage)
        first_line = text.split("\n")[0] if "\n" in text else ""
        body = "\n".join(text.split("\n")[1:]) if "\n" in text else text
        source = first_line.split(" (part ")[0].strip() if " (part " in first_line else first_line.strip()
        if not source:
            source = f"passage_{i}"
        source_groups.setdefault(source, []).append({
            "text": body, "page": None, "source": source, "cart_index": i,
        })

    saved_meta = _load_metadata_sidecar(cart_path)

    files_db.clear()
    cart_files = []
    truncated = len(source_groups) > MAX_FILE_CARDS

    for idx, (source, chunks) in enumerate(source_groups.items()):
        if idx >= MAX_FILE_CARDS:
            break
        file_id = uuid.uuid4().hex[:8]
        preview = chunks[0]["text"][:200] if chunks else ""
        ext = os.path.splitext(source)[1].lstrip(".").lower() if "." in source else "txt"
        file_meta = saved_meta.get(source, {})
        info = {
            "id": file_id,
            "name": source,
            "type": ext,
            "size": sum(len(c["text"]) for c in chunks),
            "chunks": len(chunks),
            "chars": sum(len(c["text"]) for c in chunks),
            "preview": preview,
            "owner": file_meta.get("owner", ""),
            "description": file_meta.get("description", ""),
            "tags": file_meta.get("tags", []),
            "path": "",
            "parsed_chunks": chunks,
            "from_cart": True,
        }
        files_db[file_id] = info
        cart_files.append({k: v for k, v in info.items() if k not in ("path", "parsed_chunks", "from_cart")})

    cart_name = os.path.basename(cart_path).replace(".cart.npz", "").replace(".npz", "").replace(".pkl", "")

    # Sync cart-builder's build_state so /build/status reflects this load
    try:
        from builder import build_state, _lock  # type: ignore
        with _lock:
            build_state["status"] = "done"
            build_state["cart_path"] = cart_path
            build_state["progress"] = 1.0
    except Exception:
        pass  # Cart-builder builder module unavailable — best-effort

    return {
        "ok": True,
        "cart_path": cart_path,
        "cart_name": cart_name,
        "files": cart_files,
        "total_passages": total_passages,
        "total_sources": len(source_groups),
        "truncated": truncated,
        "showing": min(len(source_groups), MAX_FILE_CARDS),
    }


@router.post("/clear_workspace")
async def clear_workspace():
    """Reset workspace state — files, build state."""
    _check_writable()
    files_db.clear()
    try:
        from builder import build_state, _lock  # type: ignore
        with _lock:
            build_state["status"] = "idle"
            build_state["cart_path"] = None
            build_state["progress"] = 0.0
            build_state["chunks_done"] = 0
            build_state["chunks_total"] = 0
    except Exception:
        pass
    return {"ok": True}


@router.get("/has_changes")
async def has_changes():
    """Dirty-flag for unsaved-changes warning."""
    active = _active_files()
    state = get_state() if _CART_BUILDER_AVAILABLE else {}
    has_files = len(active) > 0
    has_built = state.get("status") == "done" and state.get("cart_path")
    return {
        "has_files": has_files,
        "has_built": bool(has_built),
        "file_count": len(active),
        "message": "Changes won't be saved until you click Build/Update Cart." if has_files else "",
    }
