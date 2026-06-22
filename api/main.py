"""
Vector+ Studio 1.0 -- FastAPI Backend

Start with:
    cd vector-plus-studio-repo
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import os
import re
import sqlite3
import time
import threading
import numpy as np

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Sentry error monitoring (Andy 2026-05-12).
# DSN comes from SENTRY_DSN env var; gracefully no-ops if unset (local dev).
# send_default_pii is ON — we collect email/avatar/user UUID which Andy
# explicitly accepts as low-risk. The before_send hook scrubs auth headers,
# JWT-pattern strings, and Bearer tokens before transmission so we don't
# accidentally leak credentials in error context.
# ---------------------------------------------------------------------------
_SENTRY_DSN = os.environ.get("SENTRY_DSN")
if _SENTRY_DSN:
    import sentry_sdk

    _JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
    _BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9_.\-]+", re.IGNORECASE)
    _SENSITIVE_KEYS = {
        "authorization", "cookie", "x-api-key",
        "supabase_jwt_secret", "supabase_service_role_key",
        "sentry_dsn",  # don't echo our own DSN back to Sentry
    }

    def _sentry_scrub_string(s):
        if not isinstance(s, str):
            return s
        s = _JWT_RE.sub("[REDACTED-JWT]", s)
        s = _BEARER_RE.sub("Bearer [REDACTED]", s)
        return s

    def _sentry_scrub_walk(obj):
        if isinstance(obj, str):
            return _sentry_scrub_string(obj)
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                    obj[k] = "[REDACTED]"
                else:
                    obj[k] = _sentry_scrub_walk(obj[k])
            return obj
        if isinstance(obj, list):
            return [_sentry_scrub_walk(item) for item in obj]
        return obj

    def _sentry_before_send(event, hint):
        try:
            return _sentry_scrub_walk(event)
        except Exception:
            # If scrubbing itself errors, drop the event rather than leak.
            return None

    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        send_default_pii=True,
        before_send=_sentry_before_send,
        traces_sample_rate=0.0,
        environment=os.environ.get("VPS_ENVIRONMENT", "production"),
    )
    print("[Sentry] initialized with DSN scrubbing")
else:
    print("[Sentry] SENTRY_DSN not set — error monitoring disabled")

# Global read-only mode for public deploys (e.g. droplet demo).
# Set VPS_READ_ONLY=1 in the environment to refuse all unlock + write
# endpoints regardless of the per-cart engine.read_only state. This is
# Step 1 of the RWX roadmap (Andy 2026-05-05): a coarse server-wide
# lock for the public demo. Step 2 will replace it with cart-format
# RWX bits in the hippocampus row + Pattern 0 manifest.
READ_ONLY_MODE = os.environ.get("VPS_READ_ONLY", "").lower() in ("1", "true", "yes", "on")
if READ_ONLY_MODE:
    print("[VPS] READ_ONLY_MODE active (VPS_READ_ONLY env var set). All writes refused.")


def _enforce_writable(idx: int | None = None):
    """Raise 403 if writes are disallowed at any level.

    Three layers compose, checked in priority order:
      1. Server-wide VPS_READ_ONLY env var (Step 1) — public-deploy gate.
      2. Cart-level permissions sidecar (Step 2a) — cart self-declares its
         RWX via .permissions.json. Default `r` blocks all writes.
      3. Pattern-level flags byte in the hippocampus row (Step 2b) — when a
         specific pattern idx is supplied, check its `w` bit. Allows a
         `default: rw` cart to lock individual patterns down.

    Pass idx for endpoints that target a specific pattern (DELETE, restore,
    /update). Omit it for endpoints that affect the cart as a whole
    (save, unlock, add new, etc.).
    """
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail="Server is in read-only mode. Writes disabled for the public demo.",
        )
    if engine.mounted_name and not cart_permits_write(engine.cart_permissions):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Cart '{engine.mounted_name}' is read-only by its permissions sidecar "
                f"(default={(engine.cart_permissions or {}).get('default', 'r')!r}). "
                f"Edit the cart's .permissions.json to allow writes."
            ),
        )
    if idx is not None and engine.hippocampus is not None and 0 <= idx < len(engine.hippocampus):
        entry = engine.hippocampus[idx]
        if not pattern_permits_write(entry):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Pattern #{idx} is locked by its hippocampus flags "
                    f"(perms={entry.get('perms', {}).get('raw'):#x}). "
                    f"Use bin/set_pattern_permissions.py to flip the bit."
                ),
            )

from .engine import engine, TRAIN_SETTLE_FRAMES, SIG_SETTLE_FRAMES, TextRegionEncoder, TrainingEncoder
from .models import (
    MountRequest, SearchRequest, AddPassageRequest,
    CartridgeInfo, CartridgeListResponse, MountResponse,
    SearchResult, SearchResponse, StatusResponse,
    DeletedPattern, DeletedListResponse, MessageResponse,
    PatternResponse, PatternListItem, PatternListResponse,
    MemboxLockState, MemboxCartInfo, MemboxWriteEntry, MemboxStatus,
    MemboxCartListResponse, MemboxImprintRequest,
    MemboxMountRequest, MemboxUnmountRequest,
)

# Import membox + multi_cart from the sibling membot/ repo without polluting sys.path.
# We use importlib so membot's directory is NEVER added to sys.path -- this avoids
# shadowing VPS-local modules like multi_lattice_wrapper_v7.py that exist in BOTH repos.
import importlib.util as _ilu
import os as _os
_membot_dir = _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), '..', '..', 'membot'
))

def _load_membot_module(name: str):
    """Load a membot top-level .py file by absolute path, isolated from sys.path."""
    path = _os.path.join(_membot_dir, f"{name}.py")
    if not _os.path.isfile(path):
        return None
    spec = _ilu.spec_from_file_location(f"_membot_{name}", path)
    if spec is None or spec.loader is None:
        return None
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print(f"[VPS] Membox loader: membot dir = {_membot_dir}")
try:
    # Strategy: temporarily prepend membot/ to sys.path JUST for this import
    # block so cross-imports between membot's top-level files resolve correctly.
    # Then immediately remove it to avoid shadowing VPS-local modules
    # (e.g. multi_lattice_wrapper_v7.py exists in BOTH repos).
    #
    # We then verify our isolation by inspecting the loaded modules' __file__
    # attributes -- they should all point at the membot/ directory.
    import sys as _sys
    _path_added = False
    if _membot_dir not in _sys.path:
        _sys.path.insert(0, _membot_dir)
        _path_added = True
    try:
        # Drop any cached versions of these names that might shadow us
        for _name in ("multi_cart", "membot_server", "membox", "federate"):
            _sys.modules.pop(_name, None)

        import multi_cart as _multi_cart  # noqa: E402
        import membot_server as _membot_server  # noqa: E402
        import membox as _membox  # noqa: E402

        _MEMBOX_AVAILABLE = True
        print(f"[VPS] Membox loaded OK from {_membox.__file__}")
    finally:
        # IMPORTANT: remove membot from sys.path so VPS's own modules
        # (multi_lattice_wrapper_v7.py, region_fill_encoder.py, etc.) are not shadowed.
        if _path_added and _membot_dir in _sys.path:
            _sys.path.remove(_membot_dir)
except Exception as _membox_err:
    import traceback
    _membox = None
    _multi_cart = None
    _membot_server = None
    _MEMBOX_AVAILABLE = False
    print(f"[VPS] Membox not available: {_membox_err}")
    traceback.print_exc()
from .cartridge_io import (
    list_cartridges as _list_cartridges, load_cartridge, load_signatures,
    find_cartridge_path, find_companion_file, validate_brain_manifest,
    save_brain_manifest, save_signatures, DATA_DIR, parse_hippocampus,
    load_cart_permissions, cart_permits_write, pattern_permits_write,
)
from .search import search as do_search
from .forge import forge_cartridge
from . import cartbuilder
from . import uploads as uploads_mod


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.boot()
    # Pre-warm embedder in background so first search is fast
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, engine.load_embedder)
    # Sandbox-upload TTL cleanup — provider returns the currently-mounted
    # path so the sweep doesn't evict a file the user is actively using.
    uploads_mod.start_cleanup(lambda: engine.mounted_path)
    yield
    uploads_mod.stop_cleanup()
    engine.shutdown()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vector+ Studio",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                    "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cart Builder Phase 1 — port of cart-builder Flask app as /api/cartbuilder/*
# (15 backend routes; CRUD scope deferred to its own screen per 2026-05-05 decision).
app.include_router(cartbuilder.router)

# Sandbox-upload endpoint for the public demo's "Open Cartridge" picker.
# Bypasses _enforce_writable() — uploads land in cartridges/_session_uploads/
# only, never the canonical catalog. Forced r-only permissions sidecar on
# every upload. TTL cleanup runs from the lifespan task.
app.include_router(uploads_mod.router)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    # Detect whether the currently-mounted cart lives inside the upload
    # sandbox — the UI uses this to expose an "Eject" button. Path-resolved
    # relative_to() is the canonical zero-traversal check.
    mounted_is_sandboxed = False
    mounted_path = engine.mounted_path
    if mounted_path:
        try:
            from pathlib import Path as _Path
            from .uploads import SANDBOX_DIR as _SANDBOX
            target = _Path(mounted_path).resolve()
            target.relative_to(_SANDBOX.resolve())
            mounted_is_sandboxed = True
        except (ValueError, OSError):
            pass

    return StatusResponse(
        engine_ready=engine.engine_ready,
        gpu_available=engine.gpu_available,
        mounted_cartridge=engine.mounted_name,
        mounted_path=mounted_path,
        mounted_is_sandboxed=mounted_is_sandboxed,
        pattern_count=len(engine.passages),
        physics_trained=engine.physics_trained,
        training_active=engine.training_active,
        training_progress=engine.training_progress,
        training_total=engine.training_total,
        multimodal=engine.multimodal_mode,
        signatures_loaded=engine.signatures_loaded,
        deleted_count=len(engine.deleted_ids),
        dirty=engine.dirty,
        read_only=engine.read_only or READ_ONLY_MODE,
        read_only_mode=READ_ONLY_MODE,
        cart_permissions=engine.cart_permissions,
    )


# ---------------------------------------------------------------------------
# Cartridges
# ---------------------------------------------------------------------------

@app.get("/api/cartridges", response_model=CartridgeListResponse)
async def get_cartridges():
    carts = _list_cartridges()
    items = []
    known_names = set()
    for c in carts:
        known_names.add(c["name"])
        items.append(CartridgeInfo(
            name=c["name"],
            filename=c["filename"],
            size_mb=c["size_mb"],
            has_brain=c["has_brain"],
            has_signatures=c["has_signatures"],
            has_manifest=c["has_manifest"],
        ))

    # Include externally-mounted cartridge if not already in the scanned list
    if engine.mounted_name and engine.mounted_path and engine.mounted_name not in known_names:
        ext = os.path.splitext(engine.mounted_path)[1].lower()
        try:
            size_mb = round(os.path.getsize(engine.mounted_path) / (1024 * 1024), 1)
        except OSError:
            size_mb = 0.0
        items.insert(0, CartridgeInfo(
            name=engine.mounted_name,
            filename=engine.mounted_path,
            size_mb=size_mb,
            has_brain=engine.physics_trained,
            has_signatures=engine.signatures_loaded,
            has_manifest=False,
        ))

    return CartridgeListResponse(cartridges=items)


@app.get("/api/cartridges/{cart_name}/brain")
async def download_brain(cart_name: str):
    """Stream a cart's brain file (_brain.npy) for browser-side WebGPU physics.

    Public download, no auth — public-demo carts are already listed without JWT,
    and the brain weights are useless without the embeddings/passages they pair with.
    Path traversal is prevented by find_companion_file, which only searches
    whitelisted cartridge directories.
    """
    brain_path = find_companion_file(cart_name, "_brain.npy")
    if not brain_path or not os.path.exists(brain_path):
        raise HTTPException(status_code=404, detail=f"Brain file not found for '{cart_name}'")
    return FileResponse(
        brain_path,
        media_type="application/octet-stream",
        filename=f"{cart_name}_brain.npy",
    )


@app.get("/api/cartridges/{cart_name}/embeddings")
async def download_embeddings(cart_name: str):
    """Stream a cart's .cart.npz (embeddings + passages) for browser-side use.

    Same access model as the brain endpoint. The .cart.npz contains the
    embedding matrix and passage records the WebGPU engine needs to score
    candidates after settle.
    """
    for suffix in (".cart.npz", ".pkl"):
        cart_path = find_companion_file(cart_name, suffix)
        if cart_path and os.path.exists(cart_path):
            return FileResponse(
                cart_path,
                media_type="application/octet-stream",
                filename=f"{cart_name}{suffix}",
            )
    raise HTTPException(status_code=404, detail=f"Cart file not found for '{cart_name}'")


# ---------------------------------------------------------------------------
# Cart embeddings cache — lets the cart_name-keyed routes serve queries
# without requiring the cart be mounted in the engine. Tiny LRU so memory
# stays bounded across multiple cart lookups in one session.
# ---------------------------------------------------------------------------
_CART_CACHE: dict[str, tuple[np.ndarray, list[str]]] = {}
_CART_CACHE_ORDER: list[str] = []
_CART_CACHE_MAX = 3


def _load_cart_cached(cart_name: str) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings, passages) for a cart, mounting-free.

    Prefers the engine's already-loaded copy when the cart is mounted (avoids
    a redundant disk read). Otherwise loads via load_cartridge / a small NPZ
    fall-through and caches the result.
    """
    if engine.mounted_name == cart_name and engine.embeddings is not None and engine.passages:
        return engine.embeddings, engine.passages

    cached = _CART_CACHE.get(cart_name)
    if cached is not None:
        return cached

    pkl_path = find_companion_file(cart_name, ".pkl")
    npz_path = find_companion_file(cart_name, ".cart.npz")

    emb: np.ndarray | None = None
    passages: list[str] | None = None

    if pkl_path and os.path.exists(pkl_path):
        cart_data = load_cartridge(pkl_path)
        if cart_data and cart_data.get("embeddings") is not None:
            emb = cart_data["embeddings"]
            passages = list(cart_data["passages"])

    if emb is None and npz_path and os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        if "embeddings" in data.files:
            emb = data["embeddings"]
            if "passages" in data.files:
                passages = [str(p) for p in data["passages"]]
            elif "snippets" in data.files:
                passages = [str(s) for s in data["snippets"]]
        data.close()

    if emb is None or passages is None:
        raise HTTPException(status_code=404, detail=f"Cart '{cart_name}' not found or has no embeddings")

    _CART_CACHE[cart_name] = (emb, passages)
    _CART_CACHE_ORDER.append(cart_name)
    while len(_CART_CACHE_ORDER) > _CART_CACHE_MAX:
        evict = _CART_CACHE_ORDER.pop(0)
        _CART_CACHE.pop(evict, None)
    return emb, passages


@app.post("/api/embed")
async def embed_query(payload: dict):
    """Return the Nomic embedding for a query string.

    Used by browser-side WebGPU Associate: the browser sends the user's text,
    we run the SentenceTransformer query encoder server-side (cheap, ~80 ms),
    and return the 768-dim vector. Browser then runs the per-candidate physics
    settle locally on its own GPU.
    """
    text = payload.get("query")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Missing 'query' (non-empty string)")
    emb = engine.embed_query(text).astype(np.float32)
    return {"embedding": emb.tolist(), "dim": int(emb.shape[0])}


def _cosine_candidates(emb_list, pool_size, embeddings, passages):
    """Shared cosine-prefilter implementation used by both /mounted and /{name} routes."""
    if not isinstance(emb_list, list):
        raise HTTPException(status_code=400, detail="Missing 'embedding' (list of floats)")
    q = np.asarray(emb_list, dtype=np.float32)
    if q.shape[0] != embeddings.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dim mismatch: got {q.shape[0]}, expected {embeddings.shape[1]}",
        )
    q_norm = q / (float(np.linalg.norm(q)) + 1e-9)
    e_norms = np.sqrt(np.einsum('ij,ij->i', embeddings, embeddings)) + 1e-9
    cosine_sims = (embeddings @ q_norm) / e_norms

    pool_size = min(pool_size, len(embeddings))
    pool_idx = np.argpartition(-cosine_sims, pool_size - 1)[:pool_size]
    pool_idx = pool_idx[np.argsort(-cosine_sims[pool_idx])]

    candidates = []
    for idx in pool_idx:
        idx_int = int(idx)
        candidates.append({
            "idx": idx_int,
            "cosine_score": float(cosine_sims[idx_int]),
            "embedding": embeddings[idx_int].astype(np.float32).tolist(),
            "passage": passages[idx_int],
        })
    return {"candidates": candidates, "total_embeddings": len(embeddings)}


@app.get("/api/cartridges/mounted/embedding/{idx}")
async def get_embedding_by_index(idx: int):
    """Return a single embedding + passage from the currently-mounted cart.

    Convenience wrapper around the cart_name-keyed route; serves the test
    pages that don't know which cart is mounted.
    """
    if engine.embeddings is None or not engine.passages:
        raise HTTPException(status_code=400, detail="No cart mounted")
    if idx < 0 or idx >= len(engine.embeddings):
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range (0..{len(engine.embeddings) - 1})")
    emb = engine.embeddings[idx].astype(np.float32).tolist()
    return {"idx": idx, "embedding": emb, "passage": engine.passages[idx], "dim": len(emb)}


@app.post("/api/cartridges/mounted/cosine-candidates")
async def cosine_candidate_pool_mounted(payload: dict):
    """Cosine pre-filter on the mounted cart. Kept for back-compat with the
    test pages; production UI uses the cart_name-keyed route below."""
    if engine.embeddings is None or not engine.passages:
        raise HTTPException(status_code=400, detail="No cart mounted")
    pool_size = int(payload.get("pool_size", 50))
    return _cosine_candidates(payload.get("embedding"), pool_size, engine.embeddings, engine.passages)


@app.get("/api/cartridges/{cart_name}/embedding/{idx}")
async def get_cart_embedding_by_index(cart_name: str, idx: int):
    """Mount-free version: lazy-loads the cart from disk (cached) and returns
    a single embedding + passage. Lets the WebGPU Associate path survive
    backend hot-reloads and engine restarts."""
    embeddings, passages = _load_cart_cached(cart_name)
    if idx < 0 or idx >= len(embeddings):
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range (0..{len(embeddings) - 1})")
    emb = embeddings[idx].astype(np.float32).tolist()
    return {"idx": idx, "embedding": emb, "passage": passages[idx], "dim": len(emb)}


@app.post("/api/cartridges/{cart_name}/cosine-candidates")
async def cosine_candidate_pool_for_cart(cart_name: str, payload: dict):
    """Mount-free cosine pre-filter for the named cart. Loads embeddings on
    demand with a small LRU cache so the WebGPU Associate path doesn't
    depend on which cart the engine happens to have mounted."""
    embeddings, passages = _load_cart_cached(cart_name)
    pool_size = int(payload.get("pool_size", 50))
    return _cosine_candidates(payload.get("embedding"), pool_size, embeddings, passages)


@app.post("/api/walk-from")
async def walk_from_idx(payload: dict):
    """Server-side "Walk from here" — runs Associate using the embedding at
    the given (cart_name, idx) as the query. Mirrors what the WebGPU Walk
    path does in the browser. Server-side path requires the cart be mounted
    (because associate_search wires through the GPU engine's loaded brain).

    Returns results in the same enriched shape as /api/search so the
    frontend can render them through the standard ResultCard.
    """
    from api.search import associate_search

    cart_name = payload.get("cart_name")
    idx = payload.get("idx")
    top_k = int(payload.get("top_k", 10))
    keywords = payload.get("keywords") or None

    if engine.embeddings is None or not engine.passages:
        raise HTTPException(status_code=400, detail="No cart mounted")
    if cart_name and engine.mounted_name != cart_name:
        raise HTTPException(
            status_code=400,
            detail=f"Walk-from anchor cart '{cart_name}' is not the mounted cart '{engine.mounted_name}'.",
        )
    if not isinstance(idx, int) or idx < 0 or idx >= len(engine.embeddings):
        raise HTTPException(status_code=404, detail=f"Index {idx} out of range")

    q_emb = engine.embeddings[idx].astype(np.float32)
    results = associate_search(q_emb, engine.embeddings, engine.passages, top_k=top_k, keywords=keywords)

    # Enrich with title/preview/full_text/prev_idx/next_idx — matches the
    # same shape /api/search produces (the loop at the end of search.py:do_search).
    hippo = engine.hippocampus
    enriched = []
    for rank, r in enumerate(results):
        ridx = r['idx']
        text = r.get('recovered_text') or (engine.passages[ridx] if ridx < len(engine.passages) else "")
        lines = text.splitlines() if text else ["[empty]"]
        title = lines[0][:100]
        preview = " ".join(lines[1:3])[:200] if len(lines) > 1 else ""
        prev_idx = hippo[ridx].get('prev') if (hippo is not None and ridx < len(hippo)) else None
        next_idx = hippo[ridx].get('next') if (hippo is not None and ridx < len(hippo)) else None
        perms = hippo[ridx].get('perms') if (hippo is not None and ridx < len(hippo)) else None
        enriched.append({
            "rank": rank + 1,
            "idx": ridx,
            "score": r['score'],
            "cosine_score": r.get('cosine_score'),
            "physics_score": r.get('physics_score'),
            "hamming_score": r.get('hamming_score'),
            "keyword_boost": r.get('keyword_boost'),
            "title": title,
            "preview": preview,
            "full_text": text or "",
            "from_lattice": r.get('from_lattice', False),
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "perms": perms,
        })

    return {
        "query_idx": idx,
        "query_passage": engine.passages[idx][:200],
        "results": enriched,
    }


@app.get("/api/browse")
async def browse_for_file():
    """Open a native OS file picker and return the selected path."""
    path = await asyncio.to_thread(_open_file_dialog)
    return {"path": path or ""}


def _open_file_dialog() -> str:
    """Open a native file dialog via PowerShell (reliable from any thread on Windows)."""
    try:
        import subprocess
        script = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '$f = New-Object System.Windows.Forms.OpenFileDialog; '
            '$f.Title = "Open Cartridge"; '
            '$f.Filter = "Cartridge files (*.pkl)|*.pkl|Brain files (*.npy)|*.npy|Signature files (*.npz)|*.npz|All files (*.*)|*.*"; '
            'if ($f.ShowDialog() -eq "OK") { $f.FileName }'
        )
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', script],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        print(f"[Browse] Selected: '{path}' (stderr: {result.stderr.strip()!r})")
        return path
    except Exception as e:
        print(f"[Browse] File dialog error: {e}")
        return ""


def _apply_cart_permissions_after_mount(cart_path: str | None) -> None:
    """Step 2a: load the cart's permissions sidecar after a successful mount.

    Called after every mount helper. If the cart self-declares 'r' default,
    engine.read_only stays True and unlock requests will 403 (via _enforce_writable).
    """
    if not cart_path:
        return
    perms = load_cart_permissions(cart_path)
    engine.cart_permissions = perms
    if perms is not None:
        if not cart_permits_write(perms):
            engine.read_only = True
        print(f"[Mount] cart_permissions loaded: default={perms.get('default')!r}")


async def _dispatch_mount(helper_fn, *args) -> MountResponse:
    """Run a mount helper in a thread and apply cart permissions on success."""
    resp = await asyncio.to_thread(helper_fn, *args)
    if resp.success:
        _apply_cart_permissions_after_mount(engine.mounted_path)
    return resp


@app.post("/api/cartridges/mount", response_model=MountResponse)
async def mount_cartridge(req: MountRequest):
    # Unmount current if any
    engine.unmount()

    filename = req.filename
    print(f"[Mount] filename='{filename}' isabs={os.path.isabs(filename)} exists={os.path.exists(filename)}")

    # Support full file paths (from the Open dialog)
    if os.path.isabs(filename) and os.path.exists(filename):
        basename = os.path.basename(filename)
        ext = os.path.splitext(basename)[1].lower()

        if ext == '.npz':
            # Could be Studio signatures (_signatures.npz) or membot cartridge (.cart.npz)
            # Peek inside to detect format
            try:
                probe = np.load(filename, allow_pickle=True)
                probe_keys = list(probe.keys())
                probe.close()
            except Exception:
                probe_keys = []

            if 'embeddings' in probe_keys and 'passages' in probe_keys:
                # Membot-format cartridge: embeddings + passages (like a PKL)
                cart_name = basename.replace('.cart.npz', '').replace('.npz', '')
                return await _dispatch_mount(_mount_membot_npz, filename, cart_name)
            else:
                # Studio signatures format
                cart_name = basename
                for suffix in ('_signatures.npz', '.npz'):
                    if cart_name.endswith(suffix):
                        cart_name = cart_name[:-len(suffix)]
                        break
                return await _dispatch_mount(_mount_brain_by_path, filename, cart_name)

        if ext == '.npy':
            cart_name = basename
            for suffix in ('_brain.npy', '.npy'):
                if cart_name.endswith(suffix):
                    cart_name = cart_name[:-len(suffix)]
                    break
            return await _dispatch_mount(_mount_brain_by_path, filename, cart_name)

        cart_name = os.path.splitext(basename)[0]
        return await _dispatch_mount(_mount_pkl_by_path, filename, cart_name)

    is_brain_only = filename.endswith("(brain only)")
    cart_name = filename.replace(" (brain only)", "").replace(".pkl", "")

    if is_brain_only:
        return await _dispatch_mount(_mount_brain_only, cart_name)
    else:
        return await _dispatch_mount(_mount_pkl, filename, cart_name)


def _sqlite_fetch_passages(conn: sqlite3.Connection, indices: list) -> dict:
    """Fetch full passages from a split-cart SQLite sidecar by index.
    Returns {idx: {"passage": str, "title": str, "paper_id": str}}.

    LOAD-BEARING: indices must be Python ints (not numpy.int64) — sqlite3 binds
    numpy.int64 silently as a no-match value, returning zero rows. Callers MUST
    cast via int(...) at the call site. Same fix that landed in membot 2026-05-04.
    """
    if not conn or not indices:
        return {}
    placeholders = ",".join("?" for _ in indices)
    rows = conn.execute(
        f"SELECT idx, passage, title, paper_id FROM passages WHERE idx IN ({placeholders})",
        indices
    ).fetchall()
    return {r[0]: {"passage": r[1], "title": r[2], "paper_id": r[3]} for r in rows}


def _mount_membot_npz(full_path: str, cart_name: str) -> MountResponse:
    """Mount a membot-format .cart.npz (embeddings + passages, like a PKL).

    Supports two cart shapes:
      - Standard: NPZ contains 'passages' field with full text per pattern.
      - Split-cart: NPZ contains 'has_sqlite=True' + 'snippets' (200-char preview
        per pattern) + 'text_db' (sidecar filename). Full text lives in the
        SQLite sidecar at <cart_dir>/<text_db>, fetched on-demand at query time
        via /api/patterns/{idx}. Mirrors membot's split-cart support — same
        format, same NPZ keys, same SQLite schema (idx, passage, title, paper_id).
    """
    try:
        data = np.load(full_path, allow_pickle=True)
        emb = data['embeddings']

        # Detect split-cart via has_sqlite flag (membot convention)
        has_sqlite = bool(data['has_sqlite']) if 'has_sqlite' in data.files else False

        if has_sqlite:
            # Split cart: snippets in NPZ, full text in SQLite sidecar
            snippets_raw = data['snippets']
            txt = [str(s) for s in snippets_raw]
            db_filename = str(data['text_db']) if 'text_db' in data.files else None
        else:
            # Standard cart: full passages in NPZ
            passages_raw = data['passages']
            txt = [str(p) for p in passages_raw]
            db_filename = None

        hippo = parse_hippocampus(data)
        data.close()
    except Exception as e:
        return MountResponse(success=False, message=f"Failed to load membot cartridge: {e}")

    cart_dir = os.path.dirname(full_path)

    engine.mounted_name = cart_name
    engine.mounted_path = full_path
    engine.embeddings = emb
    engine.passages = txt
    engine.compressed_lens = []
    engine.compressed_texts = []
    engine.multimodal_mode = False
    engine.brain_only_mode = False
    engine.deleted_ids = set()
    engine.physics_trained = False
    engine.signatures_loaded = False
    engine.training_progress = len(emb)
    engine.training_total = len(emb)

    # Load hippocampus navigation data if present
    engine.hippocampus = hippo

    # Open split-cart SQLite sidecar if present
    engine.is_split_cart = False
    engine.sqlite_conn = None
    engine.sqlite_db_path = None
    if has_sqlite and db_filename:
        db_path = os.path.join(cart_dir, db_filename)
        if os.path.exists(db_path):
            try:
                engine.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
                engine.sqlite_db_path = db_path
                engine.is_split_cart = True
                print(f"[mount] Split cart: SQLite sidecar opened at {db_path}")
            except Exception as e:
                print(f"[mount] Split cart sidecar open failed: {e}")
        else:
            print(f"[mount] Split cart sidecar not found at {db_path}")

    message_parts = [f"{len(txt)} patterns", f"from {cart_dir}"]
    if engine.is_split_cart:
        message_parts.append("split-cart (SQLite sidecar)")
    if hippo is not None:
        n_linked = sum(1 for h in hippo if h.get("prev") is not None or h.get("next") is not None)
        message_parts.append(f"Hippo: {n_linked} linked")

    # Check for companion signatures
    sig_candidates = [
        os.path.join(cart_dir, f"{cart_name}.sigs.npz"),
        os.path.join(cart_dir, f"{cart_name}_signatures.npz"),
    ]
    for sig_path in sig_candidates:
        if os.path.exists(sig_path):
            sig_data = load_signatures(sig_path)
            if sig_data and sig_data['n_patterns'] == len(emb):
                engine.signatures = sig_data['signatures']
                engine.signatures_loaded = True
                message_parts.append("Sigs loaded")
                break

    return MountResponse(
        success=True,
        name=cart_name,
        pattern_count=len(txt),
        multimodal=False,
        brain_loaded=False,
        signatures_loaded=engine.signatures_loaded,
        message=" | ".join(message_parts),
    )


def _mount_pkl_by_path(full_path: str, cart_name: str) -> MountResponse:
    """Mount a cartridge from an absolute file path."""
    cart_dir = os.path.dirname(full_path)

    # Look for companion files next to the pkl
    brain_path = os.path.join(cart_dir, f"{cart_name}_brain.npy")
    sig_path = os.path.join(cart_dir, f"{cart_name}_signatures.npz")

    cart_data = load_cartridge(full_path)
    if not cart_data or cart_data.get('embeddings') is None:
        return MountResponse(success=False, message="Failed to load cartridge data")

    emb = cart_data['embeddings']
    txt = list(cart_data['passages'])
    is_multimodal = cart_data.get('multimodal', False)
    compressed_lens = cart_data.get('compressed_lens', [])

    engine.mounted_name = cart_name
    engine.mounted_path = full_path
    engine.embeddings = emb
    engine.passages = txt
    engine.compressed_lens = compressed_lens
    engine.compressed_texts = []
    engine.multimodal_mode = is_multimodal
    engine.brain_only_mode = False
    engine.deleted_ids = set()

    brain_loaded = False
    sigs_loaded = False
    message_parts = [f"{len(txt)} patterns", f"from {cart_dir}"]

    if engine.gpu_available and os.path.exists(brain_path):
        valid, msg = validate_brain_manifest(brain_path, emb)
        if valid:
            with engine.lock:
                engine.ml.load_brain(brain_path)
            engine.physics_trained = True
            brain_loaded = True
            size_mb = os.path.getsize(brain_path) / (1024 * 1024)
            message_parts.append(f"Brain: {size_mb:.1f}MB")
            engine.training_progress = len(emb)
            engine.training_total = len(emb)

            if os.path.exists(sig_path):
                sig_data = load_signatures(sig_path)
                if sig_data and sig_data['n_patterns'] == len(emb):
                    engine.signatures = sig_data['signatures']
                    engine.signatures_loaded = True
                    sigs_loaded = True
                    if sig_data.get('compressed_lens') is not None:
                        engine.compressed_lens = list(sig_data['compressed_lens'])
                    if sig_data.get('compressed_texts') is not None:
                        engine.compressed_texts = sig_data['compressed_texts']

    return MountResponse(
        success=True,
        name=cart_name,
        pattern_count=len(txt),
        multimodal=is_multimodal,
        brain_loaded=brain_loaded,
        signatures_loaded=sigs_loaded,
        message=" | ".join(message_parts),
    )


def _mount_brain_by_path(picked_path: str, cart_name: str) -> MountResponse:
    """Mount from an absolute .npy or .npz path. Brain is optional."""
    cart_dir = os.path.dirname(picked_path)
    ext = os.path.splitext(picked_path)[1].lower()

    # Figure out which file is brain and which is signatures
    # Convention A (Studio v83): name_brain.npy + name_signatures.npz
    # Convention B (membot):     name.cart.npz (sigs are the picked file itself)
    brain_path = os.path.join(cart_dir, f"{cart_name}_brain.npy")
    sig_path_convention = os.path.join(cart_dir, f"{cart_name}_signatures.npz")

    if ext == '.npz':
        # The picked file IS the signatures -- use it directly
        sig_path = picked_path
        # Also check the studio convention as fallback
        if not os.path.exists(sig_path):
            sig_path = sig_path_convention
    else:
        # Picked a .npy brain -- look for companion sigs
        sig_path = sig_path_convention

    has_brain = os.path.exists(brain_path)
    has_sigs = os.path.exists(sig_path)

    if not has_sigs:
        return MountResponse(success=False, message=f"Signatures required: {cart_name}_signatures.npz")

    sig_data = load_signatures(sig_path)
    if not sig_data:
        return MountResponse(success=False, message="Failed to load signatures")

    n_patterns = sig_data['n_patterns']
    compressed_lens = list(sig_data.get('compressed_lens') or [])
    compressed_texts = sig_data.get('compressed_texts') or []
    titles = list(sig_data.get('titles') or [])

    # Recover text from compressed_texts
    text_encoder = TextRegionEncoder()
    recovered_texts = []
    for i in range(n_patterns):
        if i < len(compressed_texts) and compressed_texts[i] is not None:
            try:
                txt = text_encoder.decompress_text(bytes(compressed_texts[i]))
                recovered_texts.append(txt if txt else f"[Pattern {i}]")
            except Exception:
                recovered_texts.append(titles[i] if i < len(titles) else f"[Pattern {i}]")
        elif i < len(titles):
            recovered_texts.append(str(titles[i]))
        else:
            recovered_texts.append(f"[Pattern {i}]")

    engine.mounted_name = cart_name
    engine.mounted_path = picked_path
    engine.embeddings = np.zeros((n_patterns, 768), dtype=np.float32)
    engine.passages = recovered_texts
    engine.compressed_lens = compressed_lens
    engine.compressed_texts = compressed_texts
    engine.signatures = sig_data['signatures']
    engine.signatures_loaded = True
    engine.multimodal_mode = True
    engine.brain_only_mode = True
    engine.deleted_ids = set()

    brain_loaded = False
    message_parts = [f"{n_patterns} patterns", f"from {cart_dir}"]

    if has_brain and engine.gpu_available:
        with engine.lock:
            engine.ml.load_brain(brain_path)
        engine.physics_trained = True
        brain_loaded = True
        size_mb = os.path.getsize(brain_path) / (1024 * 1024)
        message_parts.append(f"Brain: {size_mb:.1f}MB")
    else:
        engine.physics_trained = False
        message_parts.append("Signatures only (no brain)")

    engine.training_progress = n_patterns
    engine.training_total = n_patterns

    return MountResponse(
        success=True,
        name=cart_name,
        pattern_count=n_patterns,
        multimodal=True,
        brain_loaded=brain_loaded,
        signatures_loaded=True,
        message=" | ".join(message_parts),
    )


def _mount_brain_only(cart_name: str) -> MountResponse:
    """Mount a brain-only cartridge (no pkl needed)."""
    brain_path = find_companion_file(cart_name, "_brain.npy")
    sig_path = find_companion_file(cart_name, "_signatures.npz")

    if not brain_path:
        return MountResponse(success=False, message="Brain file not found")
    if not sig_path:
        return MountResponse(success=False, message="Signatures required for brain-only mode")

    sig_data = load_signatures(sig_path)
    if not sig_data:
        return MountResponse(success=False, message="Failed to load signatures")

    n_patterns = sig_data['n_patterns']
    compressed_lens = list(sig_data.get('compressed_lens') or [])
    compressed_texts = sig_data.get('compressed_texts') or []
    titles = list(sig_data.get('titles') or [])

    # Recover text from compressed_texts
    text_encoder = TextRegionEncoder()
    recovered_texts = []
    for i in range(n_patterns):
        if i < len(compressed_texts) and compressed_texts[i] is not None:
            try:
                txt = text_encoder.decompress_text(bytes(compressed_texts[i]))
                recovered_texts.append(txt if txt else f"[Pattern {i}]")
            except Exception:
                recovered_texts.append(titles[i] if i < len(titles) else f"[Pattern {i}]")
        elif i < len(titles):
            recovered_texts.append(str(titles[i]))
        else:
            recovered_texts.append(f"[Pattern {i}]")

    engine.mounted_name = cart_name
    engine.embeddings = np.zeros((n_patterns, 768), dtype=np.float32)
    engine.passages = recovered_texts
    engine.compressed_lens = compressed_lens
    engine.compressed_texts = compressed_texts
    engine.signatures = sig_data['signatures']
    engine.signatures_loaded = True
    engine.physics_trained = True
    engine.multimodal_mode = True
    engine.brain_only_mode = True
    engine.deleted_ids = set()

    # Load brain weights
    with engine.lock:
        engine.ml.load_brain(brain_path)

    size_mb = os.path.getsize(brain_path) / (1024 * 1024)
    engine.training_progress = n_patterns
    engine.training_total = n_patterns

    return MountResponse(
        success=True,
        name=cart_name,
        pattern_count=n_patterns,
        multimodal=True,
        brain_loaded=True,
        signatures_loaded=True,
        message=f"Brain-only: {n_patterns} patterns | {size_mb:.1f}MB",
    )


def _mount_pkl(filename: str, cart_name: str) -> MountResponse:
    """Mount a PKL cartridge with optional brain/signatures."""
    path = find_cartridge_path(filename)
    if not path:
        return MountResponse(success=False, message=f"Cartridge not found: {filename}")

    cart_data = load_cartridge(path)
    if not cart_data or cart_data.get('embeddings') is None:
        return MountResponse(success=False, message="Failed to load cartridge data")

    emb = cart_data['embeddings']
    txt = list(cart_data['passages'])
    is_multimodal = cart_data.get('multimodal', False)
    compressed_lens = cart_data.get('compressed_lens', [])

    engine.mounted_name = cart_name
    engine.embeddings = emb
    engine.passages = txt
    engine.compressed_lens = compressed_lens
    engine.compressed_texts = []
    engine.multimodal_mode = is_multimodal
    engine.brain_only_mode = False
    engine.deleted_ids = set()

    brain_loaded = False
    sigs_loaded = False
    message_parts = [f"{len(txt)} patterns"]

    # Try loading brain
    brain_path = find_companion_file(cart_name, "_brain.npy")
    if brain_path and engine.gpu_available:
        valid, msg = validate_brain_manifest(brain_path, emb)
        if valid:
            with engine.lock:
                engine.ml.load_brain(brain_path)
            engine.physics_trained = True
            brain_loaded = True
            size_mb = os.path.getsize(brain_path) / (1024 * 1024)
            message_parts.append(f"Brain: {size_mb:.1f}MB")
            engine.training_progress = len(emb)
            engine.training_total = len(emb)

            # Load signatures
            sig_path = find_companion_file(cart_name, "_signatures.npz")
            if sig_path:
                sig_data = load_signatures(sig_path)
                if sig_data and sig_data['n_patterns'] == len(emb):
                    engine.signatures = sig_data['signatures']
                    engine.signatures_loaded = True
                    sigs_loaded = True
                    if sig_data.get('compressed_lens') is not None:
                        engine.compressed_lens = list(sig_data['compressed_lens'])
                    if sig_data.get('compressed_texts') is not None:
                        engine.compressed_texts = sig_data['compressed_texts']
                    message_parts.append("Sigs loaded")

    # If no brain, start background training
    if engine.gpu_available and not brain_loaded:
        _start_background_training(emb, txt, cart_name)
        message_parts.append("Training started")

    return MountResponse(
        success=True,
        name=cart_name,
        pattern_count=len(txt),
        multimodal=is_multimodal,
        brain_loaded=brain_loaded,
        signatures_loaded=sigs_loaded,
        message=" | ".join(message_parts),
    )


def _start_background_training(embeddings, passages, cart_name):
    """Train physics in background thread.

    Uses TrainingEncoder (region-fill + text + hippocampus) to match
    test_cam_poseidon.py pattern assembly exactly.
    """
    INITIAL_BATCH = 100

    # Train first batch synchronously
    ml = engine.ml
    train_enc = engine.training_encoder
    # CombinedEncoder still needed for sig capture (text recovery)
    sig_enc = engine.combined_encoder
    compressed_lens = []

    n_train = min(len(embeddings), INITIAL_BATCH)
    with engine.lock:
        for i in range(n_train):
            ml.reset()
            pattern, meta = train_enc.encode(embeddings[i], passages[i], i)
            ml.imprint_pattern(pattern)
            ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)
            compressed_lens.append(meta['compressed_len'])

    engine.physics_trained = True
    engine.compressed_lens = compressed_lens

    if len(embeddings) > INITIAL_BATCH:
        engine.training_active = True
        engine.training_total = len(embeddings)
        engine.training_progress = INITIAL_BATCH

        thread = threading.Thread(
            target=_background_train,
            args=(embeddings, passages, train_enc, sig_enc, ml,
                  INITIAL_BATCH, cart_name, list(compressed_lens)),
            daemon=True,
        )
        thread.start()
    else:
        engine.training_progress = len(embeddings)
        engine.training_total = len(embeddings)


def _background_train(embeddings, passages, train_enc, sig_enc, ml,
                      start_idx, cart_name, initial_compressed_lens):
    """Background training thread.

    Uses TrainingEncoder for Hebbian training (region-fill + text + hippocampus).
    Uses CombinedEncoder (sig_enc) for signature capture (text recovery).
    """
    compressed_lens = list(initial_compressed_lens)

    try:
        for i in range(start_idx, len(embeddings)):
            with engine.lock:
                ml.reset()
                pattern, meta = train_enc.encode(embeddings[i], passages[i], i)
                ml.imprint_pattern(pattern)
                ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            if i >= len(compressed_lens):
                compressed_lens.append(meta['compressed_len'])
            engine.training_progress = i + 1
            time.sleep(0.001)

        engine.compressed_lens = compressed_lens

        # ── DIAGNOSTIC: Test Associate with in-memory weights (before save) ──
        # This isolates save/load vs training as the source of ~49% sign preservation
        try:
            test_query = "What causes earthquakes?"
            from sentence_transformers import SentenceTransformer
            _diag_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                                              trust_remote_code=True)
            _diag_emb = _diag_model.encode(f"search_query: {test_query}",
                                           normalize_embeddings=False)
            _q_bin = (_diag_emb > 0).astype(np.uint8)
            _stored_signs = (embeddings > 0).astype(np.uint8)
            _n_dims = embeddings.shape[1]

            # Encode query as region-fill (4D→reshape layout, matches encoder + standalone)
            _grid_enc = np.zeros((64, 64, 64, 64), dtype=np.float32)
            for _i in range(_n_dims):
                if _diag_emb[_i] > 0:
                    _grid_enc[_i // 64, _i % 64, :, :] = 1.0
            _q_pattern = _grid_enc.reshape(4096, 4096)

            with engine.lock:
                ml.reset()
                ml.imprint_pattern(_q_pattern)
                ml.settle(frames=30, learn=False)
                _settled = ml.recall()

            _grid = _settled.reshape(64, 64, 64, 64)
            _settled_signs = np.zeros(_n_dims, dtype=np.uint8)
            for _i in range(_n_dims):
                _settled_signs[_i] = 1 if np.mean(_grid[_i // 64, _i % 64]) > 0.5 else 0

            _sign_pres = float(np.mean(_q_bin == _settled_signs))
            _xor = np.bitwise_xor(_settled_signs, _stored_signs)
            _ham = 1.0 - _xor.sum(axis=1).astype(np.float32) / _n_dims
            _top5 = np.argsort(_ham)[-5:][::-1]

            print(f"\n{'='*70}")
            print(f"[DIAG] IN-MEMORY Associate test (before save/load)")
            print(f"[DIAG] Query: \"{test_query}\"")
            print(f"[DIAG] Sign preservation: {_sign_pres:.1%}")
            print(f"[DIAG] Settled stats: nonzero={np.count_nonzero(_settled > 0.5)}")
            for _rank, _idx in enumerate(_top5):
                _title = passages[_idx].splitlines()[0][:60] if _idx < len(passages) else "?"
                print(f"[DIAG]   {_rank+1}. {_ham[_idx]:.4f}  {_title}")
            print(f"{'='*70}\n")
            del _diag_model
        except Exception as _diag_err:
            print(f"[DIAG] In-memory test failed: {_diag_err}")

        # Save brain
        brain_path = os.path.join(DATA_DIR, f"{cart_name}_brain")
        os.makedirs(DATA_DIR, exist_ok=True)
        with engine.lock:
            ml.save_brain(brain_path)
        actual_path = brain_path + ".npy"
        save_brain_manifest(actual_path, embeddings)

        # Capture L3 signatures (256x256 = 65536 floats)
        sig_path = os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")
        signatures = np.zeros((len(embeddings), 65536), dtype=np.float32)
        compressed_texts = []

        for i in range(len(embeddings)):
            with engine.lock:
                ml.reset()
                pattern, _ = sig_enc.encode(embeddings[i], passages[i])
                ml.imprint_pattern(pattern)
                ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
                signatures[i] = ml.recall_l3().flatten()

            compressed_bytes = sig_enc.text_encoder.compress_text(passages[i])
            compressed_texts.append(compressed_bytes)

        titles = [p.splitlines()[0][:50] if p else "" for p in passages]
        save_signatures(sig_path, signatures, titles, compressed_lens, compressed_texts)

        engine.signatures = signatures
        engine.signatures_loaded = True
        engine.compressed_texts = compressed_texts

        # Restore clean post-training brain state.
        # The 10K signature capture settles accumulate fatigue/BCM state
        # that flattens the energy landscape into a single attractor.
        # Reloading the brain (saved before sig capture) resets this.
        with engine.lock:
            ml.load_brain(actual_path)
        print(f"[BG Training] Brain restored after sig capture")

        # ── DIAGNOSTIC 2: Test Associate AFTER save/load round-trip ──
        try:
            test_query = "What causes earthquakes?"
            from sentence_transformers import SentenceTransformer
            _diag_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                                              trust_remote_code=True)
            _diag_emb = _diag_model.encode(f"search_query: {test_query}",
                                           normalize_embeddings=False)
            _q_bin = (_diag_emb > 0).astype(np.uint8)
            _stored_signs = (embeddings > 0).astype(np.uint8)
            _n_dims = embeddings.shape[1]

            # Region-fill query pattern (4D→reshape layout, matches encoder + standalone)
            _grid_enc = np.zeros((64, 64, 64, 64), dtype=np.float32)
            for _i in range(_n_dims):
                if _diag_emb[_i] > 0:
                    _grid_enc[_i // 64, _i % 64, :, :] = 1.0
            _q_pattern = _grid_enc.reshape(4096, 4096)

            # Test on SHARED engine (loaded brain)
            with engine.lock:
                ml.reset()
                ml.imprint_pattern(_q_pattern)
                ml.settle(frames=30, learn=False)
                _settled = ml.recall()

            _grid = _settled.reshape(64, 64, 64, 64)
            _settled_signs = np.zeros(_n_dims, dtype=np.uint8)
            for _i in range(_n_dims):
                _settled_signs[_i] = 1 if np.mean(_grid[_i // 64, _i % 64]) > 0.5 else 0

            _sign_pres = float(np.mean(_q_bin == _settled_signs))
            _xor = np.bitwise_xor(_settled_signs, _stored_signs)
            _ham = 1.0 - _xor.sum(axis=1).astype(np.float32) / _n_dims
            _top5 = np.argsort(_ham)[-5:][::-1]

            print(f"\n{'='*70}")
            print(f"[DIAG2] POST-SAVE Associate test (shared engine, loaded brain)")
            print(f"[DIAG2] Query: \"{test_query}\"")
            print(f"[DIAG2] Sign preservation: {_sign_pres:.1%}")
            print(f"[DIAG2] Settled stats: nonzero={np.count_nonzero(_settled > 0.5)}")
            for _rank, _idx in enumerate(_top5):
                _title = passages[_idx].splitlines()[0][:60] if _idx < len(passages) else "?"
                print(f"[DIAG2]   {_rank+1}. {_ham[_idx]:.4f}  {_title}")
            print(f"{'='*70}\n")
            del _diag_model
        except Exception as _diag_err:
            print(f"[DIAG2] Post-save test failed: {_diag_err}")

    except Exception as e:
        print(f"[BG Training] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.training_active = False


@app.post("/api/cartridges/unmount", response_model=MessageResponse)
async def unmount_cartridge():
    name = engine.mounted_name or "none"
    engine.unmount()
    return MessageResponse(success=True, message=f"Unmounted {name}")


@app.post("/api/cartridges/lock", response_model=MessageResponse)
async def lock_cartridge():
    engine.read_only = True
    return MessageResponse(success=True, message="Cartridge locked (read-only)")


@app.post("/api/cartridges/unlock", response_model=MessageResponse)
async def unlock_cartridge():
    _enforce_writable()  # Public demo: refuses unlock so the per-cart lock can't be cleared
    if not engine.mounted_name:
        return MessageResponse(success=False, message="No cartridge mounted")
    engine.read_only = False
    return MessageResponse(success=True, message="Cartridge unlocked (read-write)")


@app.post("/api/cartridges/save", response_model=MessageResponse)
async def save_cartridge():
    _enforce_writable()
    if not engine.mounted_name:
        return MessageResponse(success=False, message="No cartridge mounted")
    if engine.read_only:
        return MessageResponse(success=False, message="Cartridge is read-only. Unlock first.")
    if not engine.dirty:
        return MessageResponse(success=True, message="Nothing to save")

    result = await asyncio.to_thread(_save_cartridge_sync)
    return result


def _save_cartridge_sync() -> MessageResponse:
    """Persist current in-memory cartridge state to disk."""
    import os as _os

    cart_name = engine.mounted_name
    _os.makedirs(DATA_DIR, exist_ok=True)
    save_path = _os.path.join(DATA_DIR, f"{cart_name}.pkl")

    try:
        from .cartridge_io import save_cartridge_multimodal

        save_cartridge_multimodal(
            save_path,
            engine.embeddings,
            engine.passages,
            engine.compressed_lens,
        )

        saved_parts = [f"{len(engine.passages)} patterns saved to {save_path}"]

        # Save signatures if available
        if engine.signatures is not None and engine.signatures_loaded:
            sig_path = _os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")
            titles = [p.splitlines()[0][:50] if p else "" for p in engine.passages]
            save_signatures(
                sig_path, engine.signatures, titles,
                engine.compressed_lens,
                engine.compressed_texts if isinstance(engine.compressed_texts, list) else None,
            )
            saved_parts.append("Signatures saved")

        # Save brain if trained
        if engine.physics_trained and engine.ml:
            brain_path = _os.path.join(DATA_DIR, f"{cart_name}_brain")
            with engine.lock:
                engine.ml.save_brain(brain_path)
            actual_path = brain_path + ".npy"
            save_brain_manifest(actual_path, engine.embeddings)
            saved_parts.append("Brain saved")

        # Update mounted_path to point to where we saved
        engine.mounted_path = save_path
        engine.dirty = False

        return MessageResponse(success=True, message=" | ".join(saved_parts))

    except Exception as e:
        return MessageResponse(success=False, message=f"Save failed: {e}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/api/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    if not engine.mounted_name:
        return SearchResponse(
            query=req.query, mode="none", elapsed_ms=0,
            result_count=0, results=[]
        )

    t0 = time.perf_counter()
    results, mode_label = await asyncio.to_thread(
        do_search, req.query, req.mode, req.alpha, req.top_k
    )
    elapsed = (time.perf_counter() - t0) * 1000

    # Split-cart provenance hint — for split carts, the search response surfaces
    # source_db so the frontend knows to render the "Load full passage from <db>"
    # CTA in the modal. paper_id is NOT fetched here (would require a SQLite hop
    # per result, defeating the lazy-load design); it arrives later via
    # /api/patterns/{idx} when the user clicks the CTA. Same shape as membot.
    source_db_label = (
        os.path.basename(engine.sqlite_db_path)
        if engine.is_split_cart and engine.sqlite_db_path else None
    )

    search_results = []
    hippo = engine.hippocampus
    for rank, r in enumerate(results):
        idx = r['idx']
        perms = None
        if hippo is not None and 0 <= idx < len(hippo):
            perms = hippo[idx].get('perms')
        search_results.append(SearchResult(
            rank=rank + 1,
            idx=idx,
            score=r['score'],
            cosine_score=r.get('cosine_score'),
            physics_score=r.get('physics_score'),
            hamming_score=r.get('hamming_score'),
            keyword_boost=r.get('keyword_boost'),
            title=r['title'],
            preview=r['preview'],
            full_text=r['full_text'],
            from_lattice=r.get('from_lattice', False),
            source_db=source_db_label,
            prev_idx=r.get('prev_idx'),
            next_idx=r.get('next_idx'),
            perms=perms,
        ))

    return SearchResponse(
        query=req.query,
        mode=mode_label,
        elapsed_ms=round(elapsed, 1),
        result_count=len(search_results),
        results=search_results,
    )


# ---------------------------------------------------------------------------
# Patterns (Fetch / Delete / Restore)
# ---------------------------------------------------------------------------

@app.delete("/api/patterns/{idx}", response_model=MessageResponse)
async def delete_pattern(idx: int):
    _enforce_writable(idx=idx)
    if engine.read_only:
        return MessageResponse(success=False, message="Cartridge is read-only. Unlock first.")
    if idx < 0 or idx >= len(engine.passages):
        return MessageResponse(success=False, message=f"Invalid index: {idx}")
    engine.deleted_ids.add(idx)
    return MessageResponse(success=True, message=f"Pattern {idx} tombstoned")


@app.post("/api/patterns/{idx}/restore", response_model=MessageResponse)
async def restore_pattern(idx: int):
    _enforce_writable(idx=idx)
    if engine.read_only:
        return MessageResponse(success=False, message="Cartridge is read-only. Unlock first.")
    engine.deleted_ids.discard(idx)
    return MessageResponse(success=True, message=f"Pattern {idx} restored")


@app.get("/api/patterns/deleted", response_model=DeletedListResponse)
async def list_deleted():
    deleted = []
    for idx in sorted(engine.deleted_ids):
        if idx < len(engine.passages):
            text = engine.passages[idx]
            lines = text.splitlines() if text else ["[empty]"]
            deleted.append(DeletedPattern(
                idx=idx,
                title=lines[0][:100],
                preview=" ".join(lines[1:3])[:200] if len(lines) > 1 else "",
            ))
    return DeletedListResponse(deleted=deleted)


# Lone UTF-16 surrogates (U+D800–U+DFFF) crash FastAPI's JSON serializer.
# They appear in legacy carts as decoder artifacts from earlier ingest paths.
# Drop them — astral characters (codepoints >= U+10000) are outside this range
# and are preserved. Local copy (not imported from cart-builder) per the
# droplet-reconciliation discipline: VPS code stays self-contained.
_LONE_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _scrub_lone_surrogates(text: str) -> str:
    return _LONE_SURROGATE_RE.sub('', text) if text else text


@app.get("/api/patterns", response_model=PatternListResponse)
async def list_patterns(offset: int = 0, limit: int = 25, q: str | None = None):
    """Paginated list of active (non-tombstoned) passages with first-line
    title + body preview. Used by the Edit Carts passage browser to give
    users a click-to-populate IDX experience for the Update/Delete panels.

    Optional `q` is a case-insensitive substring filter on the passage text.
    Returns total count of active+matching passages for client-side
    pagination math (page count = ceil(total / limit)).

    Andy 2026-05-10. Tombstoned passages live in /api/patterns/deleted —
    intentional separation so the browser doesn't have to filter them out.
    """
    if limit < 1:
        limit = 25
    if limit > 200:
        limit = 200  # defensive cap so a misconfigured client can't pull 100k rows
    if offset < 0:
        offset = 0

    needle = (q or "").strip().lower()
    deleted = engine.deleted_ids

    matching: list[tuple[int, str]] = []
    for i, text in enumerate(engine.passages):
        if i in deleted:
            continue
        if needle and needle not in (text or "").lower():
            continue
        matching.append((i, text or ""))

    total = len(matching)
    page = matching[offset : offset + limit]

    items: list[PatternListItem] = []
    for idx, text in page:
        text = _scrub_lone_surrogates(text)
        lines = text.splitlines() if text else ["[empty]"]
        title = lines[0][:120] if lines else "[empty]"
        preview = " ".join(lines[1:3])[:200] if len(lines) > 1 else ""
        word_count = len((text or "").split())
        items.append(PatternListItem(
            idx=idx,
            title=title,
            preview=preview,
            word_count=word_count,
        ))

    return PatternListResponse(
        passages=items,
        total=total,
        offset=offset,
        limit=limit,
        filter=q if q else None,
    )


@app.get("/api/patterns/{idx}", response_model=PatternResponse)
async def get_pattern(idx: int):
    """Fetch a single pattern by index, with hippocampus PREV/NEXT links.

    For split carts, the in-RAM `engine.passages[idx]` is the 200-char snippet;
    the FULL passage is fetched from the SQLite sidecar on demand. This is the
    user-driven 'load source' path (parity with membot's RAG+ provenance
    feature) — every full-text fetch is a labeled action against the named
    source database.
    """
    if idx < 0 or idx >= len(engine.passages):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Pattern {idx} not found")

    # Default: in-RAM text (full for standard carts, snippet for split carts)
    text = engine.passages[idx]
    paper_id = None
    source_db = None

    # Split cart: fetch the FULL passage from SQLite sidecar.
    # int(idx) cast is load-bearing: sqlite3 binds numpy.int64 silently as a
    # no-match value, so passing the raw idx (which can be int or numpy.int64
    # depending on caller) returns zero rows.
    if engine.is_split_cart and engine.sqlite_conn is not None:
        sqlite_row = _sqlite_fetch_passages(engine.sqlite_conn, [int(idx)]).get(int(idx))
        if sqlite_row:
            text = sqlite_row.get("passage") or text
            paper_id = sqlite_row.get("paper_id")
        if engine.sqlite_db_path:
            source_db = os.path.basename(engine.sqlite_db_path)

    text = _scrub_lone_surrogates(text)
    lines = text.splitlines() if text else ["[empty]"]
    title = lines[0][:100]
    preview = " ".join(lines[1:3])[:200] if len(lines) > 1 else ""

    prev_idx = None
    next_idx = None
    perms = None
    if engine.hippocampus is not None and idx < len(engine.hippocampus):
        entry = engine.hippocampus[idx]
        prev_idx = entry.get('prev')
        next_idx = entry.get('next')
        perms = entry.get('perms')

    return PatternResponse(
        idx=idx, title=title, preview=preview, full_text=text,
        prev_idx=prev_idx, next_idx=next_idx,
        source_db=source_db, paper_id=paper_id,
        perms=perms,
    )


# ---------------------------------------------------------------------------
# Add Passage
# ---------------------------------------------------------------------------

@app.post("/api/patterns", response_model=MessageResponse)
async def add_passage(req: AddPassageRequest):
    _enforce_writable()
    if not engine.mounted_name:
        return MessageResponse(success=False, message="No cartridge mounted")
    if engine.read_only:
        return MessageResponse(success=False, message="Cartridge is read-only. Unlock first.")

    text = req.text.strip()
    if not text:
        return MessageResponse(success=False, message="Text is empty")

    result = await asyncio.to_thread(_add_passage_sync, text)
    return result


def _add_passage_sync(text: str) -> MessageResponse:
    """Embed text, append to cartridge arrays, optionally imprint into brain."""
    # Embed with Nomic
    embedding = engine.embed_documents([text])[0]

    # Append to passages and embeddings
    engine.passages.append(text)
    if engine.embeddings is not None:
        engine.embeddings = np.vstack([engine.embeddings, embedding.reshape(1, -1)])
    else:
        engine.embeddings = embedding.reshape(1, -1)

    new_idx = len(engine.passages) - 1

    # If GPU + training encoder: imprint, settle+learn, capture signature
    if engine.gpu_available and engine.ml and engine.training_encoder:
        # Train with TrainingEncoder (region-fill + text + hippocampus)
        with engine.lock:
            engine.ml.reset()
            train_pattern, meta = engine.training_encoder.encode(
                embedding, text, new_idx)
            engine.ml.imprint_pattern(train_pattern)
            engine.ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            # Capture L3 signature using CombinedEncoder (for text recovery)
            engine.ml.reset()
            sig_pattern, _ = engine.combined_encoder.encode(embedding, text)
            engine.ml.imprint_pattern(sig_pattern)
            engine.ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
            new_sig = engine.ml.recall_l3().flatten()

        engine.compressed_lens.append(meta['compressed_len'])

        compressed_bytes = engine.combined_encoder.text_encoder.compress_text(text)
        if isinstance(engine.compressed_texts, list):
            engine.compressed_texts.append(compressed_bytes)

        if engine.signatures is not None:
            engine.signatures = np.vstack([engine.signatures, new_sig.reshape(1, -1)])

    engine.training_progress = len(engine.passages)
    engine.training_total = len(engine.passages)
    engine.dirty = True

    title = text.splitlines()[0][:50] if text else "[empty]"
    return MessageResponse(success=True, message=f"Added pattern #{new_idx}: {title}")


# ---------------------------------------------------------------------------
# Forge
# ---------------------------------------------------------------------------

@app.post("/api/forge", response_model=MessageResponse)
async def forge_endpoint(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    _enforce_writable()  # Forge creates new cart files on disk — refuse in read-only mode
    if engine.read_only and engine.mounted_name:
        return MessageResponse(success=False, message="Cartridge is read-only. Unlock first.")
    file_data = []
    for f in files:
        content = await f.read()
        file_data.append((f.filename, content))

    result = await asyncio.to_thread(forge_cartridge, name, file_data)
    return MessageResponse(
        success=result["success"],
        message=result["message"],
    )


# ---------------------------------------------------------------------------
# Membox visualizer (proxy to membot.membox)
# ---------------------------------------------------------------------------

def _lock_dict_to_model(d: dict, cart_id: str) -> MemboxLockState:
    """Convert membox CartLock.stats() dict to Pydantic model."""
    return MemboxLockState(
        cart_id=d.get("cart_id", cart_id),
        holder=d.get("holder"),
        held_for_seconds=d.get("held_for_seconds"),
        lease_seconds=d.get("lease_seconds", 30),
        acquire_count=d.get("acquire_count", 0),
        wait_count=d.get("wait_count", 0),
        is_locked=d.get("is_locked", False),
    )


@app.get("/api/membox/carts", response_model=MemboxCartListResponse)
async def membox_list_carts():
    if not _MEMBOX_AVAILABLE:
        return MemboxCartListResponse(carts=[])
    try:
        mounts = _membox.list_mounts()
    except Exception as err:
        print(f"[VPS] membox.list_mounts failed: {err}")
        return MemboxCartListResponse(carts=[])

    carts = []
    for m in mounts:
        cart_id = m.get("cart_id", "")
        carts.append(MemboxCartInfo(
            cart_id=cart_id,
            role=m.get("role"),
            n_patterns=m.get("n_patterns", 0),
            lock=_lock_dict_to_model(m.get("lock", {}), cart_id),
            recent_writes=m.get("recent_writes", 0),
        ))
    return MemboxCartListResponse(carts=carts)


@app.get("/api/membox/status/{cart_id}", response_model=MemboxStatus)
async def membox_get_status(cart_id: str):
    if not _MEMBOX_AVAILABLE:
        return MemboxStatus(
            cart_id=cart_id,
            n_patterns=0,
            lock=MemboxLockState(cart_id=cart_id),
            membox_enabled=False,
        )
    try:
        s = _membox.status(cart_id)
    except Exception as err:
        print(f"[VPS] membox.status({cart_id}) failed: {err}")
        return MemboxStatus(
            cart_id=cart_id,
            n_patterns=0,
            lock=MemboxLockState(cart_id=cart_id),
            membox_enabled=False,
        )

    recent = [
        MemboxWriteEntry(
            agent_id=w.get("agent_id", "?"),
            written_at=w.get("written_at", ""),
            local_addr=w.get("local_addr", -1),
            origin=w.get("origin", "agent"),
            text_preview=w.get("text_preview", ""),
        )
        for w in s.get("recent_writes", [])
    ]
    return MemboxStatus(
        cart_id=s.get("cart_id", cart_id),
        n_patterns=s.get("n_patterns", 0),
        lock=_lock_dict_to_model(s.get("lock", {}), cart_id),
        writes_by_agent=s.get("writes_by_agent", {}),
        recent_writes=recent,
        membox_enabled=s.get("membox_enabled", True),
    )


@app.post("/api/membox/mount", response_model=MessageResponse)
async def membox_mount_endpoint(req: MemboxMountRequest):
    """TEMPORARY: lets the visualizer panel mount carts directly via Membox.
    Will be removed once Phase 2 unifies the single-user mount path with Membox."""
    _enforce_writable()
    if not _MEMBOX_AVAILABLE:
        return MessageResponse(success=False, message="Membox not available")
    try:
        result = await asyncio.to_thread(
            _membox.mount,
            req.cart_path,
            req.cart_id,
            req.role,
            req.lease_seconds,
            req.verify_integrity,
        )
    except Exception as err:
        return MessageResponse(success=False, message=f"mount failed: {err}")

    cid = result.get("cart_id", "?")
    n = result.get("n_patterns", 0)
    return MessageResponse(success=True, message=f"Mounted {cid} ({n} patterns)")


@app.post("/api/membox/unmount", response_model=MessageResponse)
async def membox_unmount_endpoint(req: MemboxUnmountRequest):
    _enforce_writable()
    if not _MEMBOX_AVAILABLE:
        return MessageResponse(success=False, message="Membox not available")
    try:
        await asyncio.to_thread(_membox.unmount, req.cart_id)
    except Exception as err:
        return MessageResponse(success=False, message=f"unmount failed: {err}")
    return MessageResponse(success=True, message=f"Unmounted {req.cart_id}")


@app.post("/api/membox/imprint", response_model=MessageResponse)
async def membox_imprint(req: MemboxImprintRequest):
    _enforce_writable()
    if not _MEMBOX_AVAILABLE:
        return MessageResponse(success=False, message="Membox not available")
    try:
        result = await asyncio.to_thread(
            _membox.imprint,
            req.cart_id,
            req.text,
            req.agent_id,
            req.tags,
            req.reasoning,
            req.origin,
            req.timeout_ms,
        )
    except Exception as err:
        return MessageResponse(success=False, message=f"imprint failed: {err}")

    if result.get("ok"):
        addr = result.get("local_addr", "?")
        return MessageResponse(success=True, message=f"Wrote pattern #{addr} as {req.agent_id}")
    else:
        return MessageResponse(success=False, message=result.get("error", "unknown error"))


# ---------------------------------------------------------------------------
# WebSocket for progress
# ---------------------------------------------------------------------------

@app.websocket("/ws/progress")
async def websocket_progress(ws: WebSocket):
    await ws.accept()
    engine.ws_connections.append(ws)
    try:
        while True:
            # Push status every second
            await ws.send_json({
                "type": "status",
                "training_active": engine.training_active,
                "training_progress": engine.training_progress,
                "training_total": engine.training_total,
                "mounted": engine.mounted_name,
                "signatures_loaded": engine.signatures_loaded,
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in engine.ws_connections:
            engine.ws_connections.remove(ws)
