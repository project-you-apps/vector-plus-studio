"""
Vector+ Studio 1.0 -- FastAPI Backend

Start with:
    cd vector-plus-studio-repo
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import os
import time
import threading
import numpy as np

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from .engine import engine, TRAIN_SETTLE_FRAMES, SIG_SETTLE_FRAMES, TextRegionEncoder
from .models import (
    MountRequest, SearchRequest, AddPassageRequest,
    CartridgeInfo, CartridgeListResponse, MountResponse,
    SearchResult, SearchResponse, StatusResponse,
    DeletedPattern, DeletedListResponse, MessageResponse,
)
from .cartridge_io import (
    list_cartridges as _list_cartridges, load_cartridge, load_signatures,
    find_cartridge_path, find_companion_file, validate_brain_manifest,
    save_brain_manifest, save_signatures, DATA_DIR,
)
from .search import search as do_search
from .forge import forge_cartridge


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.boot()
    # Pre-warm embedder in background so first search is fast
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, engine.load_embedder)
    yield
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


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        engine_ready=engine.engine_ready,
        gpu_available=engine.gpu_available,
        mounted_cartridge=engine.mounted_name,
        pattern_count=len(engine.passages),
        physics_trained=engine.physics_trained,
        training_active=engine.training_active,
        training_progress=engine.training_progress,
        training_total=engine.training_total,
        multimodal=engine.multimodal_mode,
        signatures_loaded=engine.signatures_loaded,
        deleted_count=len(engine.deleted_ids),
        dirty=engine.dirty,
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
                return await asyncio.to_thread(_mount_membot_npz, filename, cart_name)
            else:
                # Studio signatures format
                cart_name = basename
                for suffix in ('_signatures.npz', '.npz'):
                    if cart_name.endswith(suffix):
                        cart_name = cart_name[:-len(suffix)]
                        break
                return await asyncio.to_thread(_mount_brain_by_path, filename, cart_name)

        if ext == '.npy':
            cart_name = basename
            for suffix in ('_brain.npy', '.npy'):
                if cart_name.endswith(suffix):
                    cart_name = cart_name[:-len(suffix)]
                    break
            return await asyncio.to_thread(_mount_brain_by_path, filename, cart_name)

        cart_name = os.path.splitext(basename)[0]
        return await asyncio.to_thread(_mount_pkl_by_path, filename, cart_name)

    is_brain_only = filename.endswith("(brain only)")
    cart_name = filename.replace(" (brain only)", "").replace(".pkl", "")

    if is_brain_only:
        return await asyncio.to_thread(_mount_brain_only, cart_name)
    else:
        return await asyncio.to_thread(_mount_pkl, filename, cart_name)


def _mount_membot_npz(full_path: str, cart_name: str) -> MountResponse:
    """Mount a membot-format .cart.npz (embeddings + passages, like a PKL)."""
    try:
        data = np.load(full_path, allow_pickle=True)
        emb = data['embeddings']
        passages_raw = data['passages']
        txt = [str(p) for p in passages_raw]
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

    message_parts = [f"{len(txt)} patterns", f"from {cart_dir}"]

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
    """Train physics in background thread."""
    INITIAL_BATCH = 100

    # Train first batch synchronously
    ml = engine.ml
    enc = engine.combined_encoder
    compressed_lens = []

    n_train = min(len(embeddings), INITIAL_BATCH)
    with engine.lock:
        ml.reset()
        for i in range(n_train):
            pattern, meta = enc.encode(embeddings[i], passages[i])
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
            args=(embeddings, passages, enc, ml, INITIAL_BATCH,
                  cart_name, list(compressed_lens)),
            daemon=True,
        )
        thread.start()
    else:
        engine.training_progress = len(embeddings)
        engine.training_total = len(embeddings)


def _background_train(embeddings, passages, enc, ml, start_idx,
                      cart_name, initial_compressed_lens):
    """Background training thread."""
    compressed_lens = list(initial_compressed_lens)

    try:
        for i in range(start_idx, len(embeddings)):
            with engine.lock:
                pattern, meta = enc.encode(embeddings[i], passages[i])
                ml.imprint_pattern(pattern)
                ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            if i >= len(compressed_lens):
                compressed_lens.append(meta['compressed_len'])
            engine.training_progress = i + 1
            time.sleep(0.001)

        engine.compressed_lens = compressed_lens

        # Save brain
        brain_path = os.path.join(DATA_DIR, f"{cart_name}_brain")
        os.makedirs(DATA_DIR, exist_ok=True)
        with engine.lock:
            ml.save_brain_compact(brain_path)
        actual_path = brain_path + ".npy"
        save_brain_manifest(actual_path, embeddings)

        # Capture signatures
        sig_path = os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")
        signatures = np.zeros((len(embeddings), 4096), dtype=np.float32)
        compressed_texts = []

        for i in range(len(embeddings)):
            with engine.lock:
                ml.reset()
                pattern, _ = enc.encode(embeddings[i], passages[i])
                ml.imprint_pattern(pattern)
                ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
                signatures[i] = ml.recall_l2().flatten()

            compressed_bytes = enc.text_encoder.compress_text(passages[i])
            compressed_texts.append(compressed_bytes)

        titles = [p.splitlines()[0][:50] if p else "" for p in passages]
        save_signatures(sig_path, signatures, titles, compressed_lens, compressed_texts)

        engine.signatures = signatures
        engine.signatures_loaded = True
        engine.compressed_texts = compressed_texts

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


@app.post("/api/cartridges/save", response_model=MessageResponse)
async def save_cartridge():
    if not engine.mounted_name:
        return MessageResponse(success=False, message="No cartridge mounted")
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
                engine.ml.save_brain_compact(brain_path)
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

    search_results = []
    for rank, r in enumerate(results):
        search_results.append(SearchResult(
            rank=rank + 1,
            idx=r['idx'],
            score=r['score'],
            cosine_score=r.get('cosine_score'),
            physics_score=r.get('physics_score'),
            title=r['title'],
            preview=r['preview'],
            full_text=r['full_text'],
            from_lattice=r.get('from_lattice', False),
        ))

    return SearchResponse(
        query=req.query,
        mode=mode_label,
        elapsed_ms=round(elapsed, 1),
        result_count=len(search_results),
        results=search_results,
    )


# ---------------------------------------------------------------------------
# Patterns (Delete / Restore)
# ---------------------------------------------------------------------------

@app.delete("/api/patterns/{idx}", response_model=MessageResponse)
async def delete_pattern(idx: int):
    if idx < 0 or idx >= len(engine.passages):
        return MessageResponse(success=False, message=f"Invalid index: {idx}")
    engine.deleted_ids.add(idx)
    return MessageResponse(success=True, message=f"Pattern {idx} tombstoned")


@app.post("/api/patterns/{idx}/restore", response_model=MessageResponse)
async def restore_pattern(idx: int):
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


# ---------------------------------------------------------------------------
# Add Passage
# ---------------------------------------------------------------------------

@app.post("/api/patterns", response_model=MessageResponse)
async def add_passage(req: AddPassageRequest):
    if not engine.mounted_name:
        return MessageResponse(success=False, message="No cartridge mounted")

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

    # If GPU + combined encoder: encode multimodal, imprint, settle+learn, capture L2
    if engine.gpu_available and engine.ml and engine.combined_encoder:
        with engine.lock:
            pattern, meta = engine.combined_encoder.encode(embedding, text)
            engine.ml.imprint_pattern(pattern)
            engine.ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            # Capture L2 signature
            engine.ml.reset()
            engine.ml.imprint_pattern(pattern)
            engine.ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
            new_sig = engine.ml.recall_l2().flatten()

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
