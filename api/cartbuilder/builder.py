"""
builder.py — Cart building pipeline (embed + sign-zero + package).

Ported into VPS subpackage 2026-06-23 (Andy's reconciliation directive).
Was: standalone cart-builder/cart-builder/builder.py with dead-code MEMBOT_DIR
sys.path injection that resolved to nonexistent cart-builder/membot/ and silently
fell through to same-directory import of cartridge_builder.py. Now: clean relative
import within api/cartbuilder/ subpackage.

NOTE: the local cartridge_builder.py (Feb 23 2026) has diverged from membot/
cartridge_builder.py (May 13 2026 — adds native JSON/JSONL handling). This port
preserves the cart-builder local copy to keep behavior identical to pre-port
state; reconciling the divergence is a separate decision.
"""
import os
import time
import json
import hashlib
import threading
import numpy as np

from .cartridge_builder import (
    get_embedder, embed_texts, build_metadata, save_cartridge, chunk_text,
)

# Build state shared with Flask
build_state = {
    "status": "idle",  # idle, building, done, error
    "progress": 0.0,
    "chunks_done": 0,
    "chunks_total": 0,
    "elapsed_ms": 0,
    "cart_path": None,
    "cart_size_mb": 0,
    "error": None,
}
_lock = threading.Lock()


def get_state() -> dict:
    with _lock:
        return dict(build_state)


def _update(kv: dict):
    with _lock:
        build_state.update(kv)


def build_cart_async(chunks: list[dict], cart_name: str, output_dir: str,
                     file_metadata: dict = None,
                     description: str = "",
                     agent_briefing: str = "",
                     owner: str = "",
                     tags: list = None,
                     creator: str = "Cart Builder (cloud)"):
    """Run cart build in a background thread.

    Args:
        chunks: list of {"text": str, "source": str, "page": int|None}
        cart_name: name for the cart file
        output_dir: where to save the cart
        file_metadata: {source_name: {"owner": str, "description": str, "tags": list}}
        description: cart-level description string (baked into pattern0_data)
        agent_briefing: cart-level agent-briefing string (baked into pattern0_data)
        owner: cart-level owner string (baked into pattern0_data)
        tags: cart-level list of tag strings (baked into pattern0_data)
        creator: cart-level creator string; falls back to "Cart Builder (cloud)"
                 for server-side builds. Frontend paths pass "Cart Builder
                 (browser)" or "Cart Builder (local)" (via the __init__ /build
                 handler's creator resolution); the "(cloud)" suffix labels
                 anything that lands here without a JWT-user or client-side
                 creator override.
    """
    t = threading.Thread(
        target=_build_cart,
        args=(chunks, cart_name, output_dir, file_metadata or {}),
        kwargs={
            "description": description,
            "agent_briefing": agent_briefing,
            "owner": owner,
            "tags": tags or [],
            "creator": creator,
        },
        daemon=True,
    )
    t.start()


def _build_cart(chunks: list[dict], cart_name: str, output_dir: str,
                file_metadata: dict,
                description: str = "",
                agent_briefing: str = "",
                owner: str = "",
                tags: list = None,
                creator: str = "Cart Builder GUI"):
    try:
        n = len(chunks)
        _update({"status": "building", "chunks_total": n, "chunks_done": 0, "progress": 0.0, "error": None})
        t0 = time.time()

        # Prepare texts and doc_map for cartridge_builder
        texts = []
        doc_map = []
        chunk_meta = []  # per-chunk metadata from file-level metadata

        # Group chunks by source to build proper prev/next links
        source_counts = {}
        for c in chunks:
            src = c["source"]
            idx = source_counts.get(src, 0)
            source_counts[src] = idx + 1

        source_indices = {}
        for c in chunks:
            src = c["source"]
            idx = source_indices.get(src, 0)
            total = source_counts[src]
            label = f"{src}"
            if total > 1:
                label = f"{src} (part {idx+1}/{total})"
            texts.append(f"{label}\n{c['text']}")
            doc_map.append((src, idx, total))
            source_indices[src] = idx + 1

            # Carry file-level metadata to each chunk
            fm = file_metadata.get(src, {})
            chunk_meta.append({
                "owner": fm.get("owner", ""),
                "description": fm.get("description", ""),
                "tags": fm.get("tags", []),
                "source": src,
            })

        # Embed
        _update({"status": "building", "progress": 0.05})
        embeddings = embed_texts(texts, batch_size=32)
        _update({"chunks_done": n, "progress": 0.7})

        # Sign-zero signatures (768 bits = 96 bytes each)
        sign_bits = (embeddings > 0).astype(np.uint8)
        packed_signs = np.packbits(sign_bits, axis=1)  # (n, 96)

        # Build per-pattern metadata (h-row style)
        per_pattern_meta = []
        for i, cm in enumerate(chunk_meta):
            per_pattern_meta.append({
                "v": 1,
                "owner": cm["owner"],
                "description": cm["description"],
                "tags": cm["tags"],
                "source": cm["source"],
                "chunk": doc_map[i][1],
                "chunks": doc_map[i][2],
                "created_at": time.time(),
                "tombstone": False,
                "content_type": "document",
            })

        # Build Pattern 0 — cart manifest
        unique_sources = {}
        for cm in chunk_meta:
            src = cm["source"]
            if src not in unique_sources:
                unique_sources[src] = {
                    "name": src,
                    "owner": cm["owner"],
                    "description": cm["description"],
                    "tags": cm["tags"],
                    "chunks": source_counts.get(src, 0),
                }

        pattern0_data = {
            "cart_name": cart_name,
            "creator": creator or "Cart Builder (cloud)",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "file_count": len(unique_sources),
            "total_chunks": n,
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 768,
            "files": list(unique_sources.values()),
            # Cart-level metadata surfaced in the Pattern-0 TOC panel. Description
            # and agent_briefing get generic-fallback text from the /build handler
            # when the caller doesn't supply one, so these fields are always
            # populated on user-built carts (Andy 2026-07-02 rich Pattern-0 path).
            "description": description or "",
            "agent_briefing": agent_briefing or "",
            "owner": owner or "",
            "tags": list(tags or []),
        }

        _update({"progress": 0.85})

        # Save cartridge as NPZ with full metadata
        os.makedirs(output_dir, exist_ok=True)
        cart_path = os.path.join(output_dir, f"{cart_name}.cart.npz")

        np.savez_compressed(
            cart_path,
            embeddings=embeddings.astype(np.float32),
            passages=np.array(texts, dtype=object),
            sign_bits=packed_signs,
            pattern0=json.dumps(pattern0_data),
            per_pattern_meta=json.dumps(per_pattern_meta),
        )

        # Manifest
        with open(cart_path, "rb") as f:
            cart_hash = hashlib.sha256(f.read()).hexdigest()
        manifest = {
            "version": 1,
            "count": n,
            "fingerprint": cart_hash[:16],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        manifest_path = os.path.join(output_dir, f"{cart_name}.cart_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Also save metadata sidecar for the cart builder to reload
        meta_sidecar = {src: {
            "owner": info["owner"],
            "description": info["description"],
            "tags": info["tags"],
        } for src, info in unique_sources.items() if info["owner"] or info["description"] or info["tags"]}
        if meta_sidecar:
            meta_path = os.path.join(output_dir, f"{cart_name}.meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta_sidecar, f, indent=2)

        size_mb = os.path.getsize(cart_path) / (1024 * 1024)

        elapsed = (time.time() - t0) * 1000
        _update({
            "status": "done",
            "progress": 1.0,
            "chunks_done": n,
            "elapsed_ms": int(elapsed),
            "cart_path": cart_path,
            "cart_size_mb": round(size_mb, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        _update({"status": "error", "error": str(e)})


def search_cart(cart_path: str, query: str, top_k: int = 5) -> list[dict]:
    """Search a built cart with cosine similarity. Handles NPZ and PKL formats."""
    model = get_embedder()
    q_emb = model.encode([f"search_query: {query}"], convert_to_numpy=True).astype(np.float32)

    if cart_path.endswith(".pkl"):
        import pickle
        with open(cart_path, "rb") as f:
            pkl = pickle.load(f)
        embeddings = pkl.get("embeddings", np.array([]))
        passages = np.array(pkl.get("passages", pkl.get("texts", [])), dtype=object)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
    else:
        data = np.load(cart_path, allow_pickle=True)
        embeddings = data["embeddings"]
        passages = data["passages"]

    if embeddings.size == 0 or len(embeddings.shape) < 2 or embeddings.shape[1] == 0:
        return [{"rank": 1, "score": 0, "text": "This cart has no embeddings — search unavailable.", "source": ""}]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms
    q_norm = q_emb / (np.linalg.norm(q_emb) or 1)
    scores = (normed @ q_norm.T).flatten()

    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx):
        passage = str(passages[idx])
        # Extract source from first line
        first_line = passage.split("\n")[0] if "\n" in passage else ""
        results.append({
            "rank": rank + 1,
            "score": round(float(scores[idx]), 4),
            "text": passage,
            "source": first_line,
        })
    return results
