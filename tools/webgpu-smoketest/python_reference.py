#!/usr/bin/env python3
"""
python_reference.py — Reference embedding for browser-vs-Python parity check.

Embeds the same sentence as embedder-test.html using nomic-embed-text-v1.5 via
SentenceTransformer. The values printed here are the ground truth that the
in-browser transformers.js output should match (within float precision).

Usage:
    cd vector-plus-studio-repo
    python tools/webgpu-smoketest/python_reference.py

The COPY_BLOCK at the end mirrors the browser page's COPY block format so
the two outputs can be diffed visually.

Parity criteria (WebGPU smoke test):
  - DIMS must match exactly (768 for Nomic v1.5)
  - L2 norm should match within ~1% (numerical drift from ONNX vs PyTorch)
  - First-10 values: signs must match, magnitudes within ~5%
  - If full-vector cosine similarity falls below 0.95, browser-built carts
    will not be cross-compatible with server-built ones — DEAL-BREAKER.
"""
import sys
import time
import numpy as np

SENTENCE = 'search_document: this is a test sentence about transformer attention'

def main():
    print(f"[1/3] Loading nomic-embed-text-v1.5 via SentenceTransformer...", file=sys.stderr)
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"      loaded in {load_ms:.0f}ms", file=sys.stderr)

    print(f"[2/3] Embedding...", file=sys.stderr)
    t1 = time.perf_counter()
    emb = model.encode(SENTENCE, normalize_embeddings=False)
    embed_ms = (time.perf_counter() - t1) * 1000
    print(f"      embedded in {embed_ms:.0f}ms", file=sys.stderr)
    print(f"[3/3] Inspecting...", file=sys.stderr)

    norm = float(np.linalg.norm(emb))
    first10 = ', '.join(f"{v:.6f}" for v in emb[:10])

    print()
    print(f"=== COPY EVERYTHING BELOW FOR PARITY CHECK ===")
    print(f"SENTENCE: {SENTENCE}")
    print(f"BACKEND: pytorch-cpu (sentence-transformers)")
    print(f"MODEL_LOAD_MS: {load_ms:.0f}")
    print(f"EMBED_MS: {embed_ms:.0f}")
    print(f"DIMS: [1, {emb.shape[0]}]")
    print(f"L2_NORM: {norm:.6f}")
    print(f"FIRST_10: [{first10}]")
    print(f"=== END COPY BLOCK ===")

    # Also dump the full vector in case we need to compute cosine similarity
    # against the JS output. Save to a sibling file rather than printing —
    # 768 floats is too noisy for stdout.
    out_path = 'tools/webgpu-smoketest/python_reference_full.npy'
    np.save(out_path, emb)
    print(f"\n[saved] full vector to {out_path}", file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
