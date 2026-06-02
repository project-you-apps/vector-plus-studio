"""
Search modes for Vector+ Studio.

Extracted from vector_plus_studio_v83.py lines 623-840 + 1143-1210.
All functions take engine state as parameters (no st.session_state).
"""

import numpy as np
from .engine import engine, SETTLE_FRAMES, SIG_SETTLE_FRAMES, TextRegionEncoder

ASSOCIATE_SETTLE_FRAMES = 30  # Full physics settle for associative search

# DLL wrapper disabled for now — Python path proven working, DLL untested.
# TODO: Debug associate.dll encode/decode/Hamming pipeline, then re-enable.
_assoc_session = None
_assoc_corpus_hash = None
_ASSOC_DLL_AVAILABLE = False

# 2026-05-30: Route the Associate Python fallback through the two-sided v83
# algorithm. The pre-existing one-sided code (query through physics; candidates
# are static stored sign vectors) was identified as a refactor regression that
# produces fixed-attractor failure (~50% sign preservation). The two-sided path
# below re-imprints each cosine-pre-filter candidate through physics independently
# and scores via cosine in analog-region-mean decoded space — matches the v83
# Reed-Richards / Beatles-Divorce algorithm.
#
# Flip to False to fall back to the old one-sided path for A/B testing.
# Receipts file: experiments/walk-test-2026-05-29/walk_compare_2026-05-30_104357.txt
USE_TWO_SIDED_ASSOCIATE = True
ASSOCIATE_CANDIDATE_POOL = 50  # cosine-pre-filter before per-candidate physics


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "what", "who", "how", "when",
    "where", "why", "which", "of", "in", "on", "for", "to", "and",
    "or", "it", "as", "at", "by", "from", "about", "with", "me",
    "tell", "show", "find", "search",
}

HAMMING_BLEND = 0.3  # 70% cosine + 30% sign_zero Hamming


def simple_stem(word: str) -> str:
    word = word.lower()
    for suffix in ['ings', 'ing', 'tion', 'sion', 'ness', 'ment', 'able',
                    'ible', 'ful', 'less', 'ous', 'ive', 'ed', 'es', 's', 'ly']:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def keyword_matches(keyword: str, text: str) -> bool:
    if keyword in text:
        return True
    stemmed_kw = simple_stem(keyword)
    for word in text.split():
        clean = word.strip(".,!?;:'\"")
        if simple_stem(clean) == stemmed_kw:
            return True
    return False


def clean_query(query: str) -> tuple[str, list[str]]:
    """Clean query and extract keywords. Returns (clean_text, keywords)."""
    words = [w.lower().strip("?.,!") for w in query.split() if len(w) > 2]
    keywords = [w for w in words if w not in STOPWORDS]
    clean = " ".join(keywords) if keywords else query
    return clean, keywords


def _keyword_rerank(results: list[dict], keywords: list[str], passages: list[str],
                    extra_pool: int = 10) -> list[dict]:
    """Boost results that contain query keywords."""
    if not keywords:
        return results
    for r in results:
        idx = r['idx']
        if idx < len(passages):
            text_lower = passages[idx].lower()
            hits = sum(1 for kw in keywords if keyword_matches(kw, text_lower))
            if hits > 0:
                r['score'] += min(hits * 0.04, 0.12)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Cosine + Hamming blend search (ported from Membot, no GPU needed)
# ---------------------------------------------------------------------------

def hamming_blend_search(q_emb: np.ndarray, embeddings: np.ndarray,
                         passages: list[str], top_k: int = 10,
                         keywords: list[str] | None = None) -> list[dict]:
    """70% cosine + 30% sign-zero Hamming + keyword reranking.

    Matches Membot's production search pipeline. No GPU needed.
    """
    # Cosine similarity (memory-efficient: no full-size float32 intermediates).
    # The naive `np.linalg.norm(embeddings, ...)` + `embeddings / e_norms`
    # pattern allocates two ~(N x 768) float32 temps every search — ~350MB on
    # a 60K-passage cart, which OOMs a 4GB droplet. einsum + scalar-divide on
    # an already-normalized query produces identical results with <1MB temp.
    q_normalized = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.sqrt(np.einsum('ij,ij->i', embeddings, embeddings)) + 1e-9
    cos_scores = (embeddings @ q_normalized) / e_norms

    # Sign-zero Hamming similarity
    q_bin = (q_emb > 0).astype(np.uint8)
    corpus_bin = (embeddings > 0).astype(np.uint8)
    n_bits = corpus_bin.shape[1]  # 768
    xor = np.bitwise_xor(q_bin, corpus_bin)
    ham_scores = 1.0 - xor.sum(axis=1).astype(np.float32) / n_bits

    # Blend: 70% cosine + 30% Hamming
    blended = (1.0 - HAMMING_BLEND) * cos_scores + HAMMING_BLEND * ham_scores

    # Keyword reranking — widen candidates, boost exact matches
    candidate_k = min(max(top_k * 4, 20), len(blended))
    candidate_idx = np.argsort(blended)[-candidate_k:][::-1]

    results = []
    for i in candidate_idx:
        i = int(i)
        if i in engine.deleted_ids:
            continue
        base_score = float(blended[i])
        if base_score < 0.05:
            continue
        kw_boost = 0.0
        if keywords:
            text_lower = passages[i].lower() if i < len(passages) else ""
            hits = sum(1 for kw in keywords if keyword_matches(kw, text_lower))
            kw_boost = min(hits * 0.03, 0.12)
        results.append({
            'idx': i,
            'score': base_score + kw_boost,
            'cosine_score': float(cos_scores[i]),
            'hamming_score': float(ham_scores[i]),
            'keyword_boost': kw_boost,
            'physics_score': None,
            'from_lattice': False,
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Cosine search (fast, CPU only)
# ---------------------------------------------------------------------------

def cosine_search(q_emb: np.ndarray, embeddings: np.ndarray,
                  passages: list[str], top_k: int = 10,
                  keywords: list[str] | None = None) -> list[dict]:
    """Pure cosine similarity search."""
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    scores = np.dot(e_unit, q_norm)

    top_n = min(top_k + 10, len(embeddings))  # extra for keyword rerank
    top_indices = np.argsort(scores)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        idx = int(idx)
        if idx in engine.deleted_ids:
            continue
        results.append({
            'idx': idx,
            'score': float(scores[idx]),
            'cosine_score': float(scores[idx]),
            'physics_score': None,
            'from_lattice': False,
        })

    if keywords:
        results = _keyword_rerank(results, keywords, passages)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Protected Multimodal search (Smart mode)
# ---------------------------------------------------------------------------

def protected_multimodal_search(q_emb: np.ndarray, embeddings: np.ndarray,
                                 passages: list[str], compressed_lens: list[int],
                                 alpha: float = 0.7, top_k: int = 10,
                                 keywords: list[str] | None = None) -> list[dict]:
    """
    Protected Multimodal search -- text survives settle via V7.2 protected rows.
    Scoring: blended = alpha * physics + (1-alpha) * cosine
    """
    ml = engine.ml
    enc = engine.combined_encoder
    if not ml or not enc:
        return cosine_search(q_emb, embeddings, passages, top_k, keywords)

    # Cosine pre-filter
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    cosine_scores = np.dot(e_unit, q_norm)

    top_n = min(50, len(embeddings))
    candidate_idx = np.argsort(cosine_scores)[-top_n:][::-1]

    results = []
    with engine.lock:
        engine.enable_text_protection()

        # Settle query
        ml.reset()
        query_pattern, _ = enc.encode(q_emb, "")
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        query_settled = ml.recall()
        q_unit = enc.decode_embedding_only(query_settled)
        q_unit = q_unit / (np.linalg.norm(q_unit) + 1e-9)

        for idx in candidate_idx:
            idx = int(idx)
            if idx in engine.deleted_ids:
                continue

            comp_len = compressed_lens[idx] if idx < len(compressed_lens) else 0
            pattern, _ = enc.encode(embeddings[idx], passages[idx])

            ml.reset()
            ml.imprint_pattern(pattern)
            ml.settle(frames=SETTLE_FRAMES, learn=False)

            post_settle = ml.recall()
            recovered_text = enc.decode_text_only(post_settle, comp_len)
            recovered_emb = enc.decode_embedding_only(post_settle)

            r_norm = recovered_emb / (np.linalg.norm(recovered_emb) + 1e-9)
            physics_score = float(np.dot(q_unit, r_norm))
            blended = alpha * physics_score + (1.0 - alpha) * cosine_scores[idx]

            results.append({
                'idx': idx,
                'score': float(blended),
                'cosine_score': float(cosine_scores[idx]),
                'physics_score': physics_score,
                'recovered_text': recovered_text,
                'from_lattice': True,
            })

        engine.disable_text_protection()

    if keywords:
        results = _keyword_rerank(results, keywords, passages)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Pure Brain search (L2 signatures)
# ---------------------------------------------------------------------------

def pure_brain_search(q_emb: np.ndarray, top_k: int = 10,
                      keywords: list[str] | None = None) -> list[dict]:
    """
    Pure Brain search -- uses L2 signatures for ranking.
    Works without the PKL embedding database.

    Uses a fresh engine per query (same approach as Associate search) to avoid
    accumulated fatigue/BCM state from training. Encodes query with region-fill
    only (no zlib text noise) to match the embedding content in stored sigs.
    """
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
    from .cartridge_io import find_companion_file

    signatures = engine.signatures
    compressed_texts = engine.compressed_texts
    passages = engine.passages
    text_encoder = TextRegionEncoder()

    if signatures is None or not engine.signatures_loaded:
        return []

    rf_enc = engine.encoder
    if not rf_enc or not engine.mounted_name:
        return []

    # Find saved brain
    brain_path = find_companion_file(engine.mounted_name, "_brain.npy")
    if not brain_path:
        return []

    n_dims = 768

    # Settle query on FRESH engine to get signature (L3 if available, else L2)
    sig_dim = signatures.shape[1] if len(signatures.shape) > 1 else 4096

    # Fresh engine — no accumulated fatigue/BCM from training
    temp_ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=0)
    temp_ml.set_profile("quality")
    temp_ml.set_row_physics(63, temp_ml.ROW_FULLY_PROTECTED)
    temp_ml.load_brain(brain_path)

    # Encode query as region-fill only (no zlib text noise)
    grid = np.zeros((64, 64, 64, 64), dtype=np.float32)
    for i in range(n_dims):
        if q_emb[i] > 0:
            grid[i // 64, i % 64, :, :] = 1.0
    query_pattern = grid.reshape(4096, 4096)

    temp_ml.reset()
    temp_ml.imprint_pattern(query_pattern)
    temp_ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
    if sig_dim >= 65536:
        query_sig = temp_ml.recall_l3().flatten()
    else:
        query_sig = temp_ml.recall_l2().flatten()
    del temp_ml  # Free GPU memory

    # Compare signatures
    q_sig_norm = query_sig / (np.linalg.norm(query_sig) + 1e-9)
    sig_norms = np.linalg.norm(signatures, axis=1, keepdims=True) + 1e-9
    sig_unit = signatures / sig_norms
    sig_scores = np.dot(sig_unit, q_sig_norm)

    top_n = min(top_k + 10, len(signatures))
    top_indices = np.argsort(sig_scores)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        idx = int(idx)
        if idx in engine.deleted_ids:
            continue

        # Recover text from compressed texts
        recovered_text = None
        if idx < len(compressed_texts) and compressed_texts[idx] is not None:
            try:
                recovered_text = text_encoder.decompress_text(bytes(compressed_texts[idx]))
            except Exception:
                pass
        if not recovered_text and idx < len(passages):
            recovered_text = passages[idx]

        results.append({
            'idx': idx,
            'score': float(sig_scores[idx]),
            'cosine_score': None,
            'physics_score': float(sig_scores[idx]),
            'recovered_text': recovered_text,
            'from_lattice': True,
        })

    if keywords:
        results = _keyword_rerank(results, keywords, passages)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Associate search (CAM: encode → settle → decode → Hamming rank)
# ---------------------------------------------------------------------------

def _get_assoc_session(embeddings):
    """Lazily create/update the DLL-backed associate session."""
    global _assoc_session, _assoc_corpus_hash

    if not _ASSOC_DLL_AVAILABLE or not engine.ml:
        return None

    # Hash corpus by shape + first/last embedding to detect changes
    corpus_hash = (embeddings.shape, embeddings[0, :4].tobytes(),
                   embeddings[-1, :4].tobytes())

    if _assoc_session is not None and _assoc_corpus_hash == corpus_hash:
        return _assoc_session

    # Create or recreate session
    try:
        if _assoc_session is not None:
            _assoc_session.destroy()
        _assoc_session = AssociateSession(engine.ml)
        _assoc_session.load_corpus(embeddings)
        _assoc_corpus_hash = corpus_hash
        return _assoc_session
    except Exception as e:
        print(f"[associate] DLL init failed, falling back to Python: {e}")
        _assoc_session = None
        return None


def associate_search(q_emb: np.ndarray, embeddings: np.ndarray,
                     passages: list[str], top_k: int = 10,
                     keywords: list[str] | None = None) -> list[dict]:
    """
    Associative search via lattice physics.

    Encode the query as a region-fill pattern, imprint it on the lattice,
    settle with full physics (30 frames), decode the settled sign vector,
    and rank all stored patterns by Hamming similarity to the settled result.

    This finds cross-domain associations that cosine similarity misses —
    e.g. "earthquakes" → Poseidon.

    Tries associate.dll first for GPU-accelerated pipeline; falls back
    to inline Python if the DLL is unavailable.
    """
    # ── DLL path (fast, GPU-accelerated) ──────────────────────
    assoc = _get_assoc_session(embeddings)
    if assoc is not None:
        try:
            indices, scores, raw_scores, sign_pct = assoc.search(
                q_emb, settle_frames=ASSOCIATE_SETTLE_FRAMES, top_k=top_k
            )
            results = []
            for i in range(len(indices)):
                idx = int(indices[i])
                if idx in engine.deleted_ids:
                    continue
                results.append({
                    'idx': idx,
                    'score': float(scores[i]),
                    'cosine_score': float(raw_scores[i]),
                    'physics_score': float(scores[i]),
                    'from_lattice': True,
                })
            if keywords:
                results = _keyword_rerank(results, keywords, passages)
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"[associate] DLL search failed, falling back: {e}")

    # ── Python fallback: FRESH ENGINE per query (matches test_cam_poseidon.py) ──
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
    from .cartridge_io import find_companion_file

    rf_enc = engine.encoder
    if not rf_enc or not engine.mounted_name:
        return hamming_blend_search(q_emb, embeddings, passages, top_k, keywords)

    # Find saved brain (post-training Hebbian weights)
    brain_path = find_companion_file(engine.mounted_name, "_brain.npy")
    if not brain_path:
        print("[Associate] No brain file found, falling back to Hamming")
        return hamming_blend_search(q_emb, embeddings, passages, top_k, keywords)

    # ── Two-sided v83 algorithm (the working path; restored 2026-05-30) ──
    if USE_TWO_SIDED_ASSOCIATE:
        return _associate_search_two_sided(
            q_emb, embeddings, passages, brain_path, top_k, keywords,
        )

    n_dims = embeddings.shape[1]  # 768

    # Pre-compute stored sign vectors
    stored_signs = (embeddings > 0).astype(np.uint8)

    # === EXACT test_cam_poseidon.py approach ===
    # Fresh engine — no accumulated fatigue/BCM from training or sig capture
    temp_ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=0)
    temp_ml.set_profile("quality")
    temp_ml.set_row_physics(63, temp_ml.ROW_FULLY_PROTECTED)  # HIPPO_ROW
    temp_ml.load_brain(brain_path)

    # Encode query as region-fill — MUST use 4D→reshape layout to match
    # standalone test_cam_poseidon.py (and the brain trained with that layout).
    # The 4D reshape produces a different physical memory layout than direct
    # 2D tile filling: reshape puts each region as a full row, tile puts each
    # region as a 64×64 block. The Hebbian weights are layout-specific.
    grid = np.zeros((64, 64, 64, 64), dtype=np.float32)
    for i in range(n_dims):
        if q_emb[i] > 0:
            grid[i // 64, i % 64, :, :] = 1.0
    query_pattern = grid.reshape(4096, 4096)

    temp_ml.reset()
    temp_ml.imprint_pattern(query_pattern)
    temp_ml.settle(frames=ASSOCIATE_SETTLE_FRAMES, learn=False)
    settled = temp_ml.recall()
    del temp_ml  # Free GPU memory

    # Decode exactly like test_cam_poseidon.py: decode_sign_vector
    grid = settled.reshape(64, 64, 64, 64)
    settled_signs = np.zeros(n_dims, dtype=np.uint8)
    for i in range(n_dims):
        settled_signs[i] = 1 if np.mean(grid[i // 64, i % 64, :, :]) > 0.5 else 0

    # Raw query signs for comparison
    q_signs = (q_emb > 0).astype(np.uint8)

    # Sign preservation: how many query signs survived settle
    sign_pres = float(np.mean(q_signs == settled_signs))

    # Hamming similarity: 1 - (XOR sum / n_dims)
    xor = np.bitwise_xor(settled_signs, stored_signs)
    ham_scores = 1.0 - xor.sum(axis=1).astype(np.float32) / n_dims

    # Also compute raw (pre-settle) Hamming for comparison metadata
    xor_raw = np.bitwise_xor(q_signs, stored_signs)
    raw_ham = 1.0 - xor_raw.sum(axis=1).astype(np.float32) / n_dims

    # Diagnostic: compare VPS path with standalone test results
    raw_top5 = np.argsort(raw_ham)[-5:][::-1]
    ham_top5 = np.argsort(ham_scores)[-5:][::-1]
    print(f"[Associate] Sign preservation: {sign_pres:.1%}")
    print(f"[Associate] Settled lattice stats: min={settled.min():.3f} max={settled.max():.3f} "
          f"mean={settled.mean():.4f} nonzero={np.count_nonzero(settled > 0.5)}")
    print(f"[Associate] Raw Hamming top-5:")
    for i, idx in enumerate(raw_top5):
        title = passages[idx].splitlines()[0][:50] if idx < len(passages) else "?"
        print(f"  {i+1}. {raw_ham[idx]:.4f}  {title}")
    print(f"[Associate] Settled Hamming top-5:")
    for i, idx in enumerate(ham_top5):
        title = passages[idx].splitlines()[0][:50] if idx < len(passages) else "?"
        print(f"  {i+1}. {ham_scores[idx]:.4f}  {title}")

    # Rank by settled Hamming
    candidate_k = min(max(top_k * 4, 20), len(ham_scores))
    candidate_idx = np.argsort(ham_scores)[-candidate_k:][::-1]

    results = []
    for idx in candidate_idx:
        idx = int(idx)
        if idx in engine.deleted_ids:
            continue
        results.append({
            'idx': idx,
            'score': float(ham_scores[idx]),
            'cosine_score': float(raw_ham[idx]),  # raw Hamming as baseline
            'physics_score': float(ham_scores[idx]),
            'from_lattice': True,
        })

    if keywords:
        results = _keyword_rerank(results, keywords, passages)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Two-sided Associate (v83 algorithm — the algorithm that produced the historic
# Reed Richards / Beatles-Divorce / Paris-Idaho receipts)
# ---------------------------------------------------------------------------
#
# Both the query AND each cosine-pre-filter candidate go through reset + imprint
# + settle + decode independently. Lattice Hebbian weights modulate each
# candidate's settled state; cosine in analog-region-mean decoded space scores
# the query-to-candidate semantic distance AFTER physics.
#
# The pre-2026-05-30 code path (above) compared a single settled-query sign
# vector against pre-computed static stored sign vectors. That one-sided path
# produces fixed-attractor failures when the query settles to a brain-dominated
# attractor (~50% sign preservation). The two-sided path stays in the working
# regime because every candidate gets its own settled state.
#
# Receipts: experiments/walk-test-2026-05-29/walk_compare_2026-05-30_104357.txt
#           experiments/walk-test-2026-05-29/walk_compare_2026-05-30_120856.txt


def _ab_encode_region_fill(emb: np.ndarray, n_dims: int = 768) -> np.ndarray:
    """Region-fill encode: each positive Nomic dim fills its 64x64 region."""
    grid = np.zeros((64, 64, 64, 64), dtype=np.float32)
    for i in range(n_dims):
        if emb[i] > 0:
            grid[i // 64, i % 64, :, :] = 1.0
    return grid.reshape(4096, 4096)


def _ab_decode_region_fill_analog(lattice: np.ndarray, n_dims: int = 768) -> np.ndarray:
    """Decode settled lattice to ANALOG region-mean vector in [0,1].

    Critical: binary > 0.5 thresholding collapses sparse settled states to
    all-zero vectors (cosine of all-zero = 0). Raw region-mean preserves
    continuous activation signal — matches v83's thermometer decode
    (active_bits / region_size).
    """
    if lattice.ndim == 1:
        lattice = lattice.reshape(4096, 4096)
    grid = lattice.reshape(64, 64, 64, 64)
    out = np.zeros(n_dims, dtype=np.float32)
    for i in range(n_dims):
        region = grid[i // 64, i % 64, :, :]
        out[i] = float(np.mean(region))
    return out


def _associate_search_two_sided(q_emb: np.ndarray, embeddings: np.ndarray,
                                 passages: list, brain_path: str,
                                 top_k: int, keywords: list | None):
    """v83 two-sided Associate. Returns same shape as legacy associate_search."""
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
    n_dims = embeddings.shape[1]

    # Init lattice ONCE, reuse across the candidate loop
    temp_ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=0)
    temp_ml.set_profile("quality")
    temp_ml.set_row_physics(63, temp_ml.ROW_FULLY_PROTECTED)  # HIPPO_ROW
    temp_ml.load_brain(brain_path)

    # Query through physics
    q_pattern = _ab_encode_region_fill(q_emb, n_dims)
    temp_ml.reset()
    temp_ml.imprint_pattern(q_pattern)
    temp_ml.settle(frames=ASSOCIATE_SETTLE_FRAMES, learn=False)
    settled_q = temp_ml.recall()
    q_decoded = _ab_decode_region_fill_analog(settled_q, n_dims)
    q_unit = q_decoded / (float(np.linalg.norm(q_decoded)) + 1e-9)

    # Sign preservation diagnostic (still using binary decode for the metric)
    q_signs = (q_emb > 0).astype(np.uint8)
    q_decoded_bin = (q_decoded > 0.5).astype(np.uint8)
    sign_pres = float(np.mean(q_signs == q_decoded_bin))
    print(f"[Associate v83] Sign preservation: {sign_pres:.1%}")

    # Cosine pre-filter to bound per-candidate physics cost
    q_normalized = q_emb / (float(np.linalg.norm(q_emb)) + 1e-9)
    e_norms = np.sqrt(np.einsum('ij,ij->i', embeddings, embeddings)) + 1e-9
    cosine_sims = (embeddings @ q_normalized) / e_norms
    pool_size = min(ASSOCIATE_CANDIDATE_POOL, len(embeddings))
    pool_idx = np.argpartition(-cosine_sims, pool_size - 1)[:pool_size]

    # Per-candidate physics
    physics_scores: dict[int, float] = {}
    cosine_lookup: dict[int, float] = {}
    for idx in pool_idx:
        idx = int(idx)
        if idx in engine.deleted_ids:
            continue
        cand_pattern = _ab_encode_region_fill(embeddings[idx], n_dims)
        temp_ml.reset()
        temp_ml.imprint_pattern(cand_pattern)
        temp_ml.settle(frames=ASSOCIATE_SETTLE_FRAMES, learn=False)
        settled_c = temp_ml.recall()
        c_decoded = _ab_decode_region_fill_analog(settled_c, n_dims)
        c_unit = c_decoded / (float(np.linalg.norm(c_decoded)) + 1e-9)
        physics_scores[idx] = float(np.dot(q_unit, c_unit))
        cosine_lookup[idx] = float(cosine_sims[idx])

    del temp_ml

    # Assemble result records in the same shape as the legacy path
    results = [
        {
            'idx': idx,
            'score': score,
            'cosine_score': cosine_lookup[idx],
            'physics_score': score,
            'from_lattice': True,
        }
        for idx, score in physics_scores.items()
    ]

    if keywords:
        results = _keyword_rerank(results, keywords, passages)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Associate + Walk: layer "you may have missed" associations on top of Associate
# ---------------------------------------------------------------------------

# Cache for normalized embeddings (keyed by id(embeddings) so a fresh cart
# triggers re-normalization; bounded size to handle multiple mounted carts).
_NORM_CACHE: dict[int, np.ndarray] = {}


def _get_normed_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return L2-normalized embeddings, cached by id(embeddings)."""
    key = id(embeddings)
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normed = embeddings / norms
    _NORM_CACHE[key] = normed
    # Cap cache size to avoid leak across many cart mounts
    while len(_NORM_CACHE) > 4:
        _NORM_CACHE.pop(next(iter(_NORM_CACHE)))
    return normed


def associate_search_with_walk(q_emb: np.ndarray, embeddings: np.ndarray,
                               passages: list[str], top_k: int = 10,
                               walk_top_k: int = 10, walk_min_hits: int = 2,
                               walk_max_show: int = 10,
                               keywords: list[str] | None = None) -> list[dict]:
    """
    Associate search + walk-the-results layer.

    Runs the standard `associate_search()` to get primary results via lattice
    physics (the editorial-association engine — Divorce-between-John-and-Paul,
    NASA/Philip-K-Dick for Reed Richards). Then performs a "walk" hop: for each
    primary item, computes its cosine-nearest neighbors in raw embedding space.
    Items walked-to multiple times but absent from primary surface as "you may
    have missed".

    Returns a flat list: primary items first (with 'from_walk'=False), then
    walked items (with 'from_walk'=True and 'walk_count' set). Existing
    associate_search() is untouched; this is opt-in via a new search mode.

    Walk hop uses cosine for speed. Running full physics per primary item
    (the "physics walk") would multiply query time by ~top_k and require
    GPU access for every hop — cosine walk is ~50ms for 100k corpus on CPU
    and surfaces depth that pure physics-Associate misses.
    """
    # 1) Primary via full Associate (physics)
    primary = associate_search(q_emb, embeddings, passages, top_k=top_k, keywords=keywords)
    primary_ids = {r['idx'] for r in primary}

    # 2) Walk hop: cosine-nearest from each primary item
    embeddings_n = _get_normed_embeddings(embeddings)

    walk_counts: dict[int, int] = {}
    walk_score_sum: dict[int, float] = {}

    for r in primary:
        idx = r['idx']
        if idx < 0 or idx >= len(embeddings_n):
            continue
        item_vec = embeddings_n[idx]
        sims = embeddings_n @ item_vec
        sims[idx] = -np.inf  # exclude self
        top_idx = np.argpartition(-sims, walk_top_k)[:walk_top_k]
        for nidx in top_idx:
            nidx = int(nidx)
            if nidx in engine.deleted_ids:
                continue
            walk_counts[nidx] = walk_counts.get(nidx, 0) + 1
            walk_score_sum[nidx] = walk_score_sum.get(nidx, 0.0) + float(sims[nidx])

    # 3) Filter: items appearing >= walk_min_hits times, not in primary
    walked = []
    for idx, cnt in walk_counts.items():
        if idx in primary_ids:
            continue
        if cnt < walk_min_hits:
            continue
        avg_score = walk_score_sum[idx] / cnt
        walked.append({
            'idx': idx,
            'score': float(avg_score),
            'cosine_score': float(avg_score),
            'physics_score': 0.0,
            'from_lattice': False,
            'from_walk': True,
            'walk_count': cnt,
        })
    walked.sort(key=lambda x: (-x['walk_count'], -x['score']))
    walked = walked[:walk_max_show]

    # 4) Mark primary items as not from walk and emit combined list
    for r in primary:
        r.setdefault('from_walk', False)
        r.setdefault('walk_count', 0)

    return primary + walked


# ---------------------------------------------------------------------------
# Unified search dispatcher
# ---------------------------------------------------------------------------

def search(query: str, mode: str = "smart", alpha: float = 0.7,
           top_k: int = 10) -> tuple[list[dict], str]:
    """
    Dispatch search to the right mode.
    Returns (results, mode_label).
    """
    clean_q, keywords = clean_query(query)

    embeddings = engine.embeddings
    passages = engine.passages
    compressed_lens = engine.compressed_lens

    if mode == "associate":
        # Associate uses the RAW query — no stopword stripping.
        # Stopword removal changes the embedding signs which alters the
        # lattice settle trajectory. Keywords still used for reranking.
        q_emb = engine.embed_query(query)
        if embeddings is not None:
            results = associate_search(q_emb, embeddings, passages, top_k, keywords)
            mode_label = f"Associate ({ASSOCIATE_SETTLE_FRAMES}f settle)"
        else:
            results = []
            mode_label = "Associate (no embeddings)"

    elif mode == "hamming":
        q_emb = engine.embed_query(clean_q)
        results = hamming_blend_search(q_emb, embeddings, passages, top_k, keywords)
        kw_label = f" +kw" if keywords else ""
        mode_label = f"Cosine+Hamming 70/30{kw_label}"

    elif mode == "pure_brain" and engine.signatures_loaded:
        q_emb = engine.embed_query(clean_q)
        results = pure_brain_search(q_emb, top_k, keywords)
        sig_dim = engine.signatures.shape[1] if engine.signatures is not None and len(engine.signatures.shape) > 1 else 4096
        mode_label = f"Pure Brain ({'L3' if sig_dim >= 65536 else 'L2'} signatures)"

    elif mode == "smart" and engine.physics_trained and engine.gpu_available:
        q_emb = engine.embed_query(clean_q)
        if embeddings is not None and len(compressed_lens) > 0:
            results = protected_multimodal_search(
                q_emb, embeddings, passages, compressed_lens,
                alpha=alpha, top_k=top_k, keywords=keywords
            )
            mode_label = f"Smart Search (blend={alpha:.2f})"
        else:
            results = cosine_search(q_emb, embeddings, passages, top_k, keywords)
            mode_label = "Cosine (no compressed lens)"

    else:
        # Fast / fallback
        q_emb = engine.embed_query(clean_q)
        if embeddings is not None:
            results = cosine_search(q_emb, embeddings, passages, top_k, keywords)
            mode_label = "Fast Search (cosine)"
        else:
            results = []
            mode_label = "No data"

    # Attach text and hippocampus navigation to results
    hippo = engine.hippocampus
    for r in results:
        idx = r['idx']
        text = r.get('recovered_text') or (passages[idx] if idx < len(passages) else "")
        lines = text.splitlines() if text else ["[empty]"]
        r['title'] = lines[0][:100]
        r['preview'] = " ".join(lines[1:3])[:200] if len(lines) > 1 else ""
        r['full_text'] = text or ""
        if 'recovered_text' in r:
            del r['recovered_text']

        # PREV/NEXT from hippocampus metadata
        if hippo is not None and idx < len(hippo):
            r['prev_idx'] = hippo[idx].get('prev')
            r['next_idx'] = hippo[idx].get('next')

    return results, mode_label
