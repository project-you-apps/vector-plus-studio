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
    # Cosine similarity
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    q_norm = np.linalg.norm(q_emb) + 1e-9
    cos_scores = np.dot(embeddings / e_norms, q_emb / q_norm)

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

    # Attach text to results
    for r in results:
        idx = r['idx']
        text = r.get('recovered_text') or (passages[idx] if idx < len(passages) else "")
        lines = text.splitlines() if text else ["[empty]"]
        r['title'] = lines[0][:100]
        r['preview'] = " ".join(lines[1:3])[:200] if len(lines) > 1 else ""
        r['full_text'] = text or ""
        if 'recovered_text' in r:
            del r['recovered_text']

    return results, mode_label
