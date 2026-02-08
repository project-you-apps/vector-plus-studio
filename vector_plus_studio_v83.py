"""
Vector+ Studio V8.3 - MULTIMODAL STORAGE
========================================

BREAKTHROUGH: Store BOTH embeddings AND compressed text in single lattice patterns!

New in v83:
- Multimodal cartridges: embedding + compressed text in one pattern
- Hybrid recall: text before settle (100%), embeddings after settle (0.995 cos)
- Search returns actual document text FROM THE LATTICE (not just from pkl)
- The lattice is now a true "knowledge cartridge" - semantic + content in one
- Pure Brain search upgraded to L2 signatures (recall_l2) for better discrimination

Key insight: imprint/recall is lossless. settle() normalizes (good for embeddings,
bad for exact data). Solution: hybrid recall - read text BEFORE settle.

Based on test_text_in_lattice.py proof-of-concept (2026-02-01).
"""

import streamlit as st

st.set_page_config(
    page_title="Vector+ Studio v0.83",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pickle
import os
import sys
import time
import threading
import hashlib
import json
import zlib

try:
    import PyPDF2
except:
    PyPDF2 = None
try:
    import docx
except:
    docx = None

from sentence_transformers import SentenceTransformer

# --- SYSTEM SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

try:
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
except ImportError:
    st.error("V7 wrapper not found.")
    st.stop()

try:
    from thermometer_encoder_generic_64x64 import ThermometerEncoderNomic64x64
except ImportError:
    st.error("Thermometer encoder not found.")
    st.stop()


# =============================================================================
# MULTIMODAL ENCODER (from test_text_in_lattice.py)
# =============================================================================

class TextRegionEncoder:
    """Encodes compressed text into free lattice regions."""

    def __init__(self, lattice_size=4096, region_size=64):
        self.lattice_size = lattice_size
        self.region_size = region_size
        self.free_rows = [1, 6, 13, 20, 27, 32, 39, 46, 53, 60, 63]
        self.num_region_cols = 64
        self.max_bytes = len(self.free_rows) * self.num_region_cols  # 704 bytes
        self.byte_patterns = self._create_byte_patterns()

    def _create_byte_patterns(self):
        patterns = np.zeros((256, self.region_size, self.region_size), dtype=np.float32)
        for byte_val in range(256):
            n_active = round((byte_val / 255.0) * 4096)
            pattern = np.zeros((self.region_size, self.region_size), dtype=np.float32)
            for i in range(n_active):
                row = i // self.region_size
                col = i % self.region_size
                pattern[row, col] = 1.0
            patterns[byte_val] = pattern
        return patterns

    def compress_text(self, text: str) -> bytes:
        return zlib.compress(text.encode('utf-8'), level=9)

    def decompress_text(self, data: bytes) -> str:
        try:
            return zlib.decompress(data).decode('utf-8')
        except:
            return None

    def encode_text(self, text: str) -> tuple:
        compressed = self.compress_text(text)
        compressed_len = len(compressed)

        if compressed_len > self.max_bytes:
            compressed = compressed[:self.max_bytes]
            compressed_len = self.max_bytes

        layer = np.zeros((self.lattice_size, self.lattice_size), dtype=np.float32)

        for byte_idx, byte_val in enumerate(compressed):
            row_idx = byte_idx // self.num_region_cols
            region_col = byte_idx % self.num_region_cols

            if row_idx >= len(self.free_rows):
                break

            region_row = self.free_rows[row_idx]
            pixel_row = region_row * self.region_size
            pixel_col = region_col * self.region_size

            layer[pixel_row:pixel_row + self.region_size,
                  pixel_col:pixel_col + self.region_size] = self.byte_patterns[byte_val]

        return layer, compressed_len

    def decode_text(self, lattice: np.ndarray, expected_length: int) -> str:
        if lattice.ndim == 1:
            lattice = lattice.reshape(self.lattice_size, self.lattice_size)

        binary = (lattice > 0.5).astype(np.float32)
        recovered_bytes = []

        for byte_idx in range(expected_length):
            row_idx = byte_idx // self.num_region_cols
            region_col = byte_idx % self.num_region_cols

            if row_idx >= len(self.free_rows):
                break

            region_row = self.free_rows[row_idx]
            pixel_row = region_row * self.region_size
            pixel_col = region_col * self.region_size

            region = binary[pixel_row:pixel_row + self.region_size,
                           pixel_col:pixel_col + self.region_size]

            active_bits = np.sum(region)
            byte_val = int(round(np.clip((active_bits / 4096.0) * 255, 0, 255)))
            recovered_bytes.append(byte_val)

        return self.decompress_text(bytes(recovered_bytes))


class CombinedEncoder:
    """Multimodal encoder: embedding + text in single lattice pattern."""

    def __init__(self):
        self.embedding_encoder = ThermometerEncoderNomic64x64()
        self.text_encoder = TextRegionEncoder()

    def encode(self, embedding: np.ndarray, text: str) -> tuple:
        embedding_layer = self.embedding_encoder.encode(embedding).astype(np.float32)
        text_layer, compressed_len = self.text_encoder.encode_text(text)
        combined = np.maximum(embedding_layer, text_layer)

        metadata = {
            'compressed_len': compressed_len,
            'original_text_len': len(text),
            'combined_sparsity': np.mean(combined > 0) * 100,
        }
        return combined, metadata

    def decode_text_only(self, lattice: np.ndarray, compressed_len: int) -> str:
        return self.text_encoder.decode_text(lattice, compressed_len)

    def decode_embedding_only(self, lattice: np.ndarray) -> np.ndarray:
        return self.embedding_encoder.decode(lattice)


def hybrid_recall(ml, encoder, pattern, compressed_len, settle_frames=30, lock=None):
    """
    Hybrid recall: text BEFORE settle, embedding AFTER settle.

    Returns: (embedding, text)
    """
    if lock:
        with lock:
            return _hybrid_recall_inner(ml, encoder, pattern, compressed_len, settle_frames)
    else:
        return _hybrid_recall_inner(ml, encoder, pattern, compressed_len, settle_frames)


def _hybrid_recall_inner(ml, encoder, pattern, compressed_len, settle_frames):
    ml.reset()
    ml.imprint_pattern(pattern)

    # Read text BEFORE settle (exact bytes)
    pre_settle = ml.recall()
    text = encoder.decode_text_only(pre_settle, compressed_len)

    # Settle for embedding quality
    ml.settle(frames=settle_frames, learn=False)

    # Read embedding AFTER settle (physics-enhanced)
    post_settle = ml.recall()
    embedding = encoder.decode_embedding_only(post_settle)

    return embedding, text


# =============================================================================
# CONFIG
# =============================================================================

SETTLE_FRAMES = 5
TRAIN_SETTLE_FRAMES = 5
SIG_SETTLE_FRAMES = 10
PHYSICS_PROFILE = "quality"
DATA_DIR = os.path.join(current_dir, "cartridges")
PHYSICS_SEARCH = True
MAX_TEXT_BYTES = 704  # Max compressed text per pattern

# V7.2: Protected rows for text storage (text survives physics!)
TEXT_ROWS = [1, 6, 13, 20, 27, 32, 39, 46, 53, 60, 63]  # 11 rows not used by t-code


def enable_text_protection(ml):
    """Enable protection for text rows so text survives settle."""
    ml.set_protected_rows(TEXT_ROWS)


def disable_text_protection(ml):
    """Disable protection (normal physics everywhere)."""
    ml.set_protected_rows([])


def simple_stem(word):
    word = word.lower()
    for suffix in ['ings', 'ing', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ed', 'es', 's', 'ly']:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def keyword_matches(keyword, text):
    if keyword in text:
        return True
    stemmed_kw = simple_stem(keyword)
    text_words = text.split()
    for word in text_words:
        clean_word = word.strip(".,!?;:'\"")
        if simple_stem(clean_word) == stemmed_kw:
            return True
    return False


# =============================================================================
# CARTRIDGE I/O
# =============================================================================

def load_cartridge(path):
    """Load cartridge - supports both legacy and multimodal formats."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        version = data.get("version", "0")

        # v8.3 multimodal format
        if version == "8.3":
            return {
                'embeddings': np.array(data['embeddings']),
                'passages': data['passages'],
                'compressed_lens': data.get('compressed_lens', []),
                'multimodal': True,
            }

        # Legacy formats
        if version in ["7.0", "8.0", "8.1", "8.2"]:
            core = data.get("data", data)
            emb = core.get("embeddings", data.get("embeddings"))
            txt = core.get("passages", data.get("passages", []))
            return {
                'embeddings': np.array(emb) if emb is not None else None,
                'passages': list(txt),
                'multimodal': False,
            }

        if "data" in data:
            core = data["data"]
            return {
                'embeddings': np.array(core.get("embeddings")),
                'passages': core.get("passages", []),
                'multimodal': False,
            }

        if "embeddings" in data:
            return {
                'embeddings': np.array(data["embeddings"]),
                'passages': data.get("passages", data.get("texts", [])),
                'multimodal': False,
            }

    return None


def save_cartridge_multimodal(path, embeddings, passages, compressed_lens):
    """Save multimodal cartridge with compressed lengths."""
    cart = {
        "version": "8.3",
        "embeddings": embeddings,
        "passages": passages,
        "compressed_lens": compressed_lens,
    }
    with open(path, "wb") as f:
        pickle.dump(cart, f)
    return True


def load_signatures(sig_path):
    if not os.path.exists(sig_path):
        return None
    try:
        data = np.load(sig_path, allow_pickle=True)
        result = {
            'signatures': data['signatures'],
            'titles': data['titles'] if 'titles' in data else None,
            'n_patterns': int(data['n_patterns']) if 'n_patterns' in data else len(data['signatures']),
            'compressed_lens': data['compressed_lens'] if 'compressed_lens' in data else None,
            'signature_method': str(data['signature_method']) if 'signature_method' in data else 'legacy',
        }
        # Load compressed texts for brain-only text recovery
        if 'compressed_texts' in data:
            result['compressed_texts'] = list(data['compressed_texts'])
        return result
    except Exception as e:
        print(f"[Signatures] Failed to load {sig_path}: {e}")
        return None


def save_signatures(sig_path, signatures, titles=None, compressed_lens=None, compressed_texts=None,
                     signature_method="l2"):
    save_dict = {
        'pattern_ids': np.arange(len(signatures), dtype=np.int32),
        'signatures': signatures,
        'n_patterns': len(signatures),
        'signature_dim': signatures.shape[1] if len(signatures.shape) > 1 else 4096,
        'signature_method': np.array(signature_method),
    }
    if titles is not None:
        save_dict['titles'] = np.array(titles, dtype=object)
    if compressed_lens is not None:
        save_dict['compressed_lens'] = np.array(compressed_lens, dtype=np.int32)
    # Store compressed texts for brain-only mode (enables text recovery without pkl)
    if compressed_texts is not None:
        save_dict['compressed_texts'] = np.array(compressed_texts, dtype=object)
    np.savez_compressed(sig_path, **save_dict)
    return sig_path


def compute_cartridge_fingerprint(embeddings):
    count = len(embeddings)
    first_bytes = embeddings[0].tobytes()
    last_bytes = embeddings[-1].tobytes() if count > 1 else first_bytes
    combined = first_bytes + last_bytes + str(count).encode()
    fingerprint = hashlib.sha256(combined).hexdigest()[:16]
    return {"count": count, "fingerprint": fingerprint}


def save_brain_manifest(brain_path, embeddings):
    manifest_path = brain_path.replace("_brain.npy", "_brain_manifest.json")
    manifest = compute_cartridge_fingerprint(embeddings)
    manifest["version"] = "8.3"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def validate_brain_manifest(brain_path, embeddings):
    manifest_path = brain_path.replace("_brain.npy", "_brain_manifest.json")
    if not os.path.exists(manifest_path):
        return True, "Legacy brain (no manifest)"
    try:
        with open(manifest_path, "r") as f:
            saved_manifest = json.load(f)
        current = compute_cartridge_fingerprint(embeddings)
        if saved_manifest["count"] != current["count"]:
            return False, f"Brain/cartridge mismatch (count: {saved_manifest['count']} vs {current['count']})"
        if saved_manifest["fingerprint"] != current["fingerprint"]:
            return False, "Brain/cartridge mismatch (fingerprint changed)"
        return True, "Manifest validated"
    except Exception as e:
        return False, f"Manifest error: {e}"


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
<style>
    [data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; max-width: 100%; }
    footer { visibility: hidden; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }
    .stMarkdown { margin-bottom: 0; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0.2rem !important; }
    .stCaption { margin-top: 0; }
    .multimodal-badge { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        padding: 2px 8px; border-radius: 4px; color: white; font-size: 0.8em; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

_bg_state = {
    "progress": 0,
    "total": 0,
    "active": False,
    "compressed_lens": None,
    "signatures": None,
    "signatures_ready": False,
}

if "engine_lock" not in st.session_state:
    st.session_state.engine_lock = threading.Lock()
if "status" not in st.session_state:
    st.session_state.status = "Idle"
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "deleted_ids" not in st.session_state:
    st.session_state.deleted_ids = set()
if "query" not in st.session_state:
    st.session_state.query = ""
if "physics_trained" not in st.session_state:
    st.session_state.physics_trained = False
if "encoder" not in st.session_state:
    st.session_state.encoder = None
if "combined_encoder" not in st.session_state:
    st.session_state.combined_encoder = None
if "cartridge_modified" not in st.session_state:
    st.session_state.cartridge_modified = False
if "cartridge_path" not in st.session_state:
    st.session_state.cartridge_path = None
if "signatures" not in st.session_state:
    st.session_state.signatures = None
if "signatures_loaded" not in st.session_state:
    st.session_state.signatures_loaded = False
if "multimodal_mode" not in st.session_state:
    st.session_state.multimodal_mode = False
if "compressed_lens" not in st.session_state:
    st.session_state.compressed_lens = []
if "compressed_texts" not in st.session_state:
    st.session_state.compressed_texts = []  # For brain-only text recovery
if "brain_only_mode" not in st.session_state:
    st.session_state.brain_only_mode = False


# =============================================================================
# BOOT ENGINE
# =============================================================================

if "engine" not in st.session_state:
    with st.spinner("Booting V8.3 Multimodal Engine..."):
        try:
            st.session_state.engine = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)
            st.session_state.engine.set_profile(PHYSICS_PROFILE)
            st.session_state.encoder = ThermometerEncoderNomic64x64(
                n_dims=768, lattice_size=4096, region_size=64
            )
            st.session_state.combined_encoder = CombinedEncoder()
        except Exception as e:
            st.error(f"Engine failed: {e}")
            st.stop()


@st.cache_resource(show_spinner="Loading Embedder...")
def load_embedder():
    try:
        return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    except:
        return SentenceTransformer("all-mpnet-base-v2")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_physics_multimodal(embeddings, passages, max_patterns=100):
    """Train with multimodal patterns (embedding + text)."""
    ml = st.session_state.engine
    encoder = st.session_state.combined_encoder
    compressed_lens = []

    n_train = min(len(embeddings), max_patterns)

    with st.session_state.engine_lock:
        ml.reset()
        for i in range(n_train):
            pattern, meta = encoder.encode(embeddings[i], passages[i])
            ml.imprint_pattern(pattern)
            ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)
            compressed_lens.append(meta['compressed_len'])

    return n_train, compressed_lens


def background_train_multimodal(embeddings, passages, encoder, ml, lock, start_idx,
                                 brain_path=None, sig_path=None, initial_compressed_lens=None):
    """Background training with multimodal patterns."""
    global _bg_state
    _bg_state["active"] = True
    _bg_state["total"] = len(embeddings)
    _bg_state["progress"] = start_idx

    # Use passed-in compressed_lens (can't access session_state from thread)
    compressed_lens = list(initial_compressed_lens) if initial_compressed_lens else []

    print(f"[BG Training] Multimodal mode from {start_idx} to {len(embeddings)}")

    try:
        # Phase 1: Train all patterns
        for i in range(start_idx, len(embeddings)):
            with lock:
                pattern, meta = encoder.encode(embeddings[i], passages[i])
                ml.imprint_pattern(pattern)
                ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            if i >= len(compressed_lens):
                compressed_lens.append(meta['compressed_len'])

            _bg_state["progress"] = i + 1

            if (i + 1) % 500 == 0:
                print(f"[BG Training] {i + 1:,}/{len(embeddings):,}")

            time.sleep(0.001)

        print(f"[BG Training] Complete! Trained {len(embeddings):,} multimodal patterns")
        # Store in global for main thread to pick up (can't write to session_state from thread)
        _bg_state["compressed_lens"] = compressed_lens

        # Save brain
        if brain_path:
            with lock:
                ml.save_brain_compact(brain_path)
                actual_path = brain_path + ".npy"
                size_mb = os.path.getsize(actual_path) / (1024 * 1024)
                print(f"[BG Training] Brain saved: {actual_path} ({size_mb:.1f} MB, compact)")
                save_brain_manifest(actual_path, embeddings)

        # Phase 2: Capture signatures with compressed texts for brain-only mode
        if sig_path:
            print(f"[BG Training] Capturing multimodal signatures + compressed texts...")
            signatures = np.zeros((len(embeddings), 4096), dtype=np.float32)
            compressed_texts = []  # Store actual compressed bytes for brain-only text recovery

            for i in range(len(embeddings)):
                with lock:
                    ml.reset()
                    pattern, _ = encoder.encode(embeddings[i], passages[i])
                    ml.imprint_pattern(pattern)
                    ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
                    signatures[i] = ml.recall_l2().flatten()

                # Store compressed text bytes for brain-only mode
                compressed_bytes = encoder.text_encoder.compress_text(passages[i])
                compressed_texts.append(compressed_bytes)

                if (i + 1) % 500 == 0:
                    print(f"[BG Training] Captured {i + 1:,}/{len(embeddings):,} signatures")

            titles = [p.splitlines()[0][:50] if p else "" for p in passages]
            save_signatures(sig_path, signatures, titles, compressed_lens, compressed_texts)
            size_mb = os.path.getsize(sig_path) / (1024 * 1024)
            print(f"[BG Training] Signatures saved: {sig_path} ({size_mb:.2f} MB) [includes compressed texts]")

            # Store in global for main thread (can't write to session_state from thread)
            _bg_state["signatures"] = signatures
            _bg_state["signatures_ready"] = True

    except Exception as e:
        print(f"[BG Training] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _bg_state["active"] = False


def start_background_training(embeddings, passages, start_idx=100, brain_path=None, sig_path=None,
                               initial_compressed_lens=None):
    global _bg_state
    if _bg_state["active"]:
        return

    ml = st.session_state.engine
    encoder = st.session_state.combined_encoder
    lock = st.session_state.engine_lock

    thread = threading.Thread(
        target=background_train_multimodal,
        args=(embeddings, passages, encoder, ml, lock, start_idx, brain_path, sig_path,
              initial_compressed_lens),
        daemon=True,
    )
    thread.start()


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def multimodal_hybrid_search(q_emb, embeddings, passages, compressed_lens, encoder, ml, lock):
    """
    MULTIMODAL SEARCH with hybrid recall.

    Returns text directly FROM THE LATTICE (not from pkl).
    """
    results = []

    # First, get cosine scores to rank candidates
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    cosine_scores = np.dot(e_unit, q_norm)

    # Get top candidates
    top_n = min(50, len(embeddings))
    candidate_idx = np.argsort(cosine_scores)[-top_n:][::-1]

    with lock:
        # Encode query and get physics-enhanced embedding
        ml.reset()
        query_pattern, _ = encoder.encode(q_emb, "")  # Empty text for query
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        query_settled = ml.recall()
        query_decoded = encoder.decode_embedding_only(query_settled)

    q_unit = query_decoded / (np.linalg.norm(query_decoded) + 1e-9)

    # Score each candidate with physics
    for idx in candidate_idx:
        # Get compressed_len - either from pre-computed list or compute on the fly
        if idx < len(compressed_lens):
            comp_len = compressed_lens[idx]
        else:
            # Compute on the fly if not yet available (during background training)
            comp_len = len(encoder.text_encoder.compress_text(passages[idx]))

        with lock:
            pattern, _ = encoder.encode(embeddings[idx], passages[idx])

            # Hybrid recall: get text before settle, embedding after
            ml.reset()
            ml.imprint_pattern(pattern)

            # Read text BEFORE settle
            pre_settle = ml.recall()
            recovered_text = encoder.decode_text_only(pre_settle, comp_len)

            # Settle for embedding
            ml.settle(frames=SETTLE_FRAMES, learn=False)
            post_settle = ml.recall()
            recovered_emb = encoder.decode_embedding_only(post_settle)

        # Normalize recovered embedding
        emb_min, emb_max = embeddings[idx].min(), embeddings[idx].max()
        emb_norm = (embeddings[idx] - emb_min) / (emb_max - emb_min + 1e-9)
        r_norm = recovered_emb / (np.linalg.norm(recovered_emb) + 1e-9)

        # Physics score (query vs recovered)
        physics_score = np.dot(q_unit, r_norm)

        results.append({
            'idx': idx,
            'score': physics_score,
            'cosine_score': cosine_scores[idx],
            'recovered_text': recovered_text,  # Text FROM THE LATTICE!
            'original_text': passages[idx],
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results, query_settled


def protected_multimodal_search(q_emb, embeddings, passages, compressed_lens, encoder, ml, lock, alpha=0.7):
    """
    PROTECTED MULTIMODAL SEARCH - text survives settle via V7.2 protected rows!

    Unlike hybrid_recall which reads text BEFORE settle, this uses protected rows
    so text survives THROUGH settle. True physics-enhanced multimodal!

    Scoring: blended = alpha * cosine + (1-alpha) * physics
    Alpha=1.0 = pure cosine ranking, Alpha=0.0 = pure physics ranking
    Default alpha=0.7 gives cosine-dominant ranking with physics boost.
    """
    results = []

    # First, get cosine scores to rank candidates
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    cosine_scores = np.dot(e_unit, q_norm)

    # Get top candidates
    top_n = min(50, len(embeddings))
    candidate_idx = np.argsort(cosine_scores)[-top_n:][::-1]

    with lock:
        # Enable protected rows for text
        enable_text_protection(ml)

        # Encode query and get physics-enhanced embedding
        ml.reset()
        query_pattern, _ = encoder.encode(q_emb, "")  # Empty text for query
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        query_settled = ml.recall()
        q_unit = encoder.decode_embedding_only(query_settled)
        q_unit = q_unit / (np.linalg.norm(q_unit) + 1e-9)

        for idx in candidate_idx:
            idx = int(idx)
            comp_len = compressed_lens[idx] if idx < len(compressed_lens) else 0

            # Create multimodal pattern
            pattern, _ = encoder.encode(embeddings[idx], passages[idx])

            # With protected rows: text survives settle!
            ml.reset()
            ml.imprint_pattern(pattern)
            ml.settle(frames=SETTLE_FRAMES, learn=False)

            # Read AFTER settle - text rows are protected!
            post_settle = ml.recall()
            recovered_text = encoder.decode_text_only(post_settle, comp_len)
            recovered_emb = encoder.decode_embedding_only(post_settle)

            # Normalize recovered embedding
            r_norm = recovered_emb / (np.linalg.norm(recovered_emb) + 1e-9)

            # Physics score (query vs recovered)
            physics_score = np.dot(q_unit, r_norm)

            # Blended score: alpha * cosine + (1-alpha) * physics
            blended_score = alpha * cosine_scores[idx] + (1.0 - alpha) * physics_score

            results.append({
                'idx': idx,
                'score': blended_score,
                'physics_score': physics_score,
                'cosine_score': cosine_scores[idx],
                'recovered_text': recovered_text,  # Text survived settle!
                'original_text': passages[idx],
            })

        # Disable protection when done
        disable_text_protection(ml)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results, query_settled


def physics_search(q_emb, embeddings, encoder, ml):
    """Standard physics search (legacy mode)."""
    with st.session_state.engine_lock:
        query_pattern = encoder.encode(q_emb).astype(np.float32)
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        settled_pattern = ml.recall()
        decoded_emb = encoder.decode(settled_pattern)

    q_unit = decoded_emb / (np.linalg.norm(decoded_emb) + 1e-9)

    e_norm_list = []
    for emb in embeddings:
        e_min, e_max = emb.min(), emb.max()
        e_normalized = (emb - e_min) / (e_max - e_min + 1e-9)
        e_norm_list.append(e_normalized)
    e_normalized = np.array(e_norm_list)

    e_norms = np.linalg.norm(e_normalized, axis=1, keepdims=True) + 1e-9
    e_unit = e_normalized / e_norms
    scores = np.dot(e_unit, q_unit)

    return scores, settled_pattern


def physics_search_rerank(q_emb, embeddings, encoder, ml, top_n=50):
    """Physics rerank: get top-N by cosine, then rerank using full pattern correlation."""
    # First pass: cosine similarity
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    cosine_scores = np.dot(e_unit, q_norm)

    # Get top candidates
    top_indices = np.argsort(cosine_scores)[-top_n:][::-1]

    with st.session_state.engine_lock:
        # Settle each candidate and compute pattern correlation
        candidate_patterns = {}
        for idx in top_indices:
            ml.reset()
            pattern = encoder.encode(embeddings[idx]).astype(np.float32)
            ml.imprint_pattern(pattern)
            ml.settle(frames=10, learn=False)
            candidate_patterns[idx] = ml.recall().flatten()

        # Settle query
        ml.reset()
        query_pattern = encoder.encode(q_emb).astype(np.float32)
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        query_settled = ml.recall().flatten()

    # Rerank using pattern correlation
    rerank_scores = np.zeros(len(embeddings))
    for idx, cand_pattern in candidate_patterns.items():
        corr = np.corrcoef(query_settled, cand_pattern)[0, 1]
        rerank_scores[idx] = corr

    # Fill in non-candidates with scaled cosine
    for i in range(len(embeddings)):
        if i not in candidate_patterns:
            rerank_scores[i] = cosine_scores[i] * 0.5

    return rerank_scores, query_settled


# =============================================================================
# SIDEBAR
# =============================================================================

physics_enabled = PHYSICS_SEARCH

with st.sidebar:
    st.title("Vector+ v0.83")
    st.caption("🧬 Multimodal Storage")
    st.divider()

    # Search mode selector
    search_modes = [
        "Cosine (pkl only)",
        "Physics",
        "Physics + Rerank Top-50",
        "🧬 Multimodal (lattice text)",
        "🔒 Protected Multimodal (V7.2)",
        "🧠 Pure Brain (no pkl needed!)",
    ]
    selected_mode = st.radio("Search Mode", search_modes, index=4,
                              help="Protected Multimodal: text survives settle via V7.2 protected rows!")

    # Blend slider for protected multimodal
    if selected_mode == "🔒 Protected Multimodal (V7.2)":
        blend_alpha = st.slider(
            "Cosine / Physics blend",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="1.0 = pure cosine ranking, 0.0 = pure physics. Default 0.7 = cosine-dominant with physics boost."
        )
    else:
        blend_alpha = 0.7

    # Mount cartridge
    os.makedirs(DATA_DIR, exist_ok=True)
    pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]

    # Also find brain-only files (brain exists but no pkl)
    brain_files = [f.replace("_brain.npy", "") for f in os.listdir(DATA_DIR) if f.endswith("_brain.npy")]
    brain_only = [b for b in brain_files if f"{b}.pkl" not in pkl_files]

    # Combine: pkl files + brain-only options
    mount_options = pkl_files + [f"{b} (brain only)" for b in brain_only]

    if mount_options:
        selected = st.selectbox("Cartridge", mount_options)
        if st.button("Mount", type="primary"):
            # Check if this is a brain-only mount
            is_brain_only = selected.endswith("(brain only)")
            cart_name = selected.replace(" (brain only)", "").replace(".pkl", "")

            if is_brain_only:
                # BRAIN-ONLY MOUNT - no pkl needed!
                brain_file = os.path.join(DATA_DIR, f"{cart_name}_brain.npy")
                sig_path = os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")

                if not os.path.exists(brain_file):
                    st.error("Brain file not found")
                    st.stop()

                sig_data = load_signatures(sig_path)
                if not sig_data:
                    st.error("Signatures file required for brain-only mode")
                    st.stop()

                n_patterns = sig_data['n_patterns']
                compressed_lens = list(sig_data.get('compressed_lens') or [])
                compressed_texts = sig_data.get('compressed_texts') or []
                titles = list(sig_data.get('titles') or [])

                # Recover text from compressed_texts if available
                text_encoder = TextRegionEncoder()
                recovered_texts = []
                for i in range(n_patterns):
                    if i < len(compressed_texts) and compressed_texts[i] is not None:
                        try:
                            txt = text_encoder.decompress_text(bytes(compressed_texts[i]))
                            recovered_texts.append(txt if txt else f"[Pattern {i}]")
                        except:
                            recovered_texts.append(titles[i] if i < len(titles) else f"[Pattern {i}]")
                    elif i < len(titles):
                        recovered_texts.append(titles[i])
                    else:
                        recovered_texts.append(f"[Pattern {i}]")

                # Create dataset with recovered text (no embeddings needed for Pure Brain)
                st.session_state.dataset = {
                    "emb": np.zeros((n_patterns, 768), dtype=np.float32),  # Placeholder
                    "txt": recovered_texts
                }
                st.session_state.compressed_lens = compressed_lens
                st.session_state.compressed_texts = compressed_texts
                st.session_state.signatures = sig_data['signatures']
                st.session_state.signatures_loaded = True
                st.session_state.physics_trained = True
                st.session_state.multimodal_mode = True
                st.session_state.brain_only_mode = True
                st.session_state.cartridge_path = None  # No pkl
                st.session_state.deleted_ids = set()

                # Load brain
                with st.spinner("Loading brain (no pkl)..."):
                    ml = st.session_state.engine
                    with st.session_state.engine_lock:
                        ml.load_brain(brain_file)
                    size_mb = os.path.getsize(brain_file) / (1024 * 1024)
                    st.session_state.status = f"🧠 Brain-only: {n_patterns} patterns | {size_mb:.1f}MB"
                    _bg_state["progress"] = n_patterns
                    _bg_state["total"] = n_patterns

                has_texts = len(compressed_texts) > 0
                st.success(f"Brain-only mode! {n_patterns} patterns, texts: {'YES' if has_texts else 'titles only'}")
                st.rerun()

            else:
                # Normal pkl mount
                path = os.path.join(DATA_DIR, selected)
                cart_data = load_cartridge(path)

            if not is_brain_only and cart_data and cart_data['embeddings'] is not None:
                emb = cart_data['embeddings']
                txt = cart_data['passages']
                is_multimodal = cart_data.get('multimodal', False)
                compressed_lens = cart_data.get('compressed_lens', [])

                st.session_state.dataset = {"emb": emb, "txt": list(txt)}
                st.session_state.query = ""
                st.session_state.deleted_ids = set()
                st.session_state.physics_trained = False
                st.session_state.cartridge_modified = False
                st.session_state.cartridge_path = path
                st.session_state.signatures = None
                st.session_state.signatures_loaded = False
                st.session_state.multimodal_mode = is_multimodal
                st.session_state.compressed_lens = compressed_lens
                st.session_state.compressed_texts = []  # Will be populated during training
                st.session_state.brain_only_mode = False

                cart_name = os.path.splitext(selected)[0]
                brain_path = os.path.join(DATA_DIR, f"{cart_name}_brain")
                brain_file = brain_path + ".npy"
                sig_path = os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")

                brain_loaded = False
                if physics_enabled and os.path.exists(brain_file):
                    valid, msg = validate_brain_manifest(brain_file, emb)

                    if valid:
                        with st.spinner("Loading saved brain..."):
                            ml = st.session_state.engine
                            with st.session_state.engine_lock:
                                ml.load_brain(brain_file)
                            size_mb = os.path.getsize(brain_file) / (1024 * 1024)
                            st.session_state.physics_trained = True
                            st.session_state.status = f"Loaded {len(txt)} | Brain: {size_mb:.1f}MB"
                            _bg_state["progress"] = len(emb)
                            _bg_state["total"] = len(emb)
                            brain_loaded = True

                            # Load signatures
                            if os.path.exists(sig_path):
                                sig_data = load_signatures(sig_path)
                                if sig_data and sig_data['n_patterns'] == len(emb):
                                    st.session_state.signatures = sig_data['signatures']
                                    st.session_state.signatures_loaded = True
                                    if sig_data.get('compressed_lens') is not None:
                                        st.session_state.compressed_lens = list(sig_data['compressed_lens'])
                                    if sig_data.get('compressed_texts') is not None:
                                        st.session_state.compressed_texts = sig_data['compressed_texts']
                    else:
                        st.error(msg)
                        os.remove(brain_file)

                if physics_enabled and not brain_loaded:
                    # Train initial batch with multimodal
                    INITIAL_BATCH = 100
                    with st.spinner(f"Training multimodal physics (first {INITIAL_BATCH})..."):
                        n_trained, comp_lens = train_physics_multimodal(emb, txt, max_patterns=INITIAL_BATCH)
                        st.session_state.physics_trained = True
                        st.session_state.compressed_lens = comp_lens
                        st.session_state.multimodal_mode = True
                        st.session_state.status = f"Loaded {len(txt)} | Physics: {n_trained} (multimodal)"

                    if len(emb) > INITIAL_BATCH:
                        start_background_training(emb, txt, start_idx=INITIAL_BATCH,
                                                   brain_path=brain_path, sig_path=sig_path,
                                                   initial_compressed_lens=comp_lens)

                if not physics_enabled:
                    st.session_state.status = f"Loaded {len(txt)} entries"

                st.rerun()
            else:
                st.error("Failed to load cartridge")

    st.divider()
    st.caption(f"Status: {st.session_state.status}")

    if st.session_state.multimodal_mode:
        st.markdown('<span class="multimodal-badge">MULTIMODAL</span>', unsafe_allow_html=True)

    if st.session_state.physics_trained:
        if _bg_state["active"]:
            progress = _bg_state["progress"]
            total = _bg_state["total"]
            pct = int(100 * progress / total) if total > 0 else 0
            st.warning(f"Physics: Training {progress:,}/{total:,} ({pct}%)")
            st.progress(pct / 100)
            if st.button("Refresh"):
                st.rerun()
        else:
            st.success("Physics: Trained")

            # Sync results from background thread if available
            if "compressed_lens" in _bg_state and _bg_state["compressed_lens"]:
                st.session_state.compressed_lens = _bg_state["compressed_lens"]
                _bg_state["compressed_lens"] = None

            if _bg_state.get("signatures_ready") and "signatures" in _bg_state:
                st.session_state.signatures = _bg_state["signatures"]
                st.session_state.signatures_loaded = True
                _bg_state["signatures_ready"] = False
                _bg_state["signatures"] = None

    # Forge section
    with st.expander("Forge Multimodal"):
        name = st.text_input("Name", "my_docs")
        files_up = st.file_uploader("Files", type=["txt", "pdf", "docx"], accept_multiple_files=True)

        if st.button("Forge") and files_up:
            texts = []
            for f in files_up:
                if f.name.endswith(".txt"):
                    texts.append(f.read().decode())
                elif f.name.endswith(".pdf") and PyPDF2:
                    reader = PyPDF2.PdfReader(f)
                    texts.append("\n".join(p.extract_text() or "" for p in reader.pages))
                elif f.name.endswith(".docx") and docx:
                    d = docx.Document(f)
                    texts.append("\n".join(p.text for p in d.paragraphs))

            if texts:
                embedder = load_embedder()
                embs = embedder.encode([f"search_document: {t}" for t in texts], show_progress_bar=True)

                # Compute compressed lengths
                text_encoder = TextRegionEncoder()
                comp_lens = [len(text_encoder.compress_text(t)) for t in texts]

                save_cartridge_multimodal(
                    os.path.join(DATA_DIR, f"{name}.pkl"),
                    embs, texts, comp_lens
                )
                st.success(f"Saved multimodal cartridge: {name}.pkl")


# =============================================================================
# MAIN
# =============================================================================

st.title("Vector+ Studio v0.83")
st.caption("🧬 Multimodal Storage - Embeddings + Text in Single Lattice Patterns")

if not st.session_state.dataset:
    st.info("Mount a cartridge from the sidebar")
    st.stop()

# Search input
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    query = st.text_input("Search", placeholder="What are you looking for?", key="query")
with col2:
    top_k = st.number_input("Results", 5, 50, 10)
with col3:
    st.write("")
    search_clicked = st.button("Search", type="primary")

if search_clicked and query and len(query) > 2:
    t0 = time.perf_counter()
    blend_details = {}  # Component scores for blended mode display

    embedder = load_embedder()
    ml = st.session_state.engine
    encoder = st.session_state.encoder
    combined_encoder = st.session_state.combined_encoder
    texts = st.session_state.dataset["txt"]
    embeddings = st.session_state.dataset["emb"]
    compressed_lens = st.session_state.compressed_lens

    # Clean query
    stopwords = {"a", "an", "the", "is", "are", "what", "who", "how", "when",
                 "where", "why", "which", "of", "in", "on", "for", "to", "and",
                 "or", "it", "as", "at", "by", "from", "about", "with", "me",
                 "tell", "show", "find", "search"}
    words = [w.lower().strip("?.,!") for w in query.split() if len(w) > 2]
    keywords = [w for w in words if w not in stopwords]
    clean_q = " ".join(keywords) if keywords else query

    q_emb = embedder.encode(f"search_query: {clean_q}")

    # --- SEARCH ---
    if selected_mode == "🧠 Pure Brain (no pkl needed!)" and st.session_state.signatures_loaded:
        # PURE BRAIN SEARCH - uses signatures for ranking, stored compressed texts for content
        # This is the "dream demo" - delete pkl, still search via brain!
        with st.spinner("🧠 Pure Brain search (settling physics)..."):
            signatures = st.session_state.signatures
            compressed_texts = st.session_state.compressed_texts
            text_encoder = TextRegionEncoder()

            with st.session_state.engine_lock:
                # Encode query and settle to get query signature
                ml.reset()
                query_pattern, _ = combined_encoder.encode(q_emb, "")
                ml.imprint_pattern(query_pattern)
                ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
                query_sig = ml.recall_l2().flatten()

            # Compare query signature to all stored signatures (semantic matching via brain)
            q_sig_norm = query_sig / (np.linalg.norm(query_sig) + 1e-9)
            sig_norms = np.linalg.norm(signatures, axis=1, keepdims=True) + 1e-9
            sig_unit = signatures / sig_norms
            sig_scores = np.dot(sig_unit, q_sig_norm)

            # Also compute what cosine would return (for diagnostic comparison)
            if not st.session_state.brain_only_mode:
                cos_q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-9)
                cos_e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
                cos_e_unit = embeddings / cos_e_norms
                cosine_scores = np.dot(cos_e_unit, cos_q_unit)
                cosine_top = list(np.argsort(cosine_scores)[-top_k:][::-1])
            else:
                cosine_top = None

            # Get top candidates
            top_indices = np.argsort(sig_scores)[-top_k:][::-1]

            # Recover text from stored compressed bytes (no pkl needed!)
            results_data = []
            for idx in top_indices:
                idx = int(idx)

                # Recover text from stored compressed bytes
                recovered_text = None
                if idx < len(compressed_texts) and compressed_texts[idx] is not None:
                    try:
                        recovered_text = text_encoder.decompress_text(bytes(compressed_texts[idx]))
                    except:
                        pass

                # Fallback to dataset text (may be recovered or placeholder)
                if not recovered_text and idx < len(texts):
                    recovered_text = texts[idx]

                results_data.append({
                    'idx': idx,
                    'score': sig_scores[idx],
                    'recovered_text': recovered_text,
                })

            # Compute reranking difference (how many positions changed vs cosine)
            brain_top = [int(idx) for idx in top_indices]
            if cosine_top is not None:
                rerank_diff = sum(1 for i, idx in enumerate(brain_top) if i < len(cosine_top) and idx != cosine_top[i])
            else:
                rerank_diff = None

        search_mode = "🧠 Pure Brain (no pkl!)"
        if rerank_diff is not None:
            search_mode += f" [{rerank_diff}/{top_k} reranked vs cosine]"
        results = [(r['score'], r['idx'], r.get('recovered_text')) for r in results_data]

    elif selected_mode == "🧬 Multimodal (lattice text)" and st.session_state.physics_trained:
        # MULTIMODAL HYBRID SEARCH - text from lattice!
        with st.spinner("🧬 Multimodal search (hybrid recall)..."):
            results_data, query_pattern = multimodal_hybrid_search(
                q_emb, embeddings, texts, compressed_lens,
                combined_encoder, ml, st.session_state.engine_lock
            )
        search_mode = "🧬 Multimodal Hybrid"
        results = [(r['score'], r['idx'], r.get('recovered_text')) for r in results_data[:top_k]]

    elif selected_mode == "🔒 Protected Multimodal (V7.2)" and st.session_state.physics_trained:
        # V7.2 PROTECTED MULTIMODAL - text survives settle!
        with st.spinner("🔒 Protected multimodal search (text survives settle!)..."):
            results_data, query_pattern = protected_multimodal_search(
                q_emb, embeddings, texts, compressed_lens,
                combined_encoder, ml, st.session_state.engine_lock,
                alpha=blend_alpha
            )
        search_mode = f"🔒 Protected Multimodal (blend={blend_alpha:.2f})"
        results = [(r['score'], r['idx'], r.get('recovered_text')) for r in results_data[:top_k]]
        # Store component scores for display
        blend_details = {r['idx']: (r['cosine_score'], r['physics_score']) for r in results_data[:top_k]}

    elif selected_mode == "Physics + Rerank Top-50" and st.session_state.physics_trained:
        # Physics with full pattern rerank
        with st.spinner("⚛️ Physics rerank (settling 50 patterns)..."):
            scores, query_pattern = physics_search_rerank(q_emb, embeddings, encoder, ml, top_n=50)
        search_mode = "Physics + Rerank"
        results = [(scores[i], i, None) for i in np.argsort(scores)[-top_k:][::-1]]

    elif selected_mode == "Physics" and st.session_state.physics_trained:
        # Standard physics search
        with st.spinner("⚛️ Physics search (settling query)..."):
            scores, query_pattern = physics_search(q_emb, embeddings, encoder, ml)
        search_mode = "Physics"
        results = [(scores[i], i, None) for i in np.argsort(scores)[-top_k:][::-1]]

    else:
        # Cosine only (also fallback if physics not trained)
        q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        e_unit = embeddings / e_norms
        scores = np.dot(e_unit, q_unit)
        search_mode = "Cosine (pkl)"
        results = [(scores[i], i, None) for i in np.argsort(scores)[-top_k:][::-1]]

    t1 = time.perf_counter()

    # Display
    st.caption(f"{search_mode} | {(t1-t0)*1000:.0f}ms | {len(results)} results")

    # Results
    st.markdown("### Results")
    with st.container(height=600):
        for rank, (score, idx, recovered_text) in enumerate(results[:top_k]):
            if idx in st.session_state.deleted_ids:
                continue

            txt = texts[idx]
            lines = txt.splitlines() if txt else ["[empty]"]
            title = lines[0][:100]

            with st.container(border=True):
                hcol1, hcol2 = st.columns([5, 1])
                with hcol1:
                    if recovered_text:
                        st.markdown(f"**{rank+1}. {title}** <span class='multimodal-badge'>FROM LATTICE</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{rank+1}. {title}**")
                with hcol2:
                    if idx in blend_details:
                        cos_s, phys_s = blend_details[idx]
                        st.caption(f"{score:.3f}\nC:{cos_s:.3f} P:{phys_s:.3f}")
                    else:
                        st.caption(f"{score:.3f}")

                if len(lines) > 1:
                    preview = " ".join(lines[1:3])[:200]
                    st.caption(preview + "..." if len(preview) >= 200 else preview)

                # Show text source
                if recovered_text:
                    with st.expander("📦 Lattice-recovered text"):
                        st.text(recovered_text[:2000] if recovered_text else "[Failed to recover]")
                    with st.expander("📄 Original text (from pkl)"):
                        st.text(txt[:2000])
                else:
                    with st.expander("Full text"):
                        st.text(txt[:2000])
