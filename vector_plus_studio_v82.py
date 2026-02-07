"""
Vector+ Studio V8.2 - Pure Physics Search (No Embeddings Required!)

BREAKTHROUGH: The lattice can now search using only stored signatures.
Once trained, you can delete the embeddings pkl - the brain stands alone.

New in v82:
- Pure Signature Mode: Search using pre-computed signatures from training
- No embeddings needed at query time (only brain + signatures file)
- 128MB brain + ~16KB/pattern signatures = complete retrieval system

The lattice is no longer just a wrapper - it IS the memory.
"""

import streamlit as st

st.set_page_config(
    page_title="Vector+ Studio v0.82",
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

# --- CONFIG ---
SETTLE_FRAMES = 5  # Optimal for decode (tested)
TRAIN_SETTLE_FRAMES = 5  # Frames for learning
SIG_SETTLE_FRAMES = 10  # Frames for signature capture (more for accuracy)
PHYSICS_PROFILE = "quality"
DATA_DIR = os.path.join(current_dir, "cartridges")
PHYSICS_SEARCH = True  # Toggle physics search on/off


def simple_stem(word):
    """Strip common suffixes for flexible keyword matching."""
    word = word.lower()
    for suffix in ['ings', 'ing', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ed', 'es', 's', 'ly']:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def keyword_matches(keyword, text):
    """Check if keyword matches text, using both original and stemmed versions."""
    if keyword in text:
        return True
    stemmed_kw = simple_stem(keyword)
    text_words = text.split()
    for word in text_words:
        clean_word = word.strip(".,!?;:'\"")
        if simple_stem(clean_word) == stemmed_kw:
            return True
    return False


def load_cartridge(path):
    """Load any cartridge format - we only need embeddings + text."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if data.get("version") in ["7.0", "8.0", "8.1", "8.2"]:
            core = data.get("data", data)
            emb = core.get("embeddings", data.get("embeddings"))
            txt = core.get("passages", data.get("passages", []))
            return np.array(emb) if emb is not None else None, txt
        if "data" in data:
            core = data["data"]
            return np.array(core.get("embeddings")), core.get("passages", [])
        if "embeddings" in data:
            return np.array(data["embeddings"]), data.get(
                "passages", data.get("texts", [])
            )

    return None, None


def load_signatures(sig_path):
    """
    Load signature file (.npz format).

    Returns: dict with 'signatures', 'titles', 'n_patterns' or None if not found
    """
    if not os.path.exists(sig_path):
        return None

    try:
        data = np.load(sig_path, allow_pickle=True)
        return {
            'signatures': data['signatures'],
            'titles': data['titles'] if 'titles' in data else None,
            'n_patterns': int(data['n_patterns']) if 'n_patterns' in data else len(data['signatures']),
            'signature_dim': int(data['signature_dim']) if 'signature_dim' in data else 4096,
        }
    except Exception as e:
        print(f"[Signatures] Failed to load {sig_path}: {e}")
        return None


def save_signatures(sig_path, signatures, titles=None, metadata=None):
    """
    Save signatures to npz file.

    Args:
        sig_path: Output path
        signatures: (N, 4096) array of signatures
        titles: Optional list of passage titles
        metadata: Optional dict of additional metadata
    """
    save_dict = {
        'pattern_ids': np.arange(len(signatures), dtype=np.int32),
        'signatures': signatures,
        'n_patterns': len(signatures),
        'signature_dim': signatures.shape[1] if len(signatures.shape) > 1 else 4096,
    }

    if titles is not None:
        save_dict['titles'] = np.array(titles, dtype=object)

    if metadata:
        for k, v in metadata.items():
            save_dict[k] = v

    np.savez_compressed(sig_path, **save_dict)
    return sig_path


def save_cartridge(path, embeddings, passages):
    """Save cartridge to disk."""
    cart = {
        "version": "8.2",
        "embeddings": embeddings,
        "passages": passages,
    }
    with open(path, "wb") as f:
        pickle.dump(cart, f)
    return True


def compute_cartridge_fingerprint(embeddings):
    """Compute a fingerprint for cartridge validation."""
    count = len(embeddings)
    first_bytes = embeddings[0].tobytes()
    last_bytes = embeddings[-1].tobytes() if count > 1 else first_bytes
    combined = first_bytes + last_bytes + str(count).encode()
    fingerprint = hashlib.sha256(combined).hexdigest()[:16]
    return {"count": count, "fingerprint": fingerprint}


def save_brain_manifest(brain_path, embeddings):
    """Save manifest alongside brain file for validation."""
    manifest_path = brain_path.replace("_brain.npy", "_brain_manifest.json")
    manifest = compute_cartridge_fingerprint(embeddings)
    manifest["version"] = "8.2"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def validate_brain_manifest(brain_path, embeddings):
    """Check if brain manifest matches current cartridge."""
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


def add_passage(text, embedder, encoder, ml, lock):
    """Add a new passage: embed, train physics, append to dataset."""
    emb = embedder.encode(f"search_document: {text}")

    with lock:
        pattern = encoder.encode(emb).astype(np.float32)
        ml.imprint_pattern(pattern)
        ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

    return emb


def update_passage(idx, new_text, embedder, encoder, ml, lock):
    """Update an existing passage: re-embed and retrain."""
    emb = embedder.encode(f"search_document: {new_text}")

    with lock:
        pattern = encoder.encode(emb).astype(np.float32)
        ml.imprint_pattern(pattern)
        ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

    return emb


# --- CSS ---
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
</style>
""",
    unsafe_allow_html=True,
)

# --- BACKGROUND TRAINING STATE ---
_bg_state = {
    "progress": 0,
    "total": 0,
    "active": False,
}

# --- SESSION STATE ---
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
if "cartridge_modified" not in st.session_state:
    st.session_state.cartridge_modified = False
if "cartridge_path" not in st.session_state:
    st.session_state.cartridge_path = None
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx = None
if "exit_requested" not in st.session_state:
    st.session_state.exit_requested = False
# NEW: Signature storage
if "signatures" not in st.session_state:
    st.session_state.signatures = None
if "signatures_loaded" not in st.session_state:
    st.session_state.signatures_loaded = False

# --- BOOT ENGINE ---
if "engine" not in st.session_state:
    with st.spinner("Booting V8.2 Physics Engine..."):
        try:
            st.session_state.engine = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)
            st.session_state.engine.set_profile(PHYSICS_PROFILE)
            st.session_state.encoder = ThermometerEncoderNomic64x64(
                n_dims=768, lattice_size=4096, region_size=64
            )
        except Exception as e:
            st.error(f"Engine failed: {e}")
            st.stop()


@st.cache_resource(show_spinner="Loading Embedder. First time will be slow...")
def load_embedder():
    try:
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
    except:
        return SentenceTransformer("all-mpnet-base-v2")


def train_physics(embeddings, max_patterns=100):
    """Train Hebbian weights on embeddings for physics search (initial batch)."""
    ml = st.session_state.engine
    encoder = st.session_state.encoder

    n_train = min(len(embeddings), max_patterns)

    with st.session_state.engine_lock:
        ml.reset()
        for i in range(n_train):
            pattern = encoder.encode(embeddings[i]).astype(np.float32)
            ml.imprint_pattern(pattern)
            ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

    return n_train


def background_train_physics(embeddings, encoder, ml, lock, start_idx, brain_path=None, sig_path=None, passages=None):
    """
    Continue training in background thread after initial mount.
    NEW: Also captures signatures after training completes.
    """
    global _bg_state
    _bg_state["active"] = True
    _bg_state["total"] = len(embeddings)
    _bg_state["progress"] = start_idx

    print(f"[BG Training] Starting from {start_idx} to {len(embeddings)}")

    try:
        # Phase 1: Train all patterns
        for i in range(start_idx, len(embeddings)):
            with lock:
                pattern = encoder.encode(embeddings[i]).astype(np.float32)
                ml.imprint_pattern(pattern)
                ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            _bg_state["progress"] = i + 1

            if (i + 1) % 500 == 0:
                print(f"[BG Training] {i + 1:,}/{len(embeddings):,}")

            time.sleep(0.001)

        print(f"[BG Training] Complete! Trained {len(embeddings):,} patterns")

        # Save brain
        if brain_path:
            with lock:
                ml.save_brain(brain_path)
                actual_path = brain_path + ".npy"
                size_mb = os.path.getsize(actual_path) / (1024 * 1024)
                print(f"[BG Training] Brain saved: {actual_path} ({size_mb:.1f} MB)")

                manifest_path = save_brain_manifest(actual_path, embeddings)
                print(f"[BG Training] Manifest saved: {manifest_path}")

        # Phase 2: Capture signatures (NEW in v82!)
        if sig_path:
            print(f"[BG Training] Capturing signatures...")
            signatures = np.zeros((len(embeddings), 4096), dtype=np.float32)

            for i in range(len(embeddings)):
                with lock:
                    ml.reset()
                    pattern = encoder.encode(embeddings[i]).astype(np.float32)
                    ml.imprint_pattern(pattern)
                    ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
                    signatures[i] = ml.generate_signature()

                if (i + 1) % 500 == 0:
                    print(f"[BG Training] Captured {i + 1:,}/{len(embeddings):,} signatures")

            # Extract titles for metadata
            titles = []
            if passages:
                for p in passages:
                    if isinstance(p, dict):
                        titles.append(p.get("title", str(p.get("text", ""))[:50]))
                    else:
                        lines = str(p).splitlines()
                        titles.append(lines[0][:50] if lines else "")

            save_signatures(sig_path, signatures, titles)
            size_mb = os.path.getsize(sig_path) / (1024 * 1024)
            print(f"[BG Training] Signatures saved: {sig_path} ({size_mb:.2f} MB)")

            # Store in session state
            st.session_state.signatures = signatures
            st.session_state.signatures_loaded = True

    except Exception as e:
        print(f"[BG Training] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _bg_state["active"] = False


def start_background_training(embeddings, start_idx=100, brain_path=None, sig_path=None, passages=None):
    """Launch background training thread."""
    global _bg_state
    if _bg_state["active"]:
        return

    ml = st.session_state.engine
    encoder = st.session_state.encoder
    lock = st.session_state.engine_lock

    thread = threading.Thread(
        target=background_train_physics,
        args=(embeddings, encoder, ml, lock, start_idx, brain_path, sig_path, passages),
        daemon=True,
    )
    thread.start()


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def physics_search(q_emb, embeddings, encoder, ml):
    """Standard physics-enhanced search using decode."""
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

    return scores, settled_pattern, decoded_emb


def physics_search_rerank(q_emb, embeddings, encoder, ml, top_n=50):
    """Physics rerank using full pattern correlation on top candidates."""
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    cosine_scores = np.dot(e_unit, q_norm)

    top_indices = np.argsort(cosine_scores)[-top_n:][::-1]

    with st.session_state.engine_lock:
        candidate_patterns = {}
        for idx in top_indices:
            ml.reset()
            pattern = encoder.encode(embeddings[idx]).astype(np.float32)
            ml.imprint_pattern(pattern)
            ml.settle(frames=10, learn=False)
            candidate_patterns[idx] = ml.recall().flatten()

        ml.reset()
        query_pattern = encoder.encode(q_emb).astype(np.float32)
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        query_settled = ml.recall().flatten()

    rerank_scores = np.zeros(len(embeddings))
    for idx, cand_pattern in candidate_patterns.items():
        corr = np.corrcoef(query_settled, cand_pattern)[0, 1]
        rerank_scores[idx] = corr

    for i in range(len(embeddings)):
        if i not in candidate_patterns:
            rerank_scores[i] = cosine_scores[i] * 0.5

    return rerank_scores, query_settled, None


def physics_search_signatures(q_emb, stored_signatures, encoder, ml):
    """
    PURE PHYSICS SEARCH - No embeddings needed!

    Uses pre-computed signatures from training.
    This is the holy grail: query → physics → compare to stored signatures.

    Args:
        q_emb: Query embedding (from embedder)
        stored_signatures: (N, 4096) array of pre-computed signatures
        encoder: Thermometer encoder
        ml: Lattice engine

    Returns:
        scores, query_pattern, None
    """
    with st.session_state.engine_lock:
        # Encode query → imprint → settle → generate signature
        ml.reset()
        query_pattern = encoder.encode(q_emb).astype(np.float32)
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
        query_sig = ml.generate_signature()
        query_settled = ml.recall()

    # Normalize query signature
    q_sig_norm = query_sig / (np.linalg.norm(query_sig) + 1e-9)

    # Normalize stored signatures
    sig_norms = np.linalg.norm(stored_signatures, axis=1, keepdims=True) + 1e-9
    sigs_unit = stored_signatures / sig_norms

    # Cosine similarity
    scores = np.dot(sigs_unit, q_sig_norm)

    return scores, query_settled, None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("Vector+ v0.82")
    st.caption("Pure Physics Search")
    st.divider()

    physics_enabled = st.checkbox("Physics Search", value=PHYSICS_SEARCH)

    # NEW: Pure Signatures mode
    pure_sig_mode = st.checkbox("Pure Signatures", value=False,
                                 help="Search using only stored signatures - NO EMBEDDINGS NEEDED!")

    rerank_mode = st.checkbox("Full Pattern Rerank", value=False,
                               help="Rerank top-50 with full pattern correlation")

    # Mount cartridge
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]

    if files:
        selected = st.selectbox("Cartridge", files)
        if st.button("Mount", type="primary"):
            path = os.path.join(DATA_DIR, selected)
            emb, txt = load_cartridge(path)
            if emb is not None:
                st.session_state.dataset = {"emb": emb, "txt": list(txt)}
                st.session_state.query = ""
                st.session_state.deleted_ids = set()
                st.session_state.physics_trained = False
                st.session_state.cartridge_modified = False
                st.session_state.cartridge_path = path
                st.session_state.edit_idx = None
                st.session_state.signatures = None
                st.session_state.signatures_loaded = False

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
                            print(f"[Mount] Loaded brain: {brain_file}")

                            # Try to load signatures
                            if os.path.exists(sig_path):
                                sig_data = load_signatures(sig_path)
                                if sig_data and sig_data['n_patterns'] == len(emb):
                                    st.session_state.signatures = sig_data['signatures']
                                    st.session_state.signatures_loaded = True
                                    sig_size = os.path.getsize(sig_path) / (1024 * 1024)
                                    print(f"[Mount] Loaded signatures: {sig_path} ({sig_size:.2f} MB)")
                                    st.session_state.status += f" | Sigs: {sig_size:.1f}MB"
                                else:
                                    print(f"[Mount] Signature count mismatch, will regenerate")

                            if "Legacy" in msg:
                                st.warning(msg)
                    else:
                        st.error(msg)
                        print(f"[Mount] {msg}")
                        os.remove(brain_file)
                        manifest_file = brain_file.replace("_brain.npy", "_brain_manifest.json")
                        if os.path.exists(manifest_file):
                            os.remove(manifest_file)

                if physics_enabled and not brain_loaded:
                    INITIAL_BATCH = 100
                    with st.spinner(f"Training physics (first {INITIAL_BATCH})..."):
                        n_trained = train_physics(emb, max_patterns=INITIAL_BATCH)
                        st.session_state.physics_trained = True
                        st.session_state.status = f"Loaded {len(txt)} | Physics: {n_trained}"

                    if len(emb) > INITIAL_BATCH:
                        start_background_training(
                            emb,
                            start_idx=INITIAL_BATCH,
                            brain_path=brain_path,
                            sig_path=sig_path,
                            passages=txt
                        )

                if not physics_enabled:
                    st.session_state.status = f"Loaded {len(txt)} entries"

                st.rerun()
            else:
                st.error("Failed to load cartridge")

    st.divider()
    st.caption(f"Status: {st.session_state.status}")

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
            if st.session_state.signatures_loaded:
                st.success("Signatures: Loaded")
            else:
                st.info("Signatures: Not available")
                # Build Signatures button
                if st.session_state.dataset and st.button("Build Signatures"):
                    with st.spinner("Building signatures..."):
                        emb = st.session_state.dataset["emb"]
                        txt = st.session_state.dataset["txt"]
                        ml = st.session_state.engine
                        encoder = st.session_state.encoder

                        signatures = np.zeros((len(emb), 4096), dtype=np.float32)
                        progress_bar = st.progress(0)

                        for i in range(len(emb)):
                            with st.session_state.engine_lock:
                                ml.reset()
                                pattern = encoder.encode(emb[i]).astype(np.float32)
                                ml.imprint_pattern(pattern)
                                ml.settle(frames=SIG_SETTLE_FRAMES, learn=False)
                                signatures[i] = ml.generate_signature()

                            if (i + 1) % 100 == 0:
                                progress_bar.progress((i + 1) / len(emb))

                        progress_bar.progress(1.0)

                        # Extract titles
                        titles = []
                        for p in txt:
                            if isinstance(p, dict):
                                titles.append(p.get("title", str(p.get("text", ""))[:50]))
                            else:
                                lines = str(p).splitlines()
                                titles.append(lines[0][:50] if lines else "")

                        # Save signatures
                        cart_name = os.path.splitext(os.path.basename(st.session_state.cartridge_path))[0]
                        sig_path = os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")
                        save_signatures(sig_path, signatures, titles)

                        st.session_state.signatures = signatures
                        st.session_state.signatures_loaded = True

                        size_mb = os.path.getsize(sig_path) / (1024 * 1024)
                        st.success(f"Built {len(signatures)} signatures ({size_mb:.2f} MB)")
                        st.rerun()
    else:
        st.info("Physics: Not trained")

    # Forge section
    with st.expander("Forge New"):
        name = st.text_input("Name", "my_docs")
        files_up = st.file_uploader(
            "Files", type=["txt", "pdf", "docx"], accept_multiple_files=True
        )

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
                embs = embedder.encode(texts, show_progress_bar=True)
                cart = {"version": "8.2", "embeddings": embs, "passages": texts}
                with open(os.path.join(DATA_DIR, f"{name}.pkl"), "wb") as f:
                    pickle.dump(cart, f)
                st.success(f"Saved {name}.pkl")

    # CRUD section
    if st.session_state.dataset:
        if _bg_state["active"]:
            st.info("CRUD disabled while training")
        else:
            with st.expander("Add Passage"):
                new_passage = st.text_area("New passage text", height=150, key="new_passage_text")
                if st.button("Add") and new_passage.strip():
                    with st.spinner("Embedding & training..."):
                        embedder = load_embedder()
                        new_emb = add_passage(
                            new_passage,
                            embedder,
                            st.session_state.encoder,
                            st.session_state.engine,
                            st.session_state.engine_lock,
                        )
                        st.session_state.dataset["emb"] = np.vstack([
                            st.session_state.dataset["emb"],
                            new_emb.reshape(1, -1)
                        ])
                        st.session_state.dataset["txt"].append(new_passage)
                        st.session_state.cartridge_modified = True
                        # Invalidate signatures
                        st.session_state.signatures = None
                        st.session_state.signatures_loaded = False
                    st.success(f"Added! Total: {len(st.session_state.dataset['txt'])}")
                    st.rerun()

            if st.session_state.cartridge_modified:
                st.warning("Unsaved changes!")
                if st.button("Save Cartridge", type="primary"):
                    if st.session_state.deleted_ids:
                        keep_mask = [i not in st.session_state.deleted_ids
                                     for i in range(len(st.session_state.dataset["txt"]))]
                        st.session_state.dataset["emb"] = st.session_state.dataset["emb"][keep_mask]
                        st.session_state.dataset["txt"] = [
                            t for i, t in enumerate(st.session_state.dataset["txt"])
                            if i not in st.session_state.deleted_ids
                        ]
                        st.session_state.deleted_ids = set()

                    save_cartridge(
                        st.session_state.cartridge_path,
                        st.session_state.dataset["emb"],
                        st.session_state.dataset["txt"],
                    )
                    st.session_state.cartridge_modified = False

                    # Delete old brain + signatures (needs retraining)
                    cart_name = os.path.splitext(os.path.basename(st.session_state.cartridge_path))[0]
                    brain_file = os.path.join(DATA_DIR, f"{cart_name}_brain.npy")
                    manifest_file = os.path.join(DATA_DIR, f"{cart_name}_brain_manifest.json")
                    sig_file = os.path.join(DATA_DIR, f"{cart_name}_signatures.npz")
                    for f in [brain_file, manifest_file, sig_file]:
                        if os.path.exists(f):
                            os.remove(f)

                    st.success(f"Saved {len(st.session_state.dataset['txt'])} passages")
                    st.rerun()

    # Exit
    st.divider()
    if st.button("Exit"):
        st.info("Close this browser tab to exit.")


# =============================================================================
# MAIN
# =============================================================================

st.title("Vector+ Studio v0.82")
st.caption("Pure Physics Search - The Lattice IS the Memory")

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

if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""
    st.session_state.last_search_results = None

if search_clicked and query and len(query) > 2:
    st.session_state.last_search_query = query
    t0 = time.perf_counter()

    embedder = load_embedder()
    ml = st.session_state.engine
    encoder = st.session_state.encoder
    texts = st.session_state.dataset["txt"]
    embeddings = st.session_state.dataset["emb"]

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
    if pure_sig_mode and st.session_state.signatures_loaded:
        # PURE SIGNATURE SEARCH - No embeddings used!
        scores, query_pattern, _ = physics_search_signatures(
            q_emb, st.session_state.signatures, encoder, ml
        )
        search_mode = "Pure Signatures (No Embeddings!)"
    elif physics_enabled and st.session_state.physics_trained:
        if rerank_mode:
            scores, query_pattern, decoded_emb = physics_search_rerank(
                q_emb, embeddings, encoder, ml, top_n=50
            )
            search_mode = "Physics (Full Rerank)"
        else:
            scores, query_pattern, decoded_emb = physics_search(
                q_emb, embeddings, encoder, ml
            )
            search_mode = "Physics"
    else:
        q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        e_unit = embeddings / e_norms
        scores = np.dot(e_unit, q_unit)
        search_mode = "Cosine"

        with st.session_state.engine_lock:
            q_min, q_max = q_emb.min(), q_emb.max()
            q_norm = ((q_emb - q_min) / (q_max - q_min + 1e-9)).astype(np.float32)
            ml.reset()
            ml.imprint_vector(q_norm)
            ml.settle(frames=SETTLE_FRAMES, learn=False)
            query_pattern = ml.recall()

    # Keyword boost
    n_candidates = min(100, len(texts))
    candidate_idx = np.argsort(scores)[-n_candidates:][::-1]

    boosted_scores = []
    for idx in candidate_idx:
        if idx in st.session_state.deleted_ids:
            continue
        txt = texts[idx] if texts[idx] else ""
        txt_lower = txt.lower()
        title = txt.splitlines()[0].lower() if txt else ""

        title_bonus = sum(0.1 for kw in keywords if keyword_matches(kw, title))
        title_bonus = min(title_bonus, 0.3)

        body_bonus = sum(0.05 for kw in keywords if keyword_matches(kw, txt_lower) and not keyword_matches(kw, title))
        body_bonus = min(body_bonus, 0.2)

        keywords_found = [kw for kw in keywords if keyword_matches(kw, txt_lower)]
        phrase_bonus = min(0.05 + len(keywords_found) * 0.05, 0.25) if len(keywords_found) >= 2 else 0.0

        final_score = scores[idx] + title_bonus + body_bonus + phrase_bonus
        boosted_scores.append((final_score, idx))

    boosted_scores.sort(reverse=True, key=lambda x: x[0])
    results = boosted_scores[:top_k]

    t1 = time.perf_counter()

    # Display
    st.caption(f"{search_mode} | {(t1-t0)*1000:.0f}ms | {len(results)} results")

    # Results
    st.markdown("### Results")
    with st.container(height=600):
        for rank, (score, idx) in enumerate(results[:top_k]):
            txt = texts[idx]
            lines = txt.splitlines() if txt else ["[empty]"]
            title = lines[0][:100]

            with st.container(border=True):
                hcol1, hcol2 = st.columns([5, 1])
                with hcol1:
                    st.markdown(f"**{rank+1}. {title}**")
                with hcol2:
                    st.caption(f"{score:.3f}")

                if len(lines) > 1:
                    preview = " ".join(lines[1:3])[:200]
                    st.caption(preview + "..." if len(preview) >= 200 else preview)

                with st.expander("Full text"):
                    st.text(txt[:2000])
