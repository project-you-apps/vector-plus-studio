"""
Vector+ Studio V8.1 - Real Physics Search

The lattice now PARTICIPATES in search:
1. Query goes through physics (encode → settle → decode)
2. Physics-cleaned embedding used for search
3. 98.9% decode accuracy preserves semantic information

This is the real deal - Hopfield associative memory for semantic search.
"""

import streamlit as st

st.set_page_config(
    page_title="Vector+ Studio v0.81",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pickle
import os
import sys
import time
import threading

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
PHYSICS_PROFILE = "quality"
DATA_DIR = os.path.join(current_dir, "cartridges")
PHYSICS_SEARCH = True  # Toggle physics search on/off


def load_cartridge(path):
    """Load any cartridge format - we only need embeddings + text."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        # V7/V8 format
        if data.get("version") in ["7.0", "8.0", "8.1"]:
            core = data.get("data", data)
            emb = core.get("embeddings", data.get("embeddings"))
            txt = core.get("passages", data.get("passages", []))
            return np.array(emb) if emb is not None else None, txt
        # V2 format
        if "data" in data:
            core = data["data"]
            return np.array(core.get("embeddings")), core.get("passages", [])
        # Simple dict
        if "embeddings" in data:
            return np.array(data["embeddings"]), data.get(
                "passages", data.get("texts", [])
            )

    return None, None


# --- CSS ---
st.markdown(
    """
<style>
    /* Hide Streamlit header/nav bar */
    [data-testid="stHeader"] { display: none; }

    /* Tighten whitespace */
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; max-width: 100%; }
    footer { visibility: hidden; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }

    /* Smaller gaps */
    .stMarkdown { margin-bottom: 0; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0.2rem !important; }
    .stCaption { margin-top: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# --- BACKGROUND TRAINING STATE (module-level, thread-safe) ---
# Use a regular dict so background thread doesn't touch st.session_state
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

# --- BOOT ENGINE ---
if "engine" not in st.session_state:
    with st.spinner("Booting V8.1 Physics Engine..."):
        try:
            st.session_state.engine = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)
            st.session_state.engine.set_profile(PHYSICS_PROFILE)
            # Create thermometer encoder for decode
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
            # Encode and imprint with learning
            pattern = encoder.encode(embeddings[i]).astype(np.float32)
            ml.imprint_pattern(pattern)
            ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

    return n_train


def background_train_physics(embeddings, encoder, ml, lock, start_idx, brain_path=None):
    """Continue training in background thread after initial mount."""
    global _bg_state
    _bg_state["active"] = True
    _bg_state["total"] = len(embeddings)
    _bg_state["progress"] = start_idx

    print(f"[BG Training] Starting from {start_idx} to {len(embeddings)}")

    try:
        for i in range(start_idx, len(embeddings)):
            # Acquire lock briefly to train one pattern
            with lock:
                pattern = encoder.encode(embeddings[i]).astype(np.float32)
                ml.imprint_pattern(pattern)
                ml.settle(frames=TRAIN_SETTLE_FRAMES, learn=True)

            _bg_state["progress"] = i + 1

            # Log progress every 500 patterns
            if (i + 1) % 500 == 0:
                print(f"[BG Training] {i + 1:,}/{len(embeddings):,}")

            # Yield to allow search operations
            time.sleep(0.001)

        print(f"[BG Training] Complete! Trained {len(embeddings):,} patterns")

        # Save brain to disk
        if brain_path:
            with lock:
                ml.save_brain(brain_path)
                # np.save adds .npy extension
                actual_path = brain_path + ".npy"
                size_mb = os.path.getsize(actual_path) / (1024 * 1024)
                print(f"[BG Training] Brain saved: {actual_path} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"[BG Training] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _bg_state["active"] = False


def start_background_training(embeddings, start_idx=100, brain_path=None):
    """Launch background training thread."""
    global _bg_state
    if _bg_state["active"]:
        return  # Already training

    ml = st.session_state.engine
    encoder = st.session_state.encoder
    lock = st.session_state.engine_lock

    thread = threading.Thread(
        target=background_train_physics,
        args=(embeddings, encoder, ml, lock, start_idx, brain_path),
        daemon=True,
    )
    thread.start()


def physics_search(q_emb, embeddings, encoder, ml):
    """
    Physics-enhanced search using decode.

    1. Encode query to lattice pattern
    2. Imprint and settle (Hebbian weights stabilize)
    3. Decode back to embedding space
    4. Cosine search with decoded embedding
    """
    with st.session_state.engine_lock:
        # Encode query with Python encoder (matches training)
        query_pattern = encoder.encode(q_emb).astype(np.float32)

        # Imprint and settle (no learning - just recall)
        ml.imprint_pattern(query_pattern)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        settled_pattern = ml.recall()

        # Decode settled pattern back to embedding
        decoded_emb = encoder.decode(settled_pattern)

    # Cosine search with physics-cleaned embedding
    q_unit = decoded_emb / (np.linalg.norm(decoded_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms

    # Normalize stored embeddings to [0,1] for fair comparison
    e_norm_list = []
    for emb in embeddings:
        e_min, e_max = emb.min(), emb.max()
        e_normalized = (emb - e_min) / (e_max - e_min + 1e-9)
        e_norm_list.append(e_normalized)
    e_normalized = np.array(e_norm_list)

    # Cosine with normalized embeddings
    e_norms = np.linalg.norm(e_normalized, axis=1, keepdims=True) + 1e-9
    e_unit = e_normalized / e_norms
    scores = np.dot(e_unit, q_unit)

    return scores, settled_pattern, decoded_emb


# --- SIDEBAR ---
with st.sidebar:
    st.title("Vector+ v0.81")
    st.caption("Physics-Enhanced Search")
    st.divider()

    # Physics toggle
    physics_enabled = st.checkbox("Physics Search", value=PHYSICS_SEARCH)

    # Mount cartridge
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]

    if files:
        selected = st.selectbox("Cartridge", files)
        if st.button("Mount", type="primary"):
            path = os.path.join(DATA_DIR, selected)
            emb, txt = load_cartridge(path)
            if emb is not None:
                st.session_state.dataset = {"emb": emb, "txt": txt}
                st.session_state.query = ""
                st.session_state.deleted_ids = set()
                st.session_state.physics_trained = False

                # Brain file path (same name as cartridge, but _brain.npy)
                cart_name = os.path.splitext(selected)[0]
                brain_path = os.path.join(DATA_DIR, f"{cart_name}_brain")
                brain_file = brain_path + ".npy"

                # Check for saved brain
                if physics_enabled and os.path.exists(brain_file):
                    with st.spinner("Loading saved brain..."):
                        ml = st.session_state.engine
                        with st.session_state.engine_lock:
                            ml.load_brain(brain_file)
                        size_mb = os.path.getsize(brain_file) / (1024 * 1024)
                        st.session_state.physics_trained = True
                        st.session_state.status = f"Loaded {len(txt)} | Brain: {size_mb:.1f}MB"
                        _bg_state["progress"] = len(emb)
                        _bg_state["total"] = len(emb)
                        print(f"[Mount] Loaded brain: {brain_file} ({size_mb:.1f} MB)")

                # Train physics on mount (initial batch, then background)
                elif physics_enabled:
                    INITIAL_BATCH = 100
                    with st.spinner(f"Training physics (first {INITIAL_BATCH})..."):
                        n_trained = train_physics(emb, max_patterns=INITIAL_BATCH)
                        st.session_state.physics_trained = True
                        st.session_state.status = (
                            f"Loaded {len(txt)} | Physics: {n_trained}"
                        )

                    # Start background training for the rest (will save when done)
                    if len(emb) > INITIAL_BATCH:
                        start_background_training(emb, start_idx=INITIAL_BATCH, brain_path=brain_path)
                else:
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
            trained = _bg_state["progress"] or 100
            st.success(f"Physics: Trained ({trained:,})")
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
                    texts.append(
                        "\n".join(p.extract_text() or "" for p in reader.pages)
                    )
                elif f.name.endswith(".docx") and docx:
                    d = docx.Document(f)
                    texts.append("\n".join(p.text for p in d.paragraphs))

            if texts:
                embedder = load_embedder()
                embs = embedder.encode(texts, show_progress_bar=True)
                cart = {"version": "8.1", "embeddings": embs, "passages": texts}
                with open(os.path.join(DATA_DIR, f"{name}.pkl"), "wb") as f:
                    pickle.dump(cart, f)
                st.success(f"Saved {name}.pkl")


# --- MAIN ---
st.title("Vector+ Studio v0.81")
st.caption("Physics-Enhanced Semantic Search")

if not st.session_state.dataset:
    st.info("Mount a cartridge from the sidebar")
    st.stop()

# Search input
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Search", placeholder="What are you looking for?", key="query"
    )
with col2:
    top_k = st.number_input("Results", 5, 50, 10)

if query and len(query) > 2:
    t0 = time.perf_counter()

    embedder = load_embedder()
    ml = st.session_state.engine
    encoder = st.session_state.encoder
    texts = st.session_state.dataset["txt"]
    embeddings = st.session_state.dataset["emb"]

    # --- QUERY PROCESSING ---
    # Clean query
    stopwords = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "what",
        "who",
        "how",
        "when",
        "where",
        "why",
        "which",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "it",
        "as",
        "at",
        "by",
        "from",
        "about",
        "with",
        "me",
        "tell",
        "show",
        "find",
        "search",
    }
    words = [w.lower().strip("?.,!") for w in query.split() if len(w) > 2]
    keywords = [w for w in words if w not in stopwords]
    clean_q = " ".join(keywords) if keywords else query

    # Embed query
    q_emb = embedder.encode(f"search_query: {clean_q}")

    # --- SEARCH ---
    if physics_enabled and st.session_state.physics_trained:
        # Physics-enhanced search
        scores, query_pattern, decoded_emb = physics_search(
            q_emb, embeddings, encoder, ml
        )
        search_mode = "Physics"
    else:
        # Fallback: standard cosine search
        q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        e_unit = embeddings / e_norms
        scores = np.dot(e_unit, q_unit)
        search_mode = "Cosine"

        # Generate query pattern for visualization
        with st.session_state.engine_lock:
            q_min, q_max = q_emb.min(), q_emb.max()
            q_norm = ((q_emb - q_min) / (q_max - q_min + 1e-9)).astype(np.float32)
            ml.reset()
            ml.imprint_vector(q_norm)
            ml.settle(frames=SETTLE_FRAMES, learn=False)
            query_pattern = ml.recall()

    # Get top candidates for keyword re-ranking (get more than we need)
    n_candidates = min(100, len(texts))
    candidate_idx = np.argsort(scores)[-n_candidates:][::-1]

    # Keyword boost on candidates only (title + full text + phrase matching)
    boosted_scores = []
    for idx in candidate_idx:
        if idx in st.session_state.deleted_ids:
            continue
        txt = texts[idx] if texts[idx] else ""
        txt_lower = txt.lower()
        title = txt.splitlines()[0].lower() if txt else ""

        # Title match: 0.1 per keyword (max 0.3)
        title_bonus = sum(0.1 for kw in keywords if kw in title)
        title_bonus = min(title_bonus, 0.3)

        # Full text match: 0.05 per keyword (max 0.2)
        # Only count if NOT already in title (avoid double-counting)
        body_bonus = sum(0.05 for kw in keywords if kw in txt_lower and kw not in title)
        body_bonus = min(body_bonus, 0.2)

        # Phrase match bonus: if MULTIPLE keywords found in text, extra boost
        # This helps "Reed Richards" beat "Keith Richards" for Fantastic Four
        keywords_found = [kw for kw in keywords if kw in txt_lower]
        if len(keywords_found) >= 2:
            # Bonus scales with how many keywords match (2=0.15, 3=0.20, 4+=0.25)
            phrase_bonus = min(0.05 + len(keywords_found) * 0.05, 0.25)
        else:
            phrase_bonus = 0.0

        final_score = scores[idx] + title_bonus + body_bonus + phrase_bonus
        boosted_scores.append((final_score, idx))

    # Sort by boosted score and take top-k
    boosted_scores.sort(reverse=True, key=lambda x: x[0])
    results = boosted_scores[:top_k]

    # Get top result pattern for comparison
    top_result_pattern = None
    if results:
        top_idx_result = results[0][1]
        top_emb = embeddings[top_idx_result]
        with st.session_state.engine_lock:
            top_pattern = encoder.encode(top_emb).astype(np.float32)
            ml.imprint_pattern(top_pattern)
            ml.settle(frames=SETTLE_FRAMES, learn=False)
            top_result_pattern = ml.recall()

    t1 = time.perf_counter()

    # --- DISPLAY ---
    st.caption(f"{search_mode} | {(t1-t0)*1000:.0f}ms | {len(results)} results")

    # Two columns: results + visualization
    col_results, col_viz = st.columns([3, 1])

    with col_viz:

        def make_lattice_image(pattern, color="cyan"):
            """Create 256x256 visualization from lattice pattern."""
            l4 = pattern.reshape(4096, 4096)
            l3 = l4.reshape(256, 16, 256, 16).mean(axis=(1, 3))
            l3_norm = np.clip(l3 * 25.0, 0, 1)
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            if color == "cyan":
                img[:, :, 0] = (20 + l3_norm * 80).astype(np.uint8)
                img[:, :, 1] = (40 + l3_norm * 180).astype(np.uint8)
                img[:, :, 2] = (60 + l3_norm * 195).astype(np.uint8)
            else:  # green for result
                img[:, :, 0] = (20 + l3_norm * 60).astype(np.uint8)
                img[:, :, 1] = (40 + l3_norm * 195).astype(np.uint8)
                img[:, :, 2] = (30 + l3_norm * 100).astype(np.uint8)
            return img

        def get_status(pattern):
            """Get sparsity status."""
            sparsity = (pattern > 0.01).sum() / 16_777_216 * 100
            if sparsity > 20:
                return "Bloom!", "warning"
            elif sparsity < 0.5:
                return "Sparse", "info"
            else:
                return "Stable", "success"

        # Query pattern (cyan)
        st.markdown("**Query**")
        q_img = make_lattice_image(query_pattern, "cyan")
        st.image(q_img, width=200)
        q_active = (query_pattern > 0.01).sum()
        q_sparsity = q_active / 16_777_216 * 100
        st.caption(f"Active: {q_active:,} | {q_sparsity:.2f}%")
        q_status, q_type = get_status(query_pattern)
        getattr(st, q_type)(q_status)

        # Top result pattern (green)
        if top_result_pattern is not None:
            st.markdown("**Top Result**")
            r_img = make_lattice_image(top_result_pattern, "green")
            st.image(r_img, width=200)
            r_active = (top_result_pattern > 0.01).sum()
            r_sparsity = r_active / 16_777_216 * 100
            st.caption(f"Active: {r_active:,} | {r_sparsity:.2f}%")
            r_status, r_type = get_status(top_result_pattern)
            getattr(st, r_type)(r_status)

    with col_results:
        st.markdown("### Results")

        # Scrollable results container (height in pixels)
        with st.container(height=550):
            for rank, (score, idx) in enumerate(results[:top_k]):
                txt = texts[idx]
                lines = txt.splitlines() if txt else ["[empty]"]
                title = lines[0][:100]

                with st.container(border=True):
                    # Header row
                    hcol1, hcol2, hcol3 = st.columns([6, 1, 1])
                    with hcol1:
                        st.markdown(f"**{rank+1}. {title}**")
                    with hcol2:
                        st.caption(f"{score:.3f}")
                    with hcol3:
                        if st.button("X", key=f"del_{idx}"):
                            st.session_state.deleted_ids.add(idx)
                            st.rerun()

                    # Preview
                    if len(lines) > 1:
                        preview = " ".join(lines[1:3])[:200]
                        st.caption(preview + "..." if len(preview) >= 200 else preview)

                    # Expand for full text
                    with st.expander("Full text"):
                        st.text(txt[:2000])
