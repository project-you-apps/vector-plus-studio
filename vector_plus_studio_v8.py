"""
Vector+ Studio V8 - Honest Physics

What the lattice ACTUALLY does well:
1. Query visualization - see the neural activation pattern
2. Query cleaning - settle stabilizes noisy/partial inputs
3. Noise-tolerant recall - 86% correlation at 30% noise

What we use for search:
- Embedding cosine similarity (fast, accurate)
- Keyword boosting for title matches

The lattice visualization shows what's happening, not magic ranking.
"""

import streamlit as st

st.set_page_config(
    page_title="Vector+ Studio v0.8",
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
sys.path.insert(0, current_dir)

try:
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
except ImportError:
    st.error("V7 wrapper not found.")
    st.stop()

# --- CONFIG ---
SETTLE_FRAMES = 10
PHYSICS_PROFILE = "quality"
DATA_DIR = os.path.join(current_dir, "cartridges")


def load_cartridge(path):
    """Load any cartridge format - we only need embeddings + text."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        # V7/V8 format
        if data.get("version") in ["7.0", "8.0"]:
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

# --- BOOT ENGINE ---
if "engine" not in st.session_state:
    with st.spinner("Booting V8 Physics Engine..."):
        try:
            st.session_state.engine = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)
            st.session_state.engine.set_profile(PHYSICS_PROFILE)
        except Exception as e:
            st.error(f"Engine failed: {e}")
            st.stop()


@st.cache_resource(show_spinner="Loading Embedder...")
def load_embedder():
    try:
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
    except:
        return SentenceTransformer("all-mpnet-base-v2")


# Don't preload - let it load lazily on first search


# --- SIDEBAR ---
with st.sidebar:
    st.title("Vector+ v0.8")
    st.caption("Neural Lattice Search")
    st.divider()

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
                st.session_state.status = f"Loaded {len(txt)} entries"
                st.session_state.query = ""  # Clear query on new mount
                st.session_state.deleted_ids = set()  # Clear deletions too
                st.rerun()
            else:
                st.error("Failed to load cartridge")

    st.divider()
    st.caption(f"Status: {st.session_state.status}")

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
                cart = {"version": "8.0", "embeddings": embs, "passages": texts}
                with open(os.path.join(DATA_DIR, f"{name}.pkl"), "wb") as f:
                    pickle.dump(cart, f)
                st.success(f"Saved {name}.pkl")


# --- MAIN ---
st.title("Vector+ Studio v0.8")
st.caption("Neural Lattice Semantic Search")

if not st.session_state.dataset:
    st.info("👈 Mount a cartridge from the sidebar")
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

    # --- LATTICE VISUALIZATION ---
    with st.session_state.engine_lock:
        # Query pattern
        q_min, q_max = q_emb.min(), q_emb.max()
        q_norm = ((q_emb - q_min) / (q_max - q_min + 1e-9)).astype(np.float32)

        ml.reset()
        ml.imprint_vector(q_norm)
        ml.settle(frames=SETTLE_FRAMES, learn=False)
        query_pattern = ml.recall()

    # --- EMBEDDING SEARCH ---
    q_unit = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    e_unit = embeddings / e_norms
    scores = np.dot(e_unit, q_unit)

    # Keyword boost
    for i, txt in enumerate(texts):
        title = txt.splitlines()[0].lower() if txt else ""
        bonus = sum(0.1 for kw in keywords if kw in title)
        scores[i] += min(bonus, 0.3)

    # Get results
    top_idx = np.argsort(scores)[-top_k:][::-1]
    results = [(scores[i], i) for i in top_idx if i not in st.session_state.deleted_ids]

    # Get top result pattern for comparison
    top_result_pattern = None
    if results:
        top_idx_result = results[0][1]
        top_emb = embeddings[top_idx_result]
        with st.session_state.engine_lock:
            t_min, t_max = top_emb.min(), top_emb.max()
            t_norm = ((top_emb - t_min) / (t_max - t_min + 1e-9)).astype(np.float32)
            ml.reset()
            ml.imprint_vector(t_norm)
            ml.settle(frames=SETTLE_FRAMES, learn=False)
            top_result_pattern = ml.recall()

    t1 = time.perf_counter()

    # --- DISPLAY ---
    st.caption(f"⚡ {(t1-t0)*1000:.0f}ms | {len(results)} results")

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
                        if st.button("🗑️", key=f"del_{idx}"):
                            st.session_state.deleted_ids.add(idx)
                            st.rerun()

                    # Preview
                    if len(lines) > 1:
                        preview = " ".join(lines[1:3])[:200]
                        st.caption(preview + "..." if len(preview) >= 200 else preview)

                    # Expand for full text
                    with st.expander("Full text"):
                        st.text(txt[:2000])
