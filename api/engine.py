"""
Singleton engine manager for Vector+ Studio.

Wraps MultiLatticeCUDAv7 + CombinedEncoder + SentenceTransformer into
a single state object that the API endpoints share.
"""

import os
import sys
import threading
import time
import numpy as np
import zlib

from . import cart_context

# Add current + parent dirs so we can import wrapper, encoder, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
from region_fill_encoder import RegionFillEncoderNomic768

# ---------------------------------------------------------------------------
# Multimodal encoders (extracted from v83)
# ---------------------------------------------------------------------------

class TextRegionEncoder:
    """Encodes compressed text into free lattice regions."""

    def __init__(self, lattice_size=4096, region_size=64):
        self.lattice_size = lattice_size
        self.region_size = region_size
        # Region-fill uses rows 0-11 for 768 embedding dims.
        # Text goes in rows 12+ to avoid collision.
        self.free_rows = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 63]
        self.num_region_cols = 64
        self.max_bytes = len(self.free_rows) * self.num_region_cols  # 704
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

    def decompress_text(self, data: bytes) -> str | None:
        try:
            return zlib.decompress(data).decode('utf-8')
        except Exception:
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

    def decode_text(self, lattice: np.ndarray, expected_length: int) -> str | None:
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
        self.embedding_encoder = RegionFillEncoderNomic768()
        self.text_encoder = TextRegionEncoder()

    def encode(self, embedding: np.ndarray, text: str) -> tuple:
        embedding_layer = self.embedding_encoder.encode(embedding).astype(np.float32)
        text_layer, compressed_len = self.text_encoder.encode_text(text)
        combined = np.maximum(embedding_layer, text_layer)
        metadata = {
            'compressed_len': compressed_len,
            'original_text_len': len(text),
        }
        return combined, metadata

    def decode_text_only(self, lattice: np.ndarray, compressed_len: int) -> str | None:
        return self.text_encoder.decode_text(lattice, compressed_len)

    def decode_embedding_only(self, lattice: np.ndarray) -> np.ndarray:
        return self.embedding_encoder.decode(lattice)


# ---------------------------------------------------------------------------
# Training encoder (matches test_cam_poseidon.py exactly)
# ---------------------------------------------------------------------------
# Text rows for training exclude row 63 — hippocampus goes there instead.
# This matches the standalone CAM test that produces ~80% sign preservation.
TRAIN_TEXT_ROWS = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


class TrainingEncoder:
    """Encode patterns for Hebbian training: region-fill + text + hippocampus.

    Matches test_cam_poseidon.py pattern assembly exactly:
    - Region-fill embedding in rows 0-11
    - Compressed text in rows [15, 20, 25, 30, 35, 40, 45, 50, 55, 60] (no row 63)
    - Hippocampus binary ID in row 63
    """

    def __init__(self):
        self.embedding_encoder = RegionFillEncoderNomic768()
        self.text_encoder = TextRegionEncoder()
        # Override: exclude row 63 from text (hippocampus goes there)
        self.text_encoder.free_rows = TRAIN_TEXT_ROWS
        self.text_encoder.max_bytes = len(TRAIN_TEXT_ROWS) * 64  # 640

    def encode(self, embedding: np.ndarray, text: str,
               pattern_id: int) -> tuple[np.ndarray, dict]:
        """Assemble training pattern: embedding + text + hippocampus."""
        emb_layer = self.embedding_encoder.encode(embedding).astype(np.float32)
        text_layer, compressed_len = self.text_encoder.encode_text(text)
        hippo_layer = encode_hippocampus(pattern_id)
        combined = np.maximum(np.maximum(emb_layer, text_layer), hippo_layer)
        metadata = {
            'compressed_len': compressed_len,
            'original_text_len': len(text),
        }
        return combined, metadata


def encode_hippocampus(pattern_id, lattice_size=4096):
    """Encode pattern ID as binary bits in hippocampus row 63.

    Each bit of the pattern_id fills one 64x64 region in row 63.
    Matches test_hippocampus_survival.py encoding exactly.
    """
    layer = np.zeros((lattice_size, lattice_size), dtype=np.float32)
    bin_str = format(pattern_id & 0xFFFFFFFFFFFFFFFF, "064b")
    region_row = 63
    for i, bit in enumerate(bin_str):
        if bit == "1":
            r0 = region_row * 64
            c0 = i * 64
            layer[r0:r0 + 64, c0:c0 + 64] = 1.0
    return layer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SETTLE_FRAMES = 5
TRAIN_SETTLE_FRAMES = 10  # Match standalone CAM test (was 5)
SIG_SETTLE_FRAMES = 10
PHYSICS_PROFILE = "quality"
HIPPO_ROW = 63  # Episodic index — protected from physics during settle
# Region-fill uses rows 0-11 for embeddings. Text goes in rows 12+.
TEXT_ROWS = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 63]


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

class EngineManager:
    """Holds all mutable state: GPU engine, encoders, embedder, loaded cartridge."""

    def __init__(self):
        # ================= MACHINE STATE =================
        # Loaded once, shared by every cart and every seat. NOT duplicated per mount -- an
        # estimate on 2026-08-05 assumed it was, which made multi-mount look ten times more
        # expensive than it is. Anything added here is per-PROCESS; anything per-cart belongs
        # in cart_context.CartFields instead.
        self.lock = threading.Lock()
        self.ml: MultiLatticeCUDAv7 | None = None
        self.encoder: RegionFillEncoderNomic768 | None = None
        self.combined_encoder: CombinedEncoder | None = None
        self.training_encoder: TrainingEncoder | None = None
        self.embedder = None  # SentenceTransformer (lazy)

        # WebSocket connections for progress broadcasts
        self.ws_connections: list = []

        self.gpu_available = False
        self.engine_ready = False

        # ================= CART STATE =================
        # `mounted_name`, `passages`, `embeddings`, `read_only`, `dirty` and the rest are NOT
        # attributes any more -- they are properties generated at the bottom of this module
        # from cart_context.CartFields, and they read and write THE CALLING REQUEST'S cart.
        # Nothing is initialised here because CartFields owns the defaults; its field
        # defaults match what this constructor used to assign, one for one.

    def boot(self):
        """Initialize the CUDA engine and encoders."""
        try:
            self.ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)
            self.ml.set_profile(PHYSICS_PROFILE)
            self.ml.set_row_physics(HIPPO_ROW, self.ml.ROW_FULLY_PROTECTED)
            self.encoder = RegionFillEncoderNomic768(
                n_dims=768, lattice_size=4096, region_size=64
            )
            self.combined_encoder = CombinedEncoder()
            self.training_encoder = TrainingEncoder()
            self.gpu_available = True
            self.engine_ready = True
            print("[Engine] CUDA engine booted successfully")
        except Exception as e:
            print(f"[Engine] CUDA failed ({e}), running CPU-only")
            self.gpu_available = False
            self.encoder = RegionFillEncoderNomic768(
                n_dims=768, lattice_size=4096, region_size=64
            )
            self.combined_encoder = CombinedEncoder()
            self.training_encoder = TrainingEncoder()
            self.engine_ready = True

    def load_embedder(self):
        """Lazy-load the SentenceTransformer embedder."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.embedder = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
                )
            except Exception:
                self.embedder = SentenceTransformer("all-mpnet-base-v2")
            print("[Engine] Embedder loaded")
        return self.embedder

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query."""
        embedder = self.load_embedder()
        return embedder.encode(f"search_query: {text}")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of documents."""
        embedder = self.load_embedder()
        return embedder.encode(
            [f"search_document: {t}" for t in texts],
            show_progress_bar=True
        )

    def enable_text_protection(self):
        if self.ml:
            self.ml.set_protected_rows(TEXT_ROWS)

    def disable_text_protection(self):
        if self.ml:
            self.ml.set_protected_rows([])

    def unmount(self):
        """Clear THIS REQUEST'S cartridge state, closing a split-cart sidecar if open.

        Delegates to `CartFields.clear()`, which is now the one definition of "what belongs
        to a cart." It used to be this method's twenty-one assignments, and keeping those in
        agreement with anything else was a matter of remembering.

        ⚠ SCOPE. This clears the cart the CALLER is bound to. With nothing bound -- startup,
        CLI, tests, the single-user studio -- that is the process-wide cart, exactly as
        before. Once the pool lands, "this seat is done with its cart" is `pool.release`,
        NOT this: two seats can share one CartFields, and wiping it because one of them
        navigated away would blank the other's screen mid-search.
        """
        cart_context.active().clear()

    def shutdown(self):
        """Clean shutdown."""
        self.unmount()
        self.ml = None
        self.engine_ready = False

    async def broadcast_progress(self, data: dict):
        """Send progress update to all connected WebSocket clients."""
        import json
        message = json.dumps(data)
        dead = []
        for ws in self.ws_connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_connections.remove(ws)


# ---------------------------------------------------------------------------
# Cart state: generated properties, not attributes
# ---------------------------------------------------------------------------
#
# `engine.passages` and its twenty siblings read and write THE CALLING REQUEST'S cart, via
# the ContextVar in cart_context. With nothing bound they use one process-wide CartFields --
# byte-identical to the single mounted cart this class used to hold, which is why startup,
# the CLI, the local studio and every existing test keep working unchanged.
#
# WHY GENERATED RATHER THAN TWENTY-ONE HAND-WRITTEN PROPERTIES. The list then cannot drift
# from CartFields, because it IS CartFields. A hand-written list would need a test to keep
# the two in agreement, and the equivalent test caught a real omission (the three training_*
# fields) the first time it ran -- which is the argument for removing the possibility rather
# than detecting it.
#
# THE COST, stated where someone debugging will find it: these do not autocomplete, and a
# type checker cannot see them. Accepted for the drift-proofing. If you are hunting "wrong
# passages," the answer is almost certainly which cart is bound -- see cart_context.
def _cart_property(field_name: str) -> property:
    def getter(self):
        return getattr(cart_context.active(), field_name)

    def setter(self, value):
        setattr(cart_context.active(), field_name, value)

    getter.__name__ = field_name
    setter.__name__ = field_name
    return property(getter, setter,
                    doc=f"Per-request cart state: {field_name}. See api/cart_context.py.")


for _field_name in cart_context.CartFields.__dataclass_fields__:
    setattr(EngineManager, _field_name, _cart_property(_field_name))
del _field_name


# Module-level singleton
engine = EngineManager()
