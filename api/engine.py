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

# Add parent dir so we can import the wrapper and encoder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
from thermometer_encoder_generic_64x64 import ThermometerEncoderNomic64x64

# ---------------------------------------------------------------------------
# Multimodal encoders (extracted from v83)
# ---------------------------------------------------------------------------

class TextRegionEncoder:
    """Encodes compressed text into free lattice regions."""

    def __init__(self, lattice_size=4096, region_size=64):
        self.lattice_size = lattice_size
        self.region_size = region_size
        self.free_rows = [1, 6, 13, 20, 27, 32, 39, 46, 53, 60, 63]
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
        self.embedding_encoder = ThermometerEncoderNomic64x64()
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
# Config
# ---------------------------------------------------------------------------

SETTLE_FRAMES = 5
TRAIN_SETTLE_FRAMES = 5
SIG_SETTLE_FRAMES = 10
PHYSICS_PROFILE = "quality"
TEXT_ROWS = [1, 6, 13, 20, 27, 32, 39, 46, 53, 60, 63]


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

class EngineManager:
    """Holds all mutable state: GPU engine, encoders, embedder, loaded cartridge."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ml: MultiLatticeCUDAv7 | None = None
        self.encoder: ThermometerEncoderNomic64x64 | None = None
        self.combined_encoder: CombinedEncoder | None = None
        self.embedder = None  # SentenceTransformer (lazy)

        # Cartridge state
        self.mounted_name: str | None = None
        self.mounted_path: str | None = None  # full path if opened from file picker
        self.embeddings: np.ndarray | None = None
        self.passages: list[str] = []
        self.compressed_lens: list[int] = []
        self.compressed_texts: list = []
        self.signatures: np.ndarray | None = None
        self.signatures_loaded = False
        self.multimodal_mode = False
        self.brain_only_mode = False
        self.physics_trained = False
        self.deleted_ids: set[int] = set()
        self.dirty = False  # True when in-memory state differs from disk

        # Background training state
        self.training_active = False
        self.training_progress = 0
        self.training_total = 0

        # WebSocket connections for progress broadcasts
        self.ws_connections: list = []

        self.gpu_available = False
        self.engine_ready = False

    def boot(self):
        """Initialize the CUDA engine and encoders."""
        try:
            self.ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)
            self.ml.set_profile(PHYSICS_PROFILE)
            self.encoder = ThermometerEncoderNomic64x64(
                n_dims=768, lattice_size=4096, region_size=64
            )
            self.combined_encoder = CombinedEncoder()
            self.gpu_available = True
            self.engine_ready = True
            print("[Engine] CUDA engine booted successfully")
        except Exception as e:
            print(f"[Engine] CUDA failed ({e}), running CPU-only")
            self.gpu_available = False
            self.encoder = ThermometerEncoderNomic64x64(
                n_dims=768, lattice_size=4096, region_size=64
            )
            self.combined_encoder = CombinedEncoder()
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
        """Clear all cartridge state."""
        self.mounted_name = None
        self.mounted_path = None
        self.embeddings = None
        self.passages = []
        self.compressed_lens = []
        self.compressed_texts = []
        self.signatures = None
        self.signatures_loaded = False
        self.multimodal_mode = False
        self.brain_only_mode = False
        self.physics_trained = False
        self.deleted_ids = set()
        self.training_active = False
        self.training_progress = 0
        self.training_total = 0
        self.dirty = False

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


# Module-level singleton
engine = EngineManager()
