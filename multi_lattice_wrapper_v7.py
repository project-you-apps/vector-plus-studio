"""
MULTI-LATTICE WRAPPER V7 - UNIFIED PHYSICS ENGINE
==================================================

Python wrapper for lattice_v7.dll with runtime profile switching.

Profiles:
    - FAST: Bulk ingestion, signature ops (minimal physics)
    - BALANCED: Interactive demo, good recall (default)
    - QUALITY: Best correlation, full V4 physics

Usage:
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7

    ml = MultiLatticeCUDAv7()
    ml.set_profile("fast")        # For bulk ingestion
    ml.imprint_vector(embedding)
    ml.settle(frames=2, learn=True)

    ml.set_profile("quality")     # For recall
    ml.imprint_vector(query)
    ml.settle(frames=30, learn=False)
    result = ml.recall()
"""

import ctypes
import numpy as np
import os
import time


# ============================================================
# PHYSICS CONFIG STRUCT (matches lattice_v7.h)
# ============================================================

class PhysicsConfig(ctypes.Structure):
    _fields_ = [
        # Core dynamics
        ("energy_persist", ctypes.c_float),
        ("fatigue_rate", ctypes.c_float),
        ("fatigue_recovery", ctypes.c_float),
        ("temperature", ctypes.c_float),

        # Mexican hat
        ("facilitation", ctypes.c_float),
        ("inhibition", ctypes.c_float),
        ("anti_facilitation_mult", ctypes.c_float),

        # Learning
        ("hebbian_rate", ctypes.c_float),

        # Hierarchy gains - bottom-up
        ("alpha_l4", ctypes.c_float),
        ("alpha_l3", ctypes.c_float),
        ("alpha_l2", ctypes.c_float),
        ("alpha_l1", ctypes.c_float),

        # Hierarchy gains - top-down
        ("beta_l4", ctypes.c_float),
        ("beta_l3", ctypes.c_float),
        ("beta_l2", ctypes.c_float),
        ("beta_l1", ctypes.c_float),

        # Layer persistence
        ("l3_persist", ctypes.c_float),
        ("l2_persist", ctypes.c_float),
        ("l1_persist", ctypes.c_float),

        # V7 Profile controls
        ("kwta_threshold", ctypes.c_float),
        ("hierarchy_depth", ctypes.c_int),
        ("temp_annealing", ctypes.c_int),
        ("hebbian_in_settle", ctypes.c_int),
        ("hybrid_topdown", ctypes.c_int),

        # Reserved
        ("reserved", ctypes.c_int * 2),
    ]

    @classmethod
    def default(cls):
        """Return BALANCED profile defaults."""
        return cls(
            energy_persist=0.96,
            fatigue_rate=0.2,
            fatigue_recovery=0.94,
            temperature=14.0,
            facilitation=0.27,
            inhibition=0.055,
            anti_facilitation_mult=0.5,
            hebbian_rate=0.25,
            alpha_l4=0.86,
            alpha_l3=0.5,
            alpha_l2=0.5,
            alpha_l1=0.5,
            beta_l4=1.0,
            beta_l3=0.7,
            beta_l2=0.6,
            beta_l1=0.5,
            l3_persist=0.1,
            l2_persist=0.2,
            l1_persist=0.2,
            kwta_threshold=0.35,
            hierarchy_depth=2,
            temp_annealing=1,
            hebbian_in_settle=1,
            hybrid_topdown=1,
        )


# ============================================================
# PHYSICS PROFILE ENUM
# ============================================================

class PhysicsProfile:
    FAST = 0
    BALANCED = 1
    QUALITY = 2


# ============================================================
# MAIN WRAPPER CLASS
# ============================================================

class MultiLatticeCUDAv7:
    """
    V7 Unified Lattice Engine with runtime profile switching.

    Key improvements over V6:
    - Restored V4 physics: anti-facilitation, hybrid top-down, kWTA, full hierarchy
    - Runtime profile switching: fast/balanced/quality
    - All V6 features: GPU imprint, hippocampus, signatures, brain dump
    """

    def __init__(self, lattice_size: int = 4096, max_layers: int = 4, verbose: int = 1,
                 cooldown_sec: float = 0.0):
        """
        Initialize V7 engine.

        Args:
            lattice_size: Width/height of L4 lattice (default 4096)
            max_layers: Reserved for future use (default 4)
            verbose: Print status messages (0=quiet, 1=normal)
            cooldown_sec: Sleep after heavy GPU ops (default 0 = disabled)
                          Recommended: 0.125 for bulk ingestion to prevent thermal issues
        """
        self.size = lattice_size
        self.verbose = verbose
        self.cooldown_sec = cooldown_sec

        # Load DLL
        dll_name = "lattice_v7.dll"
        dll_path = os.path.join(os.path.dirname(__file__), dll_name)

        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"Cannot find {dll_path}. "
                f"Compile with: nvcc -shared -o lattice_v7.dll lattice_cuda_v7.cu -O3 -DLATTICE_EXPORTS"
            )

        self.lib = ctypes.CDLL(dll_path)
        self._setup_function_signatures()

        # Create engine
        if self.verbose:
            print(f"Initializing V7 Unified Engine ({lattice_size}x{lattice_size})...")

        self.engine = self.lib.CreateEngine(lattice_size, max_layers)

        if not self.engine:
            raise RuntimeError("Failed to create V7 engine")

        self.current_profile = "balanced"

    def _setup_function_signatures(self):
        """Define ctypes argument/return types for all API functions."""

        # Lifecycle
        self.lib.CreateEngine.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.CreateEngine.restype = ctypes.c_void_p

        self.lib.DestroyEngine.argtypes = [ctypes.c_void_p]
        self.lib.DestroyEngine.restype = None

        # Configuration
        self.lib.SetPhysics.argtypes = [ctypes.c_void_p, PhysicsConfig]
        self.lib.SetPhysics.restype = None

        self.lib.SetProfile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.SetProfile.restype = None

        self.lib.GetProfileDefaults.argtypes = [ctypes.c_int]
        self.lib.GetProfileDefaults.restype = PhysicsConfig

        # Operations
        self.lib.ImprintPattern.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.ImprintPattern.restype = None

        self.lib.ImprintNomic.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
        ]
        self.lib.ImprintNomic.restype = None

        self.lib.ResetLattice.argtypes = [ctypes.c_void_p]
        self.lib.ResetLattice.restype = None

        self.lib.LearnImmediate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.LearnImmediate.restype = None

        self.lib.Settle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
        self.lib.Settle.restype = None

        self.lib.Recall.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.Recall.restype = None

        self.lib.EncodeHippocampus.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        self.lib.EncodeHippocampus.restype = None

        # Search
        self.lib.GenerateSignature.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.GenerateSignature.restype = None

        self.lib.ScanSignatures.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
        ]
        self.lib.ScanSignatures.restype = None

        # Persistence
        self.lib.SaveCore.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        ]
        self.lib.SaveCore.restype = None

        self.lib.LoadCore.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        ]
        self.lib.LoadCore.restype = None

        # Diagnostic
        self.lib.GetActiveCount.argtypes = [ctypes.c_void_p]
        self.lib.GetActiveCount.restype = ctypes.c_int

    def __del__(self):
        """Clean up GPU resources."""
        if hasattr(self, "engine") and self.engine:
            self.lib.DestroyEngine(self.engine)

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def set_physics(self, config: PhysicsConfig):
        """Set physics parameters directly."""
        self.lib.SetPhysics(self.engine, config)

    def set_profile(self, profile: str):
        """
        Switch physics profile at runtime.

        Args:
            profile: "fast", "balanced", or "quality"

        Profiles:
            - fast: Minimal physics for bulk ops (500+ patterns/sec)
            - balanced: Good recall quality (default)
            - quality: Full V4 physics, best correlation
        """
        profiles = {
            "fast": PhysicsProfile.FAST,
            "balanced": PhysicsProfile.BALANCED,
            "quality": PhysicsProfile.QUALITY,
        }

        profile_lower = profile.lower()
        if profile_lower not in profiles:
            raise ValueError(f"Unknown profile: {profile}. Use: fast, balanced, quality")

        self.lib.SetProfile(self.engine, profiles[profile_lower])
        self.current_profile = profile_lower

    def get_profile_defaults(self, profile: str) -> PhysicsConfig:
        """
        Get the default PhysicsConfig for a profile.

        Useful for inspection or as a starting point for customization.
        """
        profiles = {
            "fast": PhysicsProfile.FAST,
            "balanced": PhysicsProfile.BALANCED,
            "quality": PhysicsProfile.QUALITY,
        }
        return self.lib.GetProfileDefaults(profiles[profile.lower()])

    # ============================================================
    # CORE OPERATIONS
    # ============================================================

    def imprint_pattern(self, pattern: np.ndarray):
        """
        Imprint a raw float pattern (16M values for 4096x4096).

        Args:
            pattern: 2D or 1D array, values > 0.5 become ON
        """
        pat_c = np.ascontiguousarray(pattern.flatten(), dtype=np.float32)
        self.lib.ImprintPattern(self.engine, pat_c)

    def imprint_vector(self, embedding: np.ndarray, normalize: str = "auto"):
        """
        Imprint a Nomic embedding directly (768 or similar dims).

        Uses GPU-native thermometer expansion. The thermometer encoder
        expects values in [0, 1] range. This method auto-detects and
        normalizes embeddings that fall outside this range.

        Args:
            embedding: 1D array of embedding values (typically 768-dim)
            normalize: "auto" (default) - detect and normalize if needed
                       "always" - force normalization
                       "never" - pass through raw (use if pre-normalized)
        """
        emb = embedding.flatten().astype(np.float32)
        val_min, val_max = emb.min(), emb.max()

        needs_norm = False
        if normalize == "always":
            needs_norm = True
        elif normalize == "auto":
            # Heuristic: if values significantly outside [0,1], normalize
            # Nomic embeddings typically range [-4, +4]
            if val_min < -0.1 or val_max > 1.1:
                needs_norm = True

        if needs_norm:
            # Linear rescale to [0, 1]
            val_range = val_max - val_min
            if val_range > 1e-6:
                emb = (emb - val_min) / val_range
            else:
                emb = np.full_like(emb, 0.5)  # Degenerate case: constant embedding

            if self.verbose > 1:
                print(f"  Normalized embedding [{val_min:.2f}, {val_max:.2f}] -> [0, 1]")

        emb_c = np.ascontiguousarray(emb, dtype=np.float32)
        self.lib.ImprintNomic(self.engine, emb_c, len(emb_c))

    def reset(self):
        """
        Clear all neuron states to zero (fresh slate for new pattern).

        Call this before imprint_vector when you need independent patterns
        (e.g., signature building, query processing).
        """
        self.lib.ResetLattice(self.engine)

    def settle(self, frames: int = 30, learn: bool = True):
        """
        Run physics simulation for N frames.

        Args:
            frames: Number of physics frames to run
            learn: If True and hebbian_in_settle enabled, update weights
        """
        self.lib.Settle(self.engine, frames, learn)

        # Optional cooldown to prevent GPU thermal issues during bulk ops
        if self.cooldown_sec > 0:
            time.sleep(self.cooldown_sec)

    def learn_immediate(self, iterations: int = 5):
        """
        Apply Hebbian learning directly without running physics.

        This is V4's approach: learn the clean imprinted pattern before
        any degradation from physics. Call after imprint, before settle.

        Args:
            iterations: Number of Hebbian update passes (default: 5)
        """
        self.lib.LearnImmediate(self.engine, iterations)

    def recall(self) -> np.ndarray:
        """
        Extract current L4 state as float array.

        Returns:
            2D array (4096x4096) with 1.0 = ON, 0.0 = OFF
        """
        out = np.zeros(self.size * self.size, dtype=np.float32)
        self.lib.Recall(self.engine, out)
        return out.reshape(self.size, self.size)

    def encode_hippocampus(self, pattern_id: int, is_deleted: bool = False):
        """
        Encode pattern ID into hippocampus region (row 63).

        Args:
            pattern_id: Integer ID to encode (up to 60 bits)
            is_deleted: If True, sets tombstone flag
        """
        self.lib.EncodeHippocampus(self.engine, pattern_id, is_deleted)

    # ============================================================
    # SEARCH OPERATIONS
    # ============================================================

    def generate_signature(self) -> np.ndarray:
        """
        Generate 4096-float signature from current L4 state.

        Each value is mean activation of a 64x64 region.

        Returns:
            1D array of 4096 floats
        """
        sig = np.zeros(4096, dtype=np.float32)
        self.lib.GenerateSignature(self.engine, sig)
        return sig

    def scan_signatures(self, query_sig: np.ndarray, signature_db: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and database signatures.

        Args:
            query_sig: 4096-float query signature
            signature_db: 2D array (N x 4096) of database signatures

        Returns:
            1D array of N similarity scores
        """
        num_items = signature_db.shape[0]
        query = np.ascontiguousarray(query_sig, dtype=np.float32)
        db = np.ascontiguousarray(signature_db.flatten(), dtype=np.float32)
        indices = np.zeros(num_items, dtype=np.int32)
        scores = np.zeros(num_items, dtype=np.float32)

        self.lib.ScanSignatures(self.engine, query, db, num_items, indices, scores, 0)
        return scores

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def save_brain(self, filename: str):
        """
        Save L4 state + weights to file.

        For 4096x4096: 2 * 16M * 4 bytes = 134 MB

        Args:
            filename: Output .npy file path
        """
        total_ints = self.size * self.size * 2
        buffer = np.zeros(total_ints, dtype=np.uint32)
        self.lib.SaveCore(self.engine, buffer)
        np.save(filename, buffer)

        if self.verbose:
            print(f"Brain saved to {filename}")

    def load_brain(self, filename: str) -> bool:
        """
        Load L4 state + weights from file.

        Args:
            filename: Input .npy file path

        Returns:
            True if successful, False if file not found
        """
        if not os.path.exists(filename):
            return False

        buffer = np.load(filename)
        buffer = np.ascontiguousarray(buffer, dtype=np.uint32)
        self.lib.LoadCore(self.engine, buffer)

        if self.verbose:
            print(f"Brain restored from {filename}")

        return True

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def get_active_count(self) -> int:
        """Get current number of active (ON) neurons."""
        return self.lib.GetActiveCount(self.engine)

    def get_on_ratio(self, expected_density: float = 0.048) -> float:
        """
        Get ratio of actual active neurons to expected.

        Args:
            expected_density: Expected sparsity (default 4.8%)

        Returns:
            Ratio (1.0 = exactly as expected, >1.0 = bloom, <1.0 = death)
        """
        active = self.get_active_count()
        expected = self.size * self.size * expected_density
        return active / (expected + 1e-8)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    import pickle

    print("=" * 60)
    print("V7 WRAPPER QUICK TEST - WITH REAL EMBEDDINGS")
    print("=" * 60)

    # Create engine
    ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)

    # Test profile switching
    print("\nTesting profile switching...")
    ml.set_profile("fast")
    ml.set_profile("balanced")
    ml.set_profile("quality")

    # Load REAL embeddings from pkl file
    print("\nLoading real Nomic embeddings from pkl file...")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pkl_path = os.path.join(data_dir, "wiki_nomic_10k.pkl")

    embeddings = None  # Will hold the full array if loaded
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Handle different pkl formats
        if isinstance(data, dict) and "embeddings" in data:
            embeddings = np.array(data["embeddings"])
        elif isinstance(data, dict) and "data" in data:
            embeddings = np.array(data["data"])
        elif isinstance(data, list):
            embeddings = np.array(data)
        else:
            embeddings = np.array(data)

        print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
        test_emb = embeddings[0].astype(np.float32)
        print(f"Test embedding dims: {len(test_emb)}, range: [{test_emb.min():.4f}, {test_emb.max():.4f}]")
        if test_emb.min() < -0.1 or test_emb.max() > 1.1:
            print("  -> Will be auto-normalized to [0, 1] for thermometer encoding")
    else:
        print(f"WARNING: Could not find {pkl_path}")
        print("Falling back to synthetic wave pattern...")
        test_emb = np.zeros(768, dtype=np.float32)
        for i in range(768):
            test_emb[i] = 0.5 + 0.4 * np.sin(i * 0.05) + 0.1 * np.cos(i * 0.13)
        test_emb = np.clip(test_emb, 0.0, 1.0)

    # Test with BALANCED profile
    print("\nTesting imprint + settle + recall with BALANCED profile...")
    ml.set_profile("balanced")
    ml.imprint_vector(test_emb)
    ml.settle(frames=10, learn=True)

    result = ml.recall()
    active = (result > 0.5).sum()
    ratio = ml.get_on_ratio()

    print(f"Active neurons: {active:,}")
    print(f"On-ratio: {ratio:.3f}")

    # Check if in acceptable range
    if 0.9 <= ratio <= 1.3:
        print("On-ratio: GOOD (within 0.9-1.3 range)")
    elif ratio < 0.9:
        print("On-ratio: WARNING - possible pattern death")
    else:
        print("On-ratio: WARNING - mild bloom (may need tuning)")

    # Test signature
    print("\nTesting signature generation...")
    sig = ml.generate_signature()
    print(f"Signature shape: {sig.shape}, range: [{sig.min():.4f}, {sig.max():.4f}]")

    # Test with QUALITY profile to compare bloom control
    print("\nTesting same embedding with QUALITY profile...")
    ml.set_profile("quality")
    ml.imprint_vector(test_emb)
    ml.settle(frames=10, learn=True)
    ratio_quality = ml.get_on_ratio()
    print(f"QUALITY profile on-ratio: {ratio_quality:.3f}")

    # Test with FAST profile (no anti-facilitation)
    print("\nTesting same embedding with FAST profile...")
    ml.set_profile("fast")
    ml.imprint_vector(test_emb)
    ml.settle(frames=10, learn=True)
    ratio_fast = ml.get_on_ratio()
    print(f"FAST profile on-ratio: {ratio_fast:.3f}")

    # Summary
    print("\n" + "-" * 40)
    print("PROFILE COMPARISON:")
    print(f"  FAST:     {ratio_fast:.3f}  (no anti-facil)")
    print(f"  BALANCED: {ratio:.3f}  (0.5x anti-facil)")
    print(f"  QUALITY:  {ratio_quality:.3f}  (1.0x anti-facil)")
    print("-" * 40)

    if 0.9 <= ratio_quality <= 1.1:
        print("QUALITY profile is well-tuned!")
    elif ratio_quality > 1.1:
        print("Suggestion: Increase kWTA aggressiveness or anti-facil")
    else:
        print("Suggestion: Reduce anti-facil or kWTA threshold")

    print("\n" + "=" * 60)
    print("V7 WRAPPER TEST COMPLETE")
    print("=" * 60)
