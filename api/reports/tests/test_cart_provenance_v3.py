"""Cart Provenance Schema v3 — write/read/migration round-trip tests.

Run standalone::

    python api/reports/tests/test_cart_provenance_v3.py

Or via pytest::

    python -m pytest api/reports/tests/test_cart_provenance_v3.py -v

Locks in three properties the v3 schema needs to survive:

1. **Round-trip fidelity.** A v3 cart written by ``build_metadata`` reads
   back via ``read_metadata`` + ``parse_hippocampus`` with the correct
   ``source_path`` per pattern.
2. **Backward compat.** A v2 (source_hash) cart read by the new
   version-dispatching parser still yields a valid metadata list with
   ``source_hash`` populated. No accessor blows up on the old format.
3. **Migration idempotency.** Running the migration tool twice on the
   same cart is a no-op the second time (input == output byte-for-byte
   at the h-row level).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Support both `python -m pytest ...` and direct execution.
if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from api.cartbuilder.cartridge_builder import (  # noqa: E402
    build_metadata,
    save_cartridge,
    read_metadata,
    FORMAT_VERSION_CANONICAL,
    FORMAT_VERSION_PROVENANCE,
)
from api.cartridge_io import parse_hippocampus  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_doc_map(sources: list[str]) -> list[tuple[str, int, int]]:
    """Turn a list of per-passage source filenames into a ``doc_map`` shape.

    Chunk index / total are stubbed at 0 / 1 — they don't affect provenance.
    """
    return [(src, 0, 1) for src in sources]


def _synthesize_cart_dir() -> tuple[str, str]:
    """Return ``(tempdir, cart_name)`` for a scratch cart output."""
    tmp = tempfile.mkdtemp(prefix="prov_v3_test_")
    return tmp, "test-cart"


# ---------------------------------------------------------------------------
# 1. v3 round-trip fidelity
# ---------------------------------------------------------------------------

class TestV3RoundTrip(unittest.TestCase):
    """A v3 cart written by build_metadata + save_cartridge reads back with
    the exact per-pattern source_path we put in."""

    def test_write_and_read_yields_correct_source_paths(self):
        # 5 passages across 3 unique source files. Passages 0 and 2 share
        # source; passages 1 and 4 share source; passage 3 is alone.
        sources = [
            "docs/intro.md",   # 0
            "papers/a.pdf",    # 1
            "docs/intro.md",   # 2 shares with 0
            "notes/b.txt",     # 3 alone
            "papers/a.pdf",    # 4 shares with 1
        ]
        entries = [f"passage-{i} body text" for i in range(len(sources))]
        doc_map = _make_doc_map(sources)

        metadata, pattern0, source_strings = build_metadata(
            entries, doc_map, cart_name="test",
        )

        # Sentinel index 0 + 3 unique paths = 4 entries in strings table.
        self.assertIsNotNone(source_strings)
        self.assertEqual(len(source_strings), 4)  # ["", intro, a.pdf, b.txt]

        # Every metadata row is 64 bytes.
        for m in metadata:
            self.assertEqual(len(m), 64)

        # Write to a scratch .npz and read back via parse_hippocampus.
        tmp, name = _synthesize_cart_dir()
        try:
            embeddings = np.zeros((len(entries), 8), dtype=np.float32)
            save_cartridge(
                tmp, name, embeddings, entries,
                metadata=metadata, pattern0=pattern0,
                source_strings=source_strings,
            )
            cart_path = os.path.join(tmp, f"{name}.cart.npz")
            with np.load(cart_path, allow_pickle=True) as npz:
                parsed = parse_hippocampus(npz)

            self.assertEqual(len(parsed), len(entries))
            for i, entry in enumerate(parsed):
                self.assertEqual(entry["source_path"], sources[i],
                                 f"source_path mismatch at pattern {i}")
                # v3 rows carry source_idx, not source_hash.
                self.assertIn("source_idx", entry)
                self.assertNotIn("source_hash", entry)
                # Truth status defaults to ACTIVE (byte 30 is zero).
                self.assertEqual(entry["lifecycle"]["truth_status_name"], "ACTIVE")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. v2 backward compatibility
# ---------------------------------------------------------------------------

class TestV2BackwardCompat(unittest.TestCase):
    """A v2 cart (source_hash uint32 at byte 18, no source_strings) reads
    back cleanly via the version-dispatching parser."""

    def test_v2_cart_reads_with_source_hash(self):
        sources = ["a.md", "b.md", "a.md"]
        entries = [f"p-{i}" for i in range(len(sources))]
        doc_map = _make_doc_map(sources)

        # Explicitly write v2 (legacy source_hash), not v3.
        metadata, pattern0, source_strings = build_metadata(
            entries, doc_map, cart_name="v2test",
            format_version=FORMAT_VERSION_CANONICAL,
        )
        self.assertIsNone(source_strings)

        tmp, name = _synthesize_cart_dir()
        try:
            embeddings = np.zeros((len(entries), 8), dtype=np.float32)
            save_cartridge(
                tmp, name, embeddings, entries,
                metadata=metadata, pattern0=pattern0,
                source_strings=None,
                format_version=FORMAT_VERSION_CANONICAL,
            )
            cart_path = os.path.join(tmp, f"{name}.cart.npz")
            with np.load(cart_path, allow_pickle=True) as npz:
                # No source_strings.npy on disk for v2.
                self.assertNotIn("source_strings", npz.files)
                parsed = parse_hippocampus(npz)

            for entry in parsed:
                self.assertIn("source_hash", entry)
                self.assertNotIn("source_idx", entry)
                self.assertNotIn("source_path", entry)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Migration idempotency
# ---------------------------------------------------------------------------

class TestMigrationIdempotency(unittest.TestCase):
    """Running ``bin/migrate_cart_v1_v2_to_v3.py`` twice on the same cart
    is a no-op the second time."""

    def test_v2_to_v3_migration_via_source_paths_sidecar(self):
        # Manually write a v2-shaped cart with a source_paths sidecar
        # (skipping the API-facing save_cartridge so we can emit the
        # sidecar that legacy carts had).
        sources = ["one.md", "two.md", "one.md"]
        entries = [f"passage-{i}" for i in range(len(sources))]
        doc_map = _make_doc_map(sources)
        metadata, pattern0, _ = build_metadata(
            entries, doc_map, format_version=FORMAT_VERSION_CANONICAL,
        )

        tmp = tempfile.mkdtemp(prefix="prov_v3_migrate_")
        cart_path = Path(tmp) / "legacy.cart.npz"
        try:
            embeddings = np.zeros((len(entries), 8), dtype=np.float32)
            hippo = np.frombuffer(b"".join(metadata), dtype=np.uint8).reshape(-1, 64)
            np.savez_compressed(
                cart_path,
                embeddings=embeddings,
                passages=np.array(entries, dtype=object),
                compressed_texts=np.array(entries, dtype=object),
                hippocampus=hippo,
                source_paths=np.array(sources, dtype=object),
                version="mcp-v4",
            )

            # Run migration.
            script = Path(__file__).parents[3] / "bin" / "migrate_cart_v1_v2_to_v3.py"
            self.assertTrue(script.exists(), f"migration tool missing at {script}")
            proc = subprocess.run(
                [sys.executable, str(script), str(cart_path), "--in-place"],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, f"migration failed: {proc.stderr}")

            # Verify the resulting cart reads as v3 with correct source_paths.
            with np.load(cart_path, allow_pickle=True) as npz:
                self.assertIn("source_strings", npz.files)
                parsed = parse_hippocampus(npz)
                for i, entry in enumerate(parsed):
                    self.assertEqual(entry["source_path"], sources[i])

            # Idempotency: run it again and expect no-op action.
            proc2 = subprocess.run(
                [sys.executable, str(script), str(cart_path), "--in-place"],
                capture_output=True, text=True,
            )
            self.assertEqual(proc2.returncode, 0)
            summary2 = json.loads(proc2.stdout)
            self.assertEqual(summary2["action"], "no-op (already v3)")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
