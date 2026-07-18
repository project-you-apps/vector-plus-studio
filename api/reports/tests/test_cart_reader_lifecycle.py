"""Cart reader lifecycle-byte accessor tests (Hot Stack preliminary).

Run standalone::

    python api/reports/tests/test_cart_reader_lifecycle.py

Or via pytest::

    python -m pytest api/reports/tests/test_cart_reader_lifecycle.py -v

Validates the CartHandle accessors added 2026-07-17 for the Hot Stack
preliminary byte-30 layout:

* truth_status / truth_status_name
* is_urgent / is_important / is_to_consolidate
* is_active (composed with is_tombstoned)
* lifecycle() dict shape

Backward-compat is the load-bearing property: every existing cart in
the wild has byte 30 == 0, which must decode as ACTIVE with no flags.
Legacy carts without a hippocampus array at all must degrade to the
same defaults, matching the ``is_tombstoned`` idiom.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Optional

import numpy as np

if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from api.cartridge_io import (  # noqa: E402
    LIFECYCLE_BYTE_OFFSET,
    TRUTH_STATUS_ACTIVE,
    TRUTH_STATUS_SUPERSEDED,
    TRUTH_STATUS_INCORRECT,
    TRUTH_STATUS_DEFERRED,
    TRUTH_STATUS_ONGOING,
    LIFECYCLE_FLAG_URGENT,
    LIFECYCLE_FLAG_IMPORTANT,
    LIFECYCLE_FLAG_TO_CONSOLIDATE,
    parse_lifecycle_byte,
    pack_lifecycle_byte,
)
from api.reports.cart_reader import CartHandle  # noqa: E402


# ---------------------------------------------------------------------------
# Cart materialization helpers — one per hippocampus shape we test
# ---------------------------------------------------------------------------

def _hippo_row(
    tombstone: bool = False,
    lifecycle: int = 0,
) -> np.ndarray:
    """Return a 64-byte hippo row with configurable tombstone bit (byte 28)
    and lifecycle byte (byte 30)."""
    row = np.zeros(64, dtype=np.uint8)
    if tombstone:
        row[28] = 0x01
    row[30] = lifecycle & 0xFF
    return row


def _hippo_row_legacy_length(lifecycle: int = 0) -> np.ndarray:
    """Return a 30-byte hippo row — SHORTER than the offset of the lifecycle
    byte. Simulates the "row too short to carry the byte" backward-compat
    path so :py:meth:`CartHandle.truth_status` reads as ACTIVE.

    (Real legacy carts still ship a 64-byte row; this shape is synthetic
    to exercise the length guard specifically. Length guard fires when the
    row is shorter than or equal to the offset, so 30 (== LIFECYCLE_BYTE_OFFSET)
    triggers the fallback.)
    """
    row = np.zeros(30, dtype=np.uint8)
    return row


def _make_cart(
    rows: Optional[list[np.ndarray]] = None,
    n_passages: int = 3,
    with_hippocampus: bool = True,
) -> str:
    """Serialize a minimal synthetic cart and return its filesystem path.

    ``rows`` is a list of 64-byte hippocampus rows, one per pattern. If
    ``rows`` is ``None`` and ``with_hippocampus`` is ``True``, we create
    ``n_passages`` all-zeros rows (the default state — ACTIVE, no flags,
    not tombstoned). If ``with_hippocampus`` is ``False``, we omit the
    hippocampus key entirely (simulating brain-only Cart Builder GUI carts
    pre-tombstone convention). Caller owns cleanup.
    """
    if rows is None:
        n = n_passages
        rows = [_hippo_row() for _ in range(n)]
    else:
        n = len(rows)

    embeddings = np.random.default_rng(42).standard_normal(
        size=(n, 8), dtype=np.float32
    ) if n else np.zeros((0, 8), dtype=np.float32)

    passages = np.array(
        [f"pattern-{i} text" for i in range(n)],
        dtype=object,
    )
    sources = np.array(
        [f"src-{i}.txt" for i in range(n)],
        dtype=object,
    )

    fd, path = tempfile.mkstemp(suffix=".cart.npz", prefix="lifecycle_test_")
    os.close(fd)

    payload = {
        "embeddings": embeddings,
        "passages": passages,
        "source_paths": sources,
    }
    if with_hippocampus:
        if n:
            # Rows may have different lengths in the "row too short" test;
            # pad them to a common length so numpy can stack. The guard
            # inside CartHandle checks ``len(row) <= offset`` — 30-byte
            # rows padded to 64 with zeros would silently pass the guard
            # (byte 30 == 0 still decodes as ACTIVE, which is the same
            # answer we want), so we only pad when necessary.
            max_len = max(len(r) for r in rows)
            if all(len(r) == max_len for r in rows):
                hippo = np.stack(rows)
            else:
                padded = []
                for r in rows:
                    if len(r) < max_len:
                        pad = np.zeros(max_len - len(r), dtype=np.uint8)
                        padded.append(np.concatenate([r, pad]))
                    else:
                        padded.append(r)
                hippo = np.stack(padded)
            payload["hippocampus"] = hippo
        else:
            payload["hippocampus"] = np.zeros((0, 64), dtype=np.uint8)

    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# parse_lifecycle_byte + pack_lifecycle_byte
# ---------------------------------------------------------------------------

class TestParsePackLifecycleByte(unittest.TestCase):
    """The two module-level helpers should be exact inverses over the
    representable input space, and they should encode the byte layout
    the recap doc locks."""

    def test_zero_byte_is_active_no_flags(self):
        lc = parse_lifecycle_byte(0)
        self.assertEqual(lc["truth_status"], TRUTH_STATUS_ACTIVE)
        self.assertEqual(lc["truth_status_name"], "ACTIVE")
        self.assertFalse(lc["urgent"])
        self.assertFalse(lc["important"])
        self.assertFalse(lc["to_consolidate"])
        self.assertEqual(lc["raw"], 0)

    def test_pack_roundtrip_for_each_status(self):
        for status in (
            TRUTH_STATUS_ACTIVE,
            TRUTH_STATUS_SUPERSEDED,
            TRUTH_STATUS_INCORRECT,
            TRUTH_STATUS_DEFERRED,
            TRUTH_STATUS_ONGOING,
        ):
            b = pack_lifecycle_byte(truth_status=status)
            self.assertEqual(parse_lifecycle_byte(b)["truth_status"], status)

    def test_flags_are_independent(self):
        b = pack_lifecycle_byte(
            truth_status=TRUTH_STATUS_ONGOING,
            urgent=True,
            important=True,
            to_consolidate=True,
        )
        lc = parse_lifecycle_byte(b)
        self.assertEqual(lc["truth_status"], TRUTH_STATUS_ONGOING)
        self.assertTrue(lc["urgent"])
        self.assertTrue(lc["important"])
        self.assertTrue(lc["to_consolidate"])

    def test_reserved_status_reads_as_unknown(self):
        # Values 5-7 aren't in the enum yet; verify the reader flags them
        # rather than silently accepting them as ACTIVE.
        for reserved in (5, 6, 7):
            lc = parse_lifecycle_byte(reserved)
            self.assertEqual(lc["truth_status"], reserved)
            self.assertEqual(lc["truth_status_name"], "UNKNOWN")

    def test_pack_rejects_out_of_range_status(self):
        with self.assertRaises(ValueError):
            pack_lifecycle_byte(truth_status=8)

    def test_bit_layout_matches_the_recap_doc(self):
        # Explicit hex checks so a bit-shift regression is loud.
        self.assertEqual(pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE), 0x00)
        self.assertEqual(pack_lifecycle_byte(truth_status=TRUTH_STATUS_SUPERSEDED), 0x01)
        self.assertEqual(pack_lifecycle_byte(truth_status=TRUTH_STATUS_ONGOING), 0x04)
        self.assertEqual(pack_lifecycle_byte(urgent=True), LIFECYCLE_FLAG_URGENT)
        self.assertEqual(pack_lifecycle_byte(important=True), LIFECYCLE_FLAG_IMPORTANT)
        self.assertEqual(pack_lifecycle_byte(to_consolidate=True), LIFECYCLE_FLAG_TO_CONSOLIDATE)
        self.assertEqual(LIFECYCLE_FLAG_URGENT, 0x08)
        self.assertEqual(LIFECYCLE_FLAG_IMPORTANT, 0x10)
        self.assertEqual(LIFECYCLE_FLAG_TO_CONSOLIDATE, 0x20)
        self.assertEqual(LIFECYCLE_BYTE_OFFSET, 30)


# ---------------------------------------------------------------------------
# CartHandle accessors — default state
# ---------------------------------------------------------------------------

class TestDefaultStateReadsAsActive(unittest.TestCase):
    """Every existing cart on disk today has byte 30 == 0. Every accessor
    must return the default (ACTIVE, no flags, not tombstoned)."""

    def test_all_zero_row_reads_as_active(self):
        path = _make_cart(n_passages=3)
        try:
            cart = CartHandle(path)
            for i in range(cart.count):
                self.assertEqual(cart.truth_status(i), TRUTH_STATUS_ACTIVE)
                self.assertEqual(cart.truth_status_name(i), "ACTIVE")
                self.assertFalse(cart.is_urgent(i))
                self.assertFalse(cart.is_important(i))
                self.assertFalse(cart.is_to_consolidate(i))
                self.assertFalse(cart.is_tombstoned(i))
                self.assertTrue(cart.is_active(i))
        finally:
            os.unlink(path)

    def test_missing_hippocampus_reads_as_active(self):
        # Brain-only / pre-tombstone carts have no hippocampus key at all.
        # Every accessor must degrade to the safe default, exactly like
        # is_tombstoned already does.
        path = _make_cart(n_passages=3, with_hippocampus=False)
        try:
            cart = CartHandle(path)
            for i in range(cart.count):
                self.assertEqual(cart.truth_status(i), TRUTH_STATUS_ACTIVE)
                self.assertFalse(cart.is_urgent(i))
                self.assertFalse(cart.is_important(i))
                self.assertFalse(cart.is_to_consolidate(i))
                self.assertTrue(cart.is_active(i))
        finally:
            os.unlink(path)

    def test_row_too_short_reads_as_active(self):
        # A row of length 30 doesn't reach byte 30 (index is out of range).
        # The length guard should catch it and return the default.
        rows = [_hippo_row_legacy_length()] * 2
        path = _make_cart(rows=rows)
        try:
            cart = CartHandle(path)
            for i in range(cart.count):
                self.assertEqual(cart.truth_status(i), TRUTH_STATUS_ACTIVE)
                self.assertTrue(cart.is_active(i))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CartHandle accessors — explicit non-default states
# ---------------------------------------------------------------------------

class TestExplicitLifecycleStates(unittest.TestCase):
    """When byte 30 carries a real value, every accessor must decode it."""

    def test_each_truth_status_value_decodes(self):
        rows = [
            _hippo_row(lifecycle=pack_lifecycle_byte(truth_status=s))
            for s in (
                TRUTH_STATUS_ACTIVE,
                TRUTH_STATUS_SUPERSEDED,
                TRUTH_STATUS_INCORRECT,
                TRUTH_STATUS_DEFERRED,
                TRUTH_STATUS_ONGOING,
            )
        ]
        path = _make_cart(rows=rows)
        try:
            cart = CartHandle(path)
            self.assertEqual(cart.truth_status(0), TRUTH_STATUS_ACTIVE)
            self.assertEqual(cart.truth_status(1), TRUTH_STATUS_SUPERSEDED)
            self.assertEqual(cart.truth_status(2), TRUTH_STATUS_INCORRECT)
            self.assertEqual(cart.truth_status(3), TRUTH_STATUS_DEFERRED)
            self.assertEqual(cart.truth_status(4), TRUTH_STATUS_ONGOING)
            self.assertEqual(cart.truth_status_name(1), "SUPERSEDED")
            self.assertEqual(cart.truth_status_name(4), "ONGOING")
        finally:
            os.unlink(path)

    def test_flags_and_status_coexist(self):
        rows = [
            _hippo_row(lifecycle=pack_lifecycle_byte(
                truth_status=TRUTH_STATUS_SUPERSEDED,
                important=True,
            )),
            _hippo_row(lifecycle=pack_lifecycle_byte(
                truth_status=TRUTH_STATUS_ONGOING,
                urgent=True,
                important=True,
                to_consolidate=True,
            )),
        ]
        path = _make_cart(rows=rows)
        try:
            cart = CartHandle(path)
            # Row 0: SUPERSEDED + IMPORTANT — the canonical-correction case
            # the taxonomy doc §2.5 explicitly justifies.
            self.assertEqual(cart.truth_status(0), TRUTH_STATUS_SUPERSEDED)
            self.assertTrue(cart.is_important(0))
            self.assertFalse(cart.is_urgent(0))
            # SUPERSEDED means is_active is False even without tombstone.
            self.assertFalse(cart.is_active(0))
            # Row 1: all three flags on top of ONGOING.
            self.assertEqual(cart.truth_status(1), TRUTH_STATUS_ONGOING)
            self.assertTrue(cart.is_urgent(1))
            self.assertTrue(cart.is_important(1))
            self.assertTrue(cart.is_to_consolidate(1))
            self.assertFalse(cart.is_active(1))
        finally:
            os.unlink(path)

    def test_tombstone_and_lifecycle_are_independent(self):
        # Byte 28 (tombstone) and byte 30 (lifecycle) live in different
        # bytes and must not interfere. A tombstoned ACTIVE row is
        # NOT active (tombstone wins in is_active), but truth_status
        # itself should still read cleanly.
        rows = [
            _hippo_row(tombstone=True, lifecycle=pack_lifecycle_byte(
                truth_status=TRUTH_STATUS_ACTIVE,
            )),
            _hippo_row(tombstone=False, lifecycle=pack_lifecycle_byte(
                truth_status=TRUTH_STATUS_ACTIVE,
                urgent=True,
            )),
        ]
        path = _make_cart(rows=rows)
        try:
            cart = CartHandle(path)
            # Row 0: tombstoned, ACTIVE — is_active must be False.
            self.assertTrue(cart.is_tombstoned(0))
            self.assertEqual(cart.truth_status(0), TRUTH_STATUS_ACTIVE)
            self.assertFalse(cart.is_active(0))
            # Row 1: live, ACTIVE, URGENT — is_active must be True.
            self.assertFalse(cart.is_tombstoned(1))
            self.assertEqual(cart.truth_status(1), TRUTH_STATUS_ACTIVE)
            self.assertTrue(cart.is_urgent(1))
            self.assertTrue(cart.is_active(1))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# lifecycle() dict shape
# ---------------------------------------------------------------------------

class TestLifecycleDict(unittest.TestCase):
    def test_lifecycle_dict_matches_parse_helper(self):
        rows = [
            _hippo_row(lifecycle=pack_lifecycle_byte(
                truth_status=TRUTH_STATUS_DEFERRED,
                important=True,
            )),
        ]
        path = _make_cart(rows=rows)
        try:
            cart = CartHandle(path)
            lc = cart.lifecycle(0)
            # Same shape as parse_lifecycle_byte.
            self.assertEqual(lc["truth_status"], TRUTH_STATUS_DEFERRED)
            self.assertEqual(lc["truth_status_name"], "DEFERRED")
            self.assertTrue(lc["important"])
            self.assertFalse(lc["urgent"])
            self.assertFalse(lc["to_consolidate"])
        finally:
            os.unlink(path)

    def test_lifecycle_dict_for_default_row(self):
        # Missing byte should still produce the well-formed dict.
        path = _make_cart(n_passages=1, with_hippocampus=False)
        try:
            cart = CartHandle(path)
            lc = cart.lifecycle(0)
            self.assertEqual(lc["truth_status"], TRUTH_STATUS_ACTIVE)
            self.assertEqual(lc["truth_status_name"], "ACTIVE")
            self.assertFalse(lc["urgent"])
            self.assertFalse(lc["important"])
            self.assertFalse(lc["to_consolidate"])
            self.assertEqual(lc["raw"], 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
