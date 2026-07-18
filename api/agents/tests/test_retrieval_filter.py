"""Pattern-filter behavior tests for ``api.agents.retrieval``.

Run standalone::

    python api/agents/tests/test_retrieval_filter.py

Or via pytest::

    python -m pytest api/agents/tests/test_retrieval_filter.py -v

Exercises the ``pattern_filter`` parameter added 2026-07-17 to
``retrieve_top_patterns`` — the actual "only current relevant patterns"
unlock the Hot Stack preliminary was designed to ship. Every existing
cart in the wild has byte 30 == 0 (ACTIVE), so every mode reduces to
the pre-Hot-Stack tombstone-only behavior on those carts. This test
suite specifically exercises non-default lifecycle values to lock in
the semantics.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from api.cartridge_io import (  # noqa: E402
    TRUTH_STATUS_ACTIVE,
    TRUTH_STATUS_SUPERSEDED,
    TRUTH_STATUS_INCORRECT,
    TRUTH_STATUS_DEFERRED,
    TRUTH_STATUS_ONGOING,
    pack_lifecycle_byte,
)
from api.agents.retrieval import retrieve_top_patterns  # noqa: E402
from api.reports.cart_reader import CartHandle  # noqa: E402


def _make_cart_with_lifecycles(
    lifecycles: list[int],
    tombstones: list[bool] | None = None,
    perms: list[int] | None = None,
    passages: list[str] | None = None,
) -> str:
    """Build a small synthetic cart where pattern i carries the byte-30
    lifecycle value from ``lifecycles[i]``. Every pattern gets a distinct
    passage so retrieval by query works predictably.

    ``perms[i]`` (optional) sets the byte-29 perms_byte on row i. When
    omitted every pattern gets the legacy default (byte-29 == 0, which
    reads back as PERM_R+PERM_W). Pass an explicit 0x02 (PERM_W only)
    to exclude a pattern from retrieval via the ``is_readable`` gate.
    """
    n = len(lifecycles)
    tombstones = tombstones or [False] * n
    assert len(tombstones) == n
    perms = perms if perms is not None else [0] * n
    assert len(perms) == n
    if passages is None:
        # Distinct token per pattern so top-N retrieval by query is
        # deterministic — pattern i is uniquely identified by "unique-i".
        passages = [f"cabbage sasquatch unique-{i} filler content" for i in range(n)]
    assert len(passages) == n

    embeddings = np.random.default_rng(42).standard_normal(
        size=(n, 8), dtype=np.float32,
    )
    passages_arr = np.array(passages, dtype=object)
    sources_arr = np.array([f"src-{i}.txt" for i in range(n)], dtype=object)

    rows = []
    for lc, tomb, perm in zip(lifecycles, tombstones, perms):
        row = np.zeros(64, dtype=np.uint8)
        if tomb:
            row[28] = 0x01
        row[29] = perm & 0xFF
        row[30] = lc & 0xFF
        rows.append(row)
    hippo = np.stack(rows) if n else np.zeros((0, 64), dtype=np.uint8)

    fd, path = tempfile.mkstemp(suffix=".cart.npz", prefix="filter_test_")
    os.close(fd)
    np.savez(
        path,
        embeddings=embeddings,
        passages=passages_arr,
        source_paths=sources_arr,
        hippocampus=hippo,
    )
    return path


class TestPatternFilterActiveOnly(unittest.TestCase):
    """Default (``pattern_filter="active_only"``) surfaces only ACTIVE
    patterns — the "current relevant patterns" agent semantics."""

    def test_default_mode_skips_all_non_active_statuses(self):
        # Five patterns, one per truth_status value. Only pattern 0 (ACTIVE)
        # should survive the default filter.
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_SUPERSEDED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_INCORRECT),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_DEFERRED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ONGOING),
        ]
        path = _make_cart_with_lifecycles(lifecycles)
        try:
            cart = CartHandle(path)
            hits = retrieve_top_patterns(cart, "cabbage", 10)
            hit_idxs = {h.idx for h in hits}
            self.assertEqual(hit_idxs, {0})
        finally:
            os.unlink(path)

    def test_empty_query_fast_path_also_filters(self):
        # The empty-query fast path (Auto Briefing / Cart Curator with no
        # focus) must respect the filter too — not just the scored path.
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_SUPERSEDED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_DEFERRED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
        ]
        path = _make_cart_with_lifecycles(lifecycles)
        try:
            cart = CartHandle(path)
            hits = retrieve_top_patterns(cart, "", 10)
            hit_idxs = {h.idx for h in hits}
            self.assertEqual(hit_idxs, {1, 3})
        finally:
            os.unlink(path)


class TestPatternFilterIncludeSuperseded(unittest.TestCase):
    """``pattern_filter="include_superseded"`` allows ACTIVE + SUPERSEDED
    through, still excluding INCORRECT / DEFERRED / ONGOING / unknown."""

    def test_include_superseded_mode(self):
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_SUPERSEDED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_INCORRECT),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_DEFERRED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ONGOING),
        ]
        path = _make_cart_with_lifecycles(lifecycles)
        try:
            cart = CartHandle(path)
            hits = retrieve_top_patterns(
                cart, "cabbage", 10, pattern_filter="include_superseded",
            )
            hit_idxs = {h.idx for h in hits}
            self.assertEqual(hit_idxs, {0, 1})
        finally:
            os.unlink(path)


class TestPatternFilterAll(unittest.TestCase):
    """``pattern_filter="all"`` returns everything except tombstoned."""

    def test_all_mode_returns_every_non_tombstoned_lifecycle(self):
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_SUPERSEDED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_INCORRECT),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_DEFERRED),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ONGOING),
        ]
        path = _make_cart_with_lifecycles(lifecycles)
        try:
            cart = CartHandle(path)
            hits = retrieve_top_patterns(
                cart, "cabbage", 10, pattern_filter="all",
            )
            hit_idxs = {h.idx for h in hits}
            self.assertEqual(hit_idxs, {0, 1, 2, 3, 4})
        finally:
            os.unlink(path)


class TestTombstoneAlwaysExcluded(unittest.TestCase):
    """Tombstone is orthogonal to lifecycle — it's a delete flag, not a
    truth_status value. Tombstoned patterns must be excluded from every
    filter mode, even ``"all"``."""

    def test_tombstoned_active_excluded_from_all_modes(self):
        # Pattern 0: ACTIVE + tombstoned. Pattern 1: ACTIVE + live.
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
        ]
        tombstones = [True, False]
        path = _make_cart_with_lifecycles(lifecycles, tombstones=tombstones)
        try:
            cart = CartHandle(path)
            for mode in ("active_only", "include_superseded", "all"):
                hits = retrieve_top_patterns(
                    cart, "cabbage", 10, pattern_filter=mode,
                )
                hit_idxs = {h.idx for h in hits}
                self.assertEqual(
                    hit_idxs,
                    {1},
                    f"Tombstoned pattern leaked in mode {mode!r}",
                )
        finally:
            os.unlink(path)


class TestBackwardCompatDefaultCarts(unittest.TestCase):
    """Every existing cart on disk has byte 30 == 0 (ACTIVE). All three
    filter modes must return every non-tombstoned pattern on such carts,
    matching the pre-2026-07-17 behavior."""

    def test_all_zero_bytes_read_as_active_in_every_mode(self):
        lifecycles = [0, 0, 0, 0]
        path = _make_cart_with_lifecycles(lifecycles)
        try:
            cart = CartHandle(path)
            for mode in ("active_only", "include_superseded", "all"):
                hits = retrieve_top_patterns(
                    cart, "cabbage", 10, pattern_filter=mode,
                )
                hit_idxs = {h.idx for h in hits}
                self.assertEqual(
                    hit_idxs,
                    {0, 1, 2, 3},
                    f"Mode {mode!r} lost live patterns on a zero-byte cart",
                )
        finally:
            os.unlink(path)


class TestPermRReadInvariant(unittest.TestCase):
    """Byte-29 PERM_R invariant (added 2026-07-18): a pattern with an
    explicit non-zero perms_byte lacking PERM_R must be excluded from
    retrieval in every filter mode. Tombstone + is_readable + truth_status
    are three orthogonal exclusion axes; all three must gate independently.
    """

    def test_perm_r_cleared_excludes_from_every_mode(self):
        # Pattern 0: PERM_W only (0x02) — explicitly non-readable.
        # Pattern 1: PERM_R + PERM_W (0x03) — normal readable.
        # Pattern 2: all-zero perms_byte — legacy default, treated as R+W.
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),
        ]
        perms = [0x02, 0x03, 0x00]  # W-only, R+W, legacy-default
        path = _make_cart_with_lifecycles(lifecycles, perms=perms)
        try:
            cart = CartHandle(path)
            for mode in ("active_only", "include_superseded", "all"):
                hits = retrieve_top_patterns(
                    cart, "cabbage", 10, pattern_filter=mode,
                )
                hit_idxs = {h.idx for h in hits}
                self.assertEqual(
                    hit_idxs,
                    {1, 2},
                    f"PERM_R=0 pattern leaked in mode {mode!r}",
                )
        finally:
            os.unlink(path)

    def test_perm_r_compose_with_truth_status(self):
        # Perm gate applies BEFORE the truth_status filter. A pattern that
        # would pass truth_status but not PERM_R stays hidden. A pattern
        # that would pass PERM_R but not truth_status also stays hidden.
        lifecycles = [
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),      # R+W ACTIVE
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_ACTIVE),      # W-only ACTIVE (should hide)
            pack_lifecycle_byte(truth_status=TRUTH_STATUS_SUPERSEDED),  # R+W SUPERSEDED (hides via active_only)
        ]
        perms = [0x03, 0x02, 0x03]
        path = _make_cart_with_lifecycles(lifecycles, perms=perms)
        try:
            cart = CartHandle(path)
            hits = retrieve_top_patterns(cart, "cabbage", 10, pattern_filter="active_only")
            self.assertEqual({h.idx for h in hits}, {0})
            hits = retrieve_top_patterns(cart, "cabbage", 10, pattern_filter="include_superseded")
            self.assertEqual({h.idx for h in hits}, {0, 2})
            hits = retrieve_top_patterns(cart, "cabbage", 10, pattern_filter="all")
            self.assertEqual({h.idx for h in hits}, {0, 2})  # #1 still gated by PERM_R
        finally:
            os.unlink(path)

    def test_legacy_perms_byte_zero_reads_as_readable(self):
        # Load-bearing backward-compat: every cart on disk today has
        # perms_byte == 0 for every pattern. All patterns must remain
        # readable.
        lifecycles = [0, 0, 0, 0]
        perms = [0, 0, 0, 0]
        path = _make_cart_with_lifecycles(lifecycles, perms=perms)
        try:
            cart = CartHandle(path)
            hits = retrieve_top_patterns(cart, "cabbage", 10, pattern_filter="active_only")
            self.assertEqual({h.idx for h in hits}, {0, 1, 2, 3})
            # is_readable direct check for good measure
            for i in range(4):
                self.assertTrue(cart.is_readable(i))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
