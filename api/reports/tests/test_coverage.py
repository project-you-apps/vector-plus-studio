"""Coverage Report smoke + section-level tests.

Run via pytest::

    python -m pytest api/reports/tests/test_coverage.py -v

Or standalone::

    python api/reports/tests/test_coverage.py

Validates the four requirements the design brief called out:

1. Empty cart doesn't crash — all six section headers render with a
   "No items found" placeholder.
2. A cart with items but no extracted dates produces a graceful Time
   Gaps section (not a crash).
3. Tombstoned items are skipped from every section.
4. Output is valid markdown with balanced fenced blocks.

A synthetic cart is materialized on disk via ``numpy.savez`` so we
exercise the same ``CartHandle`` code path as production. We do NOT
touch the real ``cartridges/`` directory — everything lands in a
tempdir the test cleans up after itself.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Optional

import numpy as np

# Support both `python -m pytest ...` and direct execution.
if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from api.reports.base import ReportInput, ReportOptions  # noqa: E402
from api.reports.coverage import CoverageReport  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic cart builder
# ---------------------------------------------------------------------------

def _hippo_row(tombstone: bool = False) -> np.ndarray:
    """Return a fresh 64-byte hippocampus row. Byte 28 carries the flags."""
    row = np.zeros(64, dtype=np.uint8)
    if tombstone:
        row[28] = 0x01
    return row


def _make_cart(
    passages: list[str],
    sources: list[str],
    tombstones: Optional[list[bool]] = None,
    dim: int = 16,
    cart_name: str = "synthetic-coverage-test",
) -> str:
    """Serialize a synthetic cart to a tempfile; return its path.

    ``passages[i]`` becomes the text for pattern i; ``sources[i]`` its
    source_path label; ``tombstones[i]`` (if given) marks that row's
    hippocampus flag byte. Embeddings are random but shape-correct so
    ``CartHandle`` opens successfully. The caller owns the returned
    path and should ``os.unlink`` it when finished.
    """
    n = len(passages)
    assert len(sources) == n
    tombstones = tombstones or [False] * n
    assert len(tombstones) == n

    embeddings = np.random.default_rng(42).standard_normal(
        size=(n, dim), dtype=np.float32
    ) if n else np.zeros((0, dim), dtype=np.float32)

    passages_arr = np.array(passages, dtype=object)
    sources_arr = np.array(sources, dtype=object)
    hippo = (
        np.stack([_hippo_row(t) for t in tombstones])
        if n
        else np.zeros((0, 64), dtype=np.uint8)
    )

    # Pattern-0 as JSON string — the shape CartHandle recognizes.
    pattern0_payload = json.dumps({
        "cart_name": cart_name,
        "description": "synthetic cart for CoverageReport tests",
    })
    pattern0 = np.array(pattern0_payload, dtype=object)

    fd, path = tempfile.mkstemp(suffix=".cart.npz", prefix="coverage_test_")
    os.close(fd)
    np.savez(
        path,
        embeddings=embeddings,
        passages=passages_arr,
        source_paths=sources_arr,
        hippocampus=hippo,
        pattern0=pattern0,
    )
    return path


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

class TestCoverageEmptyCart(unittest.TestCase):
    """A cart with zero patterns should render all six section headers
    and not raise."""

    def test_empty_cart_renders_all_sections(self):
        path = _make_cart(passages=[], sources=[])
        try:
            report = CoverageReport()
            out = report.generate(
                path, ReportInput(raw={}), ReportOptions()
            )
        finally:
            os.unlink(path)

        self.assertIn("# Coverage Report", out.markdown)
        for section in (
            "## Coverage Overview",
            "## Underrepresented Themes",
            "## Orphan Entities",
            "## Source Coverage",
            "## Time Gaps",
            "## Context-Poor Items",
        ):
            self.assertIn(section, out.markdown)
        # Metadata flag confirms the empty path was taken.
        self.assertTrue(out.metadata.get("empty"))


class TestCoverageNoDates(unittest.TestCase):
    """A cart with items but no extractable dates should produce a valid
    Time Gaps section (skipped-gracefully, not a crash)."""

    def test_no_dates_skips_time_gaps_cleanly(self):
        passages = [
            "The Sysco Portland delivery arrived on schedule.",
            "Fuel surcharge on delivery: $8.95.",
            "Empty passage below has no context.",
            "",
        ]
        sources = ["invoice.pdf", "invoice.pdf", "unknown.pdf", "empty.pdf"]
        path = _make_cart(passages=passages, sources=sources)
        try:
            report = CoverageReport()
            out = report.generate(
                path, ReportInput(raw={}), ReportOptions()
            )
        finally:
            os.unlink(path)

        # Time Gaps section should still render, and it should not have
        # crashed the report.
        self.assertIn("## Time Gaps", out.markdown)
        # Body says "no dates" — we assert on the human-readable copy
        # rather than a specific sentence so a copy tweak doesn't break
        # the test.
        gaps_section = out.markdown.split("## Time Gaps", 1)[1]
        gaps_body_lower = gaps_section.split("##", 1)[0].lower()
        self.assertTrue(
            "no dates" in gaps_body_lower
            or "one distinct date" in gaps_body_lower,
            f"Expected 'no dates' language in Time Gaps section, got: "
            f"{gaps_section[:200]!r}",
        )


class TestCoverageSkipsTombstones(unittest.TestCase):
    """Tombstoned items must not appear in any section, and the
    warnings list must surface a skip count."""

    def test_tombstoned_items_excluded(self):
        passages = [
            "Sysco Portland invoice dated 2026-05-17. Total: $234.56.",
            "Sysco Portland invoice dated 2026-06-01. Total: $189.20.",
            "Deleted-should-not-appear content Nintendo Switch $99.",
        ]
        sources = ["sysco.pdf", "sysco.pdf", "deleted.pdf"]
        tombstones = [False, False, True]
        path = _make_cart(
            passages=passages, sources=sources, tombstones=tombstones,
        )
        try:
            report = CoverageReport()
            out = report.generate(
                path, ReportInput(raw={}), ReportOptions()
            )
        finally:
            os.unlink(path)

        # The tombstoned passage's unique tokens must NOT appear anywhere
        # in the rendered report — that would prove the skip failed.
        # (Note: "deleted.pdf" — as of Phase A source-file links, source
        # names ship as vps://source/{slug} links in the markdown; the
        # display text "deleted.pdf" is what we're screening for either
        # way.)
        self.assertNotIn("Nintendo", out.markdown)
        self.assertNotIn("deleted.pdf", out.markdown)
        # Belt-and-suspenders: the link form also shouldn't sneak in.
        self.assertNotIn("vps://source/deleted-pdf", out.markdown)
        # Warnings should surface the skip count.
        self.assertTrue(
            any("tombstoned" in w.lower() for w in out.warnings),
            f"Expected tombstone warning in {out.warnings!r}",
        )
        # Metadata should record the tombstone count.
        self.assertEqual(out.metadata.get("tombstoned_skipped"), 1)
        self.assertEqual(out.metadata.get("live_count"), 2)


class TestCoverageMarkdownBalance(unittest.TestCase):
    """Rendered markdown must be well-formed: no unclosed code blocks."""

    def test_no_unclosed_fenced_blocks(self):
        passages = [
            # Rich passage — enables every section to render non-empty.
            "Sysco Portland invoice dated 2026-05-17. Total: $234.56.",
            "US Foods delivery scheduled 2026-06-05 with fuel surcharge.",
            "Random one-off Acme Widgets appears only here.",
        ]
        sources = ["invoice-may.pdf", "invoice-june.pdf", "misc.pdf"]
        path = _make_cart(passages=passages, sources=sources)
        try:
            report = CoverageReport()
            out = report.generate(
                path, ReportInput(raw={}), ReportOptions()
            )
        finally:
            os.unlink(path)

        # Odd number of ``` fences would indicate an unclosed code block.
        fence_count = out.markdown.count("```")
        self.assertEqual(
            fence_count % 2,
            0,
            f"Unbalanced fenced blocks — {fence_count} fences found",
        )
        # Sanity: report actually produced structured content.
        self.assertIn("# Coverage Report", out.markdown)


class TestCoverageEndToEnd(unittest.TestCase):
    """Full-cart run that exercises all six sections with content flowing
    through — the synthetic cart mimics a small invoice / notes mix."""

    def test_full_synthetic_cart(self):
        passages = [
            # 2 invoice-shaped passages sharing a source (Sysco theme).
            "Sysco Portland invoice dated 2026-01-15. Total $234.56.",
            "Sysco Portland delivery 2026-02-10 with fuel surcharge $8.95.",
            # 2 more Sysco items to make "sysco portland" a well-represented theme.
            "Sysco Portland order 2026-02-20 was on time.",
            "Sysco Portland followup 2026-03-01 email received.",
            # US Foods — deliberately just 2 items (below the default min_theme=3
            # would surface if it's in the top themes; borderline).
            "US Foods delivery 2026-03-05. Amount $180.00.",
            "US Foods delivery 2026-03-12. Amount $195.20.",
            # A source with only ONE item — should be under-utilized when
            # source_coverage_min defaults to 5.
            "Acme Widgets receipt 2026-04-01 for $12.00.",
            # A big time gap: nothing between April and December.
            "Grant end-of-year review 2026-12-15 tallied results.",
            # Context-poor item: no date, no entity, no currency.
            "some random uncategorized text with nothing extractable",
        ]
        sources = [
            "sysco.pdf",
            "sysco.pdf",
            "sysco.pdf",
            "sysco.pdf",
            "usfoods.pdf",
            "usfoods.pdf",
            "acme.pdf",
            "grant.pdf",
            "notes.pdf",
        ]
        path = _make_cart(passages=passages, sources=sources)
        try:
            report = CoverageReport()
            out = report.generate(
                path, ReportInput(raw={}), ReportOptions()
            )
        finally:
            os.unlink(path)

        md = out.markdown

        # All 6 sections rendered.
        for section in (
            "## Coverage Overview",
            "## Underrepresented Themes",
            "## Orphan Entities",
            "## Source Coverage",
            "## Time Gaps",
            "## Context-Poor Items",
        ):
            self.assertIn(section, md)

        # Time gap Apr → Dec is > 30 days — must surface.
        self.assertIn("## Time Gaps", md)
        gaps_section = md.split("## Time Gaps", 1)[1].split("##", 1)[0]
        # Expect the December date to appear somewhere in the gaps.
        self.assertIn("2026-12-15", gaps_section)

        # Source coverage table exists and Acme (1 item) is flagged.
        # (Phase A — source-file name is emitted as a markdown link;
        # the raw display text is still present.)
        self.assertIn("acme.pdf", md)
        self.assertIn("vps://source/acme-pdf", md)
        self.assertIn("under-utilized", md.lower())

        # Context-poor section should surface the "random uncategorized"
        # passage — it's the one item without date/entity/currency.
        self.assertIn("Context-Poor Items", md)
        self.assertGreaterEqual(out.metadata["context_poor"]["total"], 1)

        # Overview % strings should render (with_dates non-zero).
        self.assertIn("dates", md.lower())
        self.assertIn("entities", md.lower())
        self.assertIn("currency", md.lower())


if __name__ == "__main__":
    unittest.main()
