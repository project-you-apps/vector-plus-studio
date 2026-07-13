"""Unit tests for the source_link helper.

The slug rule + link syntax is load-bearing for the frontend's Phase A
drill-down (custom ``a`` renderer in ``ReportResultsView.tsx`` intercepts
``vps://source/{slug}`` and dispatches a Search-tab focus). If either
side of the contract drifts, clicks silently no-op — hence the specific
assertions here on the exact slug + link shape.

Run standalone::

    python api/reports/tests/test_source_link.py
"""
from __future__ import annotations

import os
import sys
import unittest

# Support both `python -m pytest ...` and direct execution.
if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from api.reports.source_link import source_link, source_slug  # noqa: E402


class TestSourceSlug(unittest.TestCase):
    """Slug rule tests — one per rule bullet in source_link.py's docstring."""

    def test_lowercase_and_dashify(self):
        self.assertEqual(
            source_slug("War Poems by Siegfried Sassoon"),
            "war-poems-by-siegfried-sassoon",
        )

    def test_strips_poem_prefix(self):
        self.assertEqual(
            source_slug("Poem: War Poems by Siegfried Sassoon"),
            "war-poems-by-siegfried-sassoon",
        )

    def test_strips_doc_prefix_case_insensitive(self):
        self.assertEqual(source_slug("doc: Meeting Notes 2026-05"), "meeting-notes-2026-05")

    def test_strips_pdf_prefix(self):
        self.assertEqual(source_slug("PDF: 2026 Annual Report v2"), "2026-annual-report-v2")

    def test_pdf_extension_survives(self):
        # Extension is NOT a stripped prefix, it's just alphanumeric that
        # gets slugified. "invoice-may04.pdf" → "invoice-may04-pdf".
        self.assertEqual(source_slug("invoice-may04.pdf"), "invoice-may04-pdf")

    def test_collapses_consecutive_dashes(self):
        self.assertEqual(source_slug("foo   bar___baz"), "foo-bar-baz")

    def test_trims_edges(self):
        self.assertEqual(source_slug("!!! whatever !!!"), "whatever")

    def test_empty_input(self):
        self.assertEqual(source_slug(""), "")
        self.assertEqual(source_slug(None), "")
        self.assertEqual(source_slug("   "), "")

    def test_all_punctuation_returns_empty(self):
        self.assertEqual(source_slug("!!!"), "")


class TestSourceLink(unittest.TestCase):
    def test_basic_link(self):
        self.assertEqual(
            source_link("War Poems by Siegfried Sassoon"),
            "[War Poems by Siegfried Sassoon](vps://source/war-poems-by-siegfried-sassoon)",
        )

    def test_link_preserves_prefix_in_display(self):
        # Display text keeps the "Poem:" prefix — only the slug strips
        # it — so the rendered surface reads the same as before Phase A.
        self.assertEqual(
            source_link("Poem: War Poems by Siegfried Sassoon"),
            "[Poem: War Poems by Siegfried Sassoon](vps://source/war-poems-by-siegfried-sassoon)",
        )

    def test_empty_falls_back(self):
        self.assertEqual(source_link(""), "(no source)")
        self.assertEqual(source_link(None), "(no source)")
        # Custom empty_label so callers can preserve their prior copy.
        self.assertEqual(source_link("", empty_label="(unknown source)"), "(unknown source)")

    def test_escapes_display_pipe_for_gfm_tables(self):
        # Pipes in the display text would prematurely close a GFM table
        # column. Backslash-escape so the table survives.
        got = source_link("foo|bar")
        self.assertIn("foo\\|bar", got)
        self.assertIn("(vps://source/foo-bar)", got)

    def test_escapes_display_close_bracket(self):
        # Close-brackets in display text would prematurely close the
        # markdown link.
        got = source_link("foo]bar")
        self.assertIn("foo\\]bar", got)

    def test_all_punctuation_source_no_link(self):
        # If the slug reduces to empty, we don't emit a live-looking link
        # that points at vps://source/. Display text (escaped) survives.
        self.assertEqual(source_link("!!!"), "!!!")


if __name__ == "__main__":
    unittest.main()
