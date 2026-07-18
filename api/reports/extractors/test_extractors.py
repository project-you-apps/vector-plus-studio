"""Unit tests + fixture-driven smoke tests for the extractors module.

Run via pytest::

    python -m pytest api/reports/extractors/test_extractors.py -v

Or standalone::

    python api/reports/extractors/test_extractors.py

Fixtures cover the four supported date formats, four currency shapes, and
entity-mention behavior (aliases + case-insensitivity + sentence-aware
context).
"""
from __future__ import annotations

import datetime
import sys
import unittest
from decimal import Decimal

# Support both `python -m pytest ...` and direct execution. When run
# directly the module lives 3 levels below the repo root; add repo root
# to sys.path so the ``api.reports.extractors`` import resolves.
if __name__ == "__main__" and __package__ is None:
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from api.reports.extractors import (  # noqa: E402
    DateExtraction,
    MentionSpan,
    MoneyExtraction,
    discover_entities,
    extract_currency,
    extract_dates,
    extract_entity_mentions,
)


# ---------------------------------------------------------------------------
# Fixture strings
# ---------------------------------------------------------------------------

# Invoice filename with embedded date — canonical shape.
SYSCO_FILENAME = "752657234_20260517_034701258.pdf"

# Vendor line-item shape.
SYSCO_LINE = (
    "Invoice from Sysco Portland dated 2026-05-17. "
    "Total invoice: $234.56. Fuel surcharge: $8.95. "
    "State fee: $1.25 USD. Delivery date 05/24/2026. "
    "Refund adjustment: ($12.34) for damaged goods."
)

# Long-form date variants.
LONG_DATES = (
    "The audit began January 5, 2026 and concluded on 5 February 2026. "
    "A follow-up review is scheduled for Mar 3, 2026, per the schedule "
    "revision of March 3rd, 2026."
)

# Multi-mention entity fixture with aliases + varying casing.
ENTITY_TEXT = (
    "Sysco Portland delivered on Monday. The Sysco truck arrived at 6am. "
    "sysco-portland invoiced $234.56 for the shipment. "
    "Later that week SYSCO PORTLAND sent a fuel-surcharge adjustment. "
    "Compare with US Foods on the same route — US Foods was cheaper."
)


# ---------------------------------------------------------------------------
# Date tests
# ---------------------------------------------------------------------------

class TestExtractDates(unittest.TestCase):

    def test_iso_format(self):
        hits = extract_dates("Report filed 2026-05-17 at HQ.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 5, 17))
        self.assertEqual(hits[0].source_format, "iso")
        self.assertEqual(hits[0].confidence, 1.0)

    def test_us_slash_4digit_year(self):
        hits = extract_dates("Delivered 05/17/2026 in Portland.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 5, 17))
        self.assertEqual(hits[0].source_format, "us_slash")

    def test_us_slash_2digit_year(self):
        hits = extract_dates("Delivered 5/17/26 in Portland.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 5, 17))

    def test_us_slash_2digit_year_pivot_backward(self):
        # 2-digit year >= 70 should pivot into the 1900s (POSIX-style).
        hits = extract_dates("Filed on 1/1/85 originally.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(1985, 1, 1))

    def test_compact_yyyymmdd_sysco_style(self):
        hits = extract_dates(SYSCO_FILENAME)
        # Should find exactly the embedded 20260517.
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 5, 17))
        self.assertEqual(hits[0].source_format, "compact")

    def test_compact_ignores_bogus_year_window(self):
        # 30001231 would parse as a valid date if we let it through, but
        # the year gate (1900-2099) should skip it.
        hits = extract_dates("Junk digits 30001231 not a date.")
        self.assertEqual(hits, [])

    def test_long_form_mdy(self):
        hits = extract_dates("The audit began January 5, 2026.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 1, 5))
        self.assertEqual(hits[0].source_format, "long")

    def test_long_form_dmy(self):
        hits = extract_dates("Concluded on 5 February 2026.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 2, 5))
        self.assertEqual(hits[0].source_format, "long")

    def test_long_form_abbreviated(self):
        hits = extract_dates("Follow-up review is scheduled for Mar 3, 2026.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 3, 3))

    def test_long_form_ordinal_suffix(self):
        hits = extract_dates("Revision of March 3rd, 2026.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].date, datetime.date(2026, 3, 3))

    def test_sorted_by_offset(self):
        hits = extract_dates(LONG_DATES)
        self.assertGreaterEqual(len(hits), 3)
        # Sorted by start offset.
        self.assertEqual(hits, sorted(hits, key=lambda h: h.start))

    def test_invalid_date_rejected(self):
        # Month 13 shouldn't be admitted.
        hits = extract_dates("Bogus 2026-13-01 date")
        self.assertEqual(hits, [])

    def test_returns_empty_on_empty_input(self):
        self.assertEqual(extract_dates(""), [])
        self.assertEqual(extract_dates("no dates here"), [])

    def test_iso_beats_compact_when_overlapping(self):
        # ISO and compact could theoretically overlap on a stripped
        # form. We accept both distinct forms — this test just asserts
        # both formats work independently.
        text = "Old date: 20260517 vs new: 2026-05-17."
        hits = extract_dates(text)
        self.assertEqual(len(hits), 2)
        # Both resolve to the same calendar date.
        self.assertEqual({h.date for h in hits}, {datetime.date(2026, 5, 17)})


# ---------------------------------------------------------------------------
# Currency tests
# ---------------------------------------------------------------------------

class TestExtractCurrency(unittest.TestCase):

    def test_usd_prefix_basic(self):
        hits = extract_currency("Total $12.34 due.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("12.34"))
        self.assertEqual(hits[0].currency, "USD")
        self.assertFalse(hits[0].is_negative)

    def test_usd_prefix_with_thousands_separator(self):
        hits = extract_currency("Invoice total $1,234.56")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("1234.56"))
        # Uses Decimal (exact), so no FP surprises on the sum.
        self.assertNotEqual(hits[0].value, float("1234.56"))
        self.assertEqual(hits[0].value + Decimal("0.44"), Decimal("1235.00"))

    def test_usd_prefix_integer(self):
        hits = extract_currency("Charge $12 tomorrow.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("12"))

    def test_parenthesized_refund(self):
        hits = extract_currency("Refund ($12.34) posted.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("12.34"))
        self.assertTrue(hits[0].is_negative)

    def test_parenthesized_does_not_double_match_prefix(self):
        # ($12.34) should NOT produce both a paren hit AND a prefix hit.
        hits = extract_currency("Refund ($12.34) only please.")
        self.assertEqual(len(hits), 1)

    def test_trailing_code(self):
        hits = extract_currency("Amount 12.34 USD posted.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("12.34"))
        self.assertEqual(hits[0].currency, "USD")

    def test_trailing_code_variant_currencies(self):
        hits = extract_currency("Charged 45.00 EUR and 30.00 GBP")
        self.assertEqual(len(hits), 2)
        self.assertEqual({h.currency for h in hits}, {"EUR", "GBP"})

    def test_euro_prefix(self):
        hits = extract_currency("Total €99.00 including VAT.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].currency, "EUR")
        self.assertEqual(hits[0].value, Decimal("99.00"))

    def test_pound_prefix(self):
        hits = extract_currency("Total £120.50 including VAT.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].currency, "GBP")

    def test_context_hint_extraction(self):
        # Nearest word to the LEFT should win.
        hits = extract_currency("Fuel surcharge: $8.95 per delivery.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].context_hint, "surcharge")

    def test_preset_pattern_filter(self):
        text = (
            "Fuel surcharge $8.95 applied. Total $234.56 due. "
            "State fee $1.25 also assessed."
        )
        # Restrict to fuel_surcharge preset (matches "fuel" or "surcharge").
        hits = extract_currency(text, patterns=["fuel_surcharge"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("8.95"))
        # Restrict to invoice_total preset (matches "total" or "invoice").
        hits = extract_currency(text, patterns=["invoice_total"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].value, Decimal("234.56"))

    def test_sysco_line_all_extractions(self):
        hits = extract_currency(SYSCO_LINE)
        # 4 currency hits: total, fuel surcharge, state fee, refund.
        self.assertEqual(len(hits), 4)
        values = [h.value for h in hits]
        self.assertIn(Decimal("234.56"), values)
        self.assertIn(Decimal("8.95"), values)
        self.assertIn(Decimal("1.25"), values)
        self.assertIn(Decimal("12.34"), values)
        # The refund is the only negative.
        neg_hits = [h for h in hits if h.is_negative]
        self.assertEqual(len(neg_hits), 1)
        self.assertEqual(neg_hits[0].value, Decimal("12.34"))

    def test_empty_input(self):
        self.assertEqual(extract_currency(""), [])
        self.assertEqual(extract_currency("no money here"), [])


# ---------------------------------------------------------------------------
# Entity mention tests
# ---------------------------------------------------------------------------

class TestExtractEntityMentions(unittest.TestCase):

    def test_basic_case_insensitive(self):
        text = "Sysco Portland delivered. sysco portland invoiced."
        hits = extract_entity_mentions(text, "Sysco Portland")
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].entity_name, "Sysco Portland")
        # matched_text preserves original casing.
        self.assertEqual(hits[0].matched_text, "Sysco Portland")
        self.assertEqual(hits[1].matched_text, "sysco portland")

    def test_alias_matching(self):
        text = "The Sysco truck arrived. Later, SYSCO PORTLAND sent an invoice."
        hits = extract_entity_mentions(
            text,
            "Sysco Portland",
            aliases=["Sysco", "SYSCO PORTLAND", "sysco-portland"],
        )
        # 2 hits: "Sysco" and "SYSCO PORTLAND".
        self.assertEqual(len(hits), 2)
        # Longer alias ("SYSCO PORTLAND") beats shorter ("Sysco") on overlap.
        matched = [h.matched_text for h in hits]
        self.assertIn("Sysco", matched)
        self.assertIn("SYSCO PORTLAND", matched)

    def test_longest_match_wins_on_overlap(self):
        # "Sysco Portland" as canonical + "Sysco" as alias should NOT
        # produce a redundant "Sysco" hit inside "Sysco Portland".
        text = "Only Sysco Portland here."
        hits = extract_entity_mentions(
            text, "Sysco Portland", aliases=["Sysco"]
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].matched_text, "Sysco Portland")

    def test_context_sentence_aware(self):
        text = (
            "First sentence, no mention. "
            "Sysco Portland delivered on Monday. "
            "A third sentence follows."
        )
        hits = extract_entity_mentions(text, "Sysco Portland", context_chars=200)
        self.assertEqual(len(hits), 1)
        # Context should start after previous "." + trailing whitespace
        # and include the sentence terminator on the right.
        self.assertTrue(hits[0].context.startswith("Sysco Portland delivered"))
        self.assertTrue(hits[0].context.endswith("Monday."))

    def test_context_fallback_no_boundary_in_window(self):
        # Force a small window so no sentence boundary is in reach.
        text = "abc Sysco Portland xyz"
        hits = extract_entity_mentions(text, "Sysco Portland", context_chars=3)
        self.assertEqual(len(hits), 1)
        # Falls back to raw char slice; strip may remove leading spaces.
        self.assertIn("Sysco Portland", hits[0].context)

    def test_sysco_fixture_full(self):
        hits = extract_entity_mentions(
            ENTITY_TEXT,
            "Sysco Portland",
            aliases=["Sysco", "SYSCO PORTLAND", "sysco-portland"],
        )
        # Expected 4 mentions: "Sysco Portland", "Sysco" (in "Sysco truck"),
        # "sysco-portland", "SYSCO PORTLAND".
        self.assertEqual(len(hits), 4)
        # Sorted by offset.
        offsets = [h.start for h in hits]
        self.assertEqual(offsets, sorted(offsets))

    def test_no_match_returns_empty(self):
        hits = extract_entity_mentions("Just plain text.", "Sysco")
        self.assertEqual(hits, [])

    def test_empty_input_returns_empty(self):
        self.assertEqual(extract_entity_mentions("", "anything"), [])
        self.assertEqual(extract_entity_mentions("text", ""), [])


# ---------------------------------------------------------------------------
# Entity discovery tests (companion to extract_entity_mentions)
# ---------------------------------------------------------------------------

class TestDiscoverEntities(unittest.TestCase):

    def test_basic_discovery(self):
        text = (
            "Sysco Portland delivered on Monday. "
            "Franklin Foods invoiced separately."
        )
        out = discover_entities(text, min_length=3)
        # Sysco Portland + Franklin Foods surface; Monday filtered by
        # the built-in weekday stopword.
        self.assertIn("Sysco Portland", out)
        self.assertIn("Franklin Foods", out)
        self.assertNotIn("Monday", out)

    def test_dedup_case_insensitive_first_seen_wins(self):
        text = "Sysco Portland shipped. Later SYSCO PORTLAND invoiced."
        out = discover_entities(text, min_length=3)
        # Exactly one entry; first-seen surface form ("Sysco Portland") wins.
        self.assertEqual(out.count("Sysco Portland"), 1)
        self.assertNotIn("SYSCO PORTLAND", out)

    def test_first_seen_order_preserved(self):
        text = "Bravo shipped. Alpha arrived. Charlie followed."
        out = discover_entities(text, min_length=3)
        # Order preserved by first appearance in text.
        self.assertEqual(out, ["Bravo", "Alpha", "Charlie"])

    def test_stopword_first_token_rejected(self):
        text = "The Alpha team shipped. But Bravo failed. Our Charlie held."
        out = discover_entities(text, min_length=3)
        # "The Alpha team" / "But Bravo" / "Our Charlie" all get rejected
        # because the first token is a stopword.
        self.assertNotIn("The Alpha", out)
        self.assertNotIn("But Bravo", out)
        self.assertNotIn("Our Charlie", out)

    def test_extra_stopwords_merged(self):
        text = "Acme shipped. Widgets arrived."
        out = discover_entities(
            text, min_length=3, extra_stopwords=frozenset({"acme"}),
        )
        # Acme filtered by extra_stopwords; Widgets remains.
        self.assertNotIn("Acme", out)
        self.assertIn("Widgets", out)

    def test_min_length_gate(self):
        # min_length=3 rejects 2-char tokens (matches Coverage's
        # historical 3+ char behavior); min_length=2 admits them.
        text = "AI Group launched."
        out_strict = discover_entities(text, min_length=3)
        self.assertNotIn("AI Group", out_strict)
        # "Group" alone still surfaces under the strict gate.
        self.assertIn("Group", out_strict)
        # Default (min_length=2) admits the 2-char "AI".
        out_default = discover_entities(text)
        self.assertIn("AI Group", out_default)

    def test_empty_input_returns_empty(self):
        self.assertEqual(discover_entities(""), [])
        self.assertEqual(discover_entities("no proper nouns here."), [])

    def test_at_most_three_token_span(self):
        # Regex caps candidates at 3 tokens; a 4-token capitalized run
        # surfaces the first 3 as one entity and the 4th as its own.
        text = "Alpha Bravo Charlie Delta arrived."
        out = discover_entities(text, min_length=3)
        self.assertIn("Alpha Bravo Charlie", out)
        self.assertIn("Delta", out)


# ---------------------------------------------------------------------------
# Sysco integration — walks each extractor over the demo-cart-shaped
# fixture together so a regression in one shows up next to the others.
# ---------------------------------------------------------------------------

class TestSyscoDemoIntegration(unittest.TestCase):

    SYSCO_FULL = (
        "Source: 752657234_20260517_034701258.pdf\n"
        "Invoice from Sysco Portland dated 2026-05-17. "
        "Total invoice: $234.56. Fuel surcharge: $8.95. "
        # Second fuel-surcharge line keeps the label token adjacent to
        # the amount — 20-char context window relies on this. Real
        # trend-report code will fall back to per-passage grouping when
        # the label is separated by prose (that's a wave-2 concern).
        "Follow-up fuel surcharge: $10.00. "
        "State fee: $1.25 USD. Delivery date 05/24/2026. "
        "Refund adjustment: ($12.34) for damaged goods."
    )

    def test_dates_full_walk(self):
        hits = extract_dates(self.SYSCO_FULL)
        dates = {h.date for h in hits}
        # Filename compact + ISO in body (dedup collapses to the same date)
        # + one US-slash date in body.
        self.assertIn(datetime.date(2026, 5, 17), dates)
        self.assertIn(datetime.date(2026, 5, 24), dates)

    def test_currency_full_walk(self):
        hits = extract_currency(self.SYSCO_FULL)
        # 5 currency hits: total, fuel surcharge x2, state fee, refund.
        self.assertEqual(len(hits), 5)
        # The two fuel surcharges frame the Trend $8.95 → $10.00 story.
        fuel_hits = extract_currency(
            self.SYSCO_FULL, patterns=["fuel_surcharge"]
        )
        self.assertEqual(len(fuel_hits), 2)
        self.assertEqual(
            {h.value for h in fuel_hits}, {Decimal("8.95"), Decimal("10.00")}
        )

    def test_entity_full_walk(self):
        hits = extract_entity_mentions(
            self.SYSCO_FULL,
            "Sysco Portland",
            aliases=["Sysco"],
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].matched_text, "Sysco Portland")


# ---------------------------------------------------------------------------
# Standalone runner + demo-cart pretty-print
# ---------------------------------------------------------------------------

def _print_sample_outputs() -> None:
    """Pretty-print sample outputs on the Sysco fixtures.

    Runs when this file is executed directly; useful for the completion-
    report snapshot Andy asks for in the brief.
    """
    print("=" * 70)
    print("SAMPLE OUTPUTS — Sysco fixtures")
    print("=" * 70)

    print("\n[extract_dates] SYSCO_FILENAME:")
    for h in extract_dates(SYSCO_FILENAME):
        print(f"  {h.date} ({h.source_format}, conf={h.confidence}) "
              f"span={h.raw_span!r} [{h.start}:{h.end}]")

    print("\n[extract_dates] SYSCO_LINE:")
    for h in extract_dates(SYSCO_LINE):
        print(f"  {h.date} ({h.source_format}, conf={h.confidence}) "
              f"span={h.raw_span!r} [{h.start}:{h.end}]")

    print("\n[extract_currency] SYSCO_LINE (unfiltered):")
    for h in extract_currency(SYSCO_LINE):
        print(f"  {h.currency} {h.value} (neg={h.is_negative}) "
              f"hint={h.context_hint!r} span={h.raw_span!r}")

    print("\n[extract_currency] SYSCO_LINE (fuel_surcharge preset):")
    for h in extract_currency(SYSCO_LINE, patterns=["fuel_surcharge"]):
        print(f"  {h.currency} {h.value} hint={h.context_hint!r}")

    print("\n[extract_entity_mentions] ENTITY_TEXT (Sysco Portland + aliases):")
    for h in extract_entity_mentions(
        ENTITY_TEXT, "Sysco Portland",
        aliases=["Sysco", "SYSCO PORTLAND", "sysco-portland"],
    ):
        print(f"  matched={h.matched_text!r} at [{h.start}:{h.end}]")
        print(f"    context: {h.context!r}")


if __name__ == "__main__":
    # Print sample outputs first for the completion report...
    _print_sample_outputs()
    print("\n" + "=" * 70)
    print("RUNNING UNITTEST SUITE")
    print("=" * 70)
    # ...then run the test suite. Exit code propagates.
    unittest.main(argv=[sys.argv[0], "-v"], exit=True)
