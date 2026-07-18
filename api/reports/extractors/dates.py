"""Date extraction from raw passage text.

Uses pure regex — four formats cover the target universe
(ISO, US-slash, long-name, and Sysco-style ``YYYYMMDD`` compact
embeddings from invoice filenames). a future hook noted at the bottom
for LLM fallback ("yesterday", "last Tuesday", etc).

See ``Report Types Design 2026-07-10.md`` §0.3 for the shared-extractor
contract. Timeline (§2), Trend (§3), and Financial Rollup (§6) all
consume this.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# DateExtraction — the record every extractor hit produces
# ---------------------------------------------------------------------------

@dataclass
class DateExtraction:
    """A single extracted date span.

    ``confidence`` is 1.0 for regex hits across the board; the
    field is here so a future release LLM-fallback hits can carry lower scores
    without a shape change. ``source_format`` records which regex bucket
    fired so consumers (Trend flag thresholds, Timeline granularity
    fallback) can reason about ambiguity.
    """

    date: datetime.date          # normalized calendar date
    raw_span: str                # original substring matched
    start: int                   # char offset in source text (inclusive)
    end: int                     # char offset in source text (exclusive)
    confidence: float            # 0.0-1.0; regex hits = 1.0 for wave 1
    source_format: str           # "iso" | "us_slash" | "long" | "compact" | "unknown"


# ---------------------------------------------------------------------------
# Regex patterns — one per source format
# ---------------------------------------------------------------------------

# ISO YYYY-MM-DD — unambiguous, highest priority. Bounded by non-digits
# to avoid nibbling into YYYYMMDD hits.
_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")

# US-slash M/D/Y — accepts 1-2 digit month + day, 2 or 4 digit year.
# Bounded so we don't chew into version strings like "1/2/3/4".
_US_SLASH_RE = re.compile(
    r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})(?!\d)"
)

# Compact YYYYMMDD — 8 consecutive digits shaped like a plausible date.
# Bounded (not preceded / followed by another digit) so a longer numeric
# run doesn't false-hit. This is the Sysco invoice-filename embedding:
# ``752657234_20260517_034701258.pdf`` → 20260517.
_COMPACT_RE = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")

# Long form — three flavors:
#   1. "January 5, 2026" / "Jan 5, 2026"
#   2. "January 5 2026"  / "Jan 5 2026"    (no comma)
#   3. "5 January 2026"  / "5 Jan 2026"    (day-first)
_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
# Month D(,)? YYYY
_LONG_MDY_RE = re.compile(
    rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
# D Month YYYY
_LONG_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\s+(\d{{4}})\b",
    re.IGNORECASE,
)

_MONTH_INDEX = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ---------------------------------------------------------------------------
# extract_dates — public entry point
# ---------------------------------------------------------------------------

def extract_dates(text: str) -> list[DateExtraction]:
    """Extract all datable spans from raw text via regex.

    Formats supported (wave 1):

    - ISO: ``YYYY-MM-DD`` (unambiguous — highest confidence)
    - US slash: ``MM/DD/YYYY``, ``M/D/YY``, ``M/D/YYYY``
    - Long: "January 5, 2026" / "Jan 5 2026" / "5 January 2026"
    - Compact: ``YYYYMMDD`` (Sysco-style invoice-number embedding,
      e.g. ``20260517``)

    Returns non-overlapping matches sorted by ``start`` offset. When two
    regexes overlap (e.g. compact + slash both fire on the same span),
    the highest-confidence hit wins; ties break on longer span, then on
    earlier ``source_format`` priority.
    """
    if not text:
        return []

    hits: list[DateExtraction] = []

    # -- ISO ---------------------------------------------------------------
    for m in _ISO_RE.finditer(text):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d is not None:
            hits.append(DateExtraction(
                date=d, raw_span=m.group(0),
                start=m.start(), end=m.end(),
                confidence=1.0, source_format="iso",
            ))

    # -- Compact YYYYMMDD --------------------------------------------------
    # Only accept plausible year windows so "20261234" (junk) doesn't
    # mint a bogus date. 1900-2099 covers everything a reports pipeline
    # would sanely encounter.
    for m in _COMPACT_RE.finditer(text):
        y, mo, d_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1900 <= y <= 2099):
            continue
        d = _safe_date(y, mo, d_)
        if d is not None:
            hits.append(DateExtraction(
                date=d, raw_span=m.group(0),
                start=m.start(), end=m.end(),
                # Compact is unambiguous once the year gate passes; still
                # very slightly lower than ISO because format is inferred
                # from position rather than a delimiter.
                confidence=0.95, source_format="compact",
            ))

    # -- US slash ----------------------------------------------------------
    # US convention (Andy's demo cart is all US invoices). If we hit
    # non-US carts later we can add a locale hint.
    for m in _US_SLASH_RE.finditer(text):
        mo, d_, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            # 2-digit year: pivot on 70 (POSIX convention). "26" → 2026.
            y = 2000 + y if y < 70 else 1900 + y
        d = _safe_date(y, mo, d_)
        if d is not None:
            hits.append(DateExtraction(
                date=d, raw_span=m.group(0),
                start=m.start(), end=m.end(),
                # US-slash is technically ambiguous with DD/MM (European);
                # slightly lower confidence than ISO.
                confidence=0.9, source_format="us_slash",
            ))

    # -- Long-form (Month D, Y) -------------------------------------------
    for m in _LONG_MDY_RE.finditer(text):
        mo = _MONTH_INDEX.get(m.group(1).lower())
        if mo is None:
            continue
        d = _safe_date(int(m.group(3)), mo, int(m.group(2)))
        if d is not None:
            hits.append(DateExtraction(
                date=d, raw_span=m.group(0),
                start=m.start(), end=m.end(),
                confidence=0.95, source_format="long",
            ))

    # -- Long-form (D Month Y) --------------------------------------------
    for m in _LONG_DMY_RE.finditer(text):
        mo = _MONTH_INDEX.get(m.group(2).lower())
        if mo is None:
            continue
        d = _safe_date(int(m.group(3)), mo, int(m.group(1)))
        if d is not None:
            hits.append(DateExtraction(
                date=d, raw_span=m.group(0),
                start=m.start(), end=m.end(),
                confidence=0.95, source_format="long",
            ))

    # -- Resolve overlaps --------------------------------------------------
    # Sort by start offset, then descending confidence so the "winning"
    # hit at each overlap comes first in the tie-break.
    hits.sort(key=lambda h: (h.start, -h.confidence, -(h.end - h.start)))

    winners: list[DateExtraction] = []
    for h in hits:
        # Discard any hit that overlaps a previously accepted (higher-
        # priority) winner. Since we sorted by start ascending and
        # winners is start-ascending, the tail is enough.
        if winners and h.start < winners[-1].end:
            continue
        winners.append(h)

    # Ensure final list is sorted by start.
    winners.sort(key=lambda h: h.start)

    # TODO(a future release): LLM-fallback for relative dates ("yesterday", "last
    # Tuesday", "next Q3"). Plug an
    # optional callable that receives (text, anchor_date) and returns
    # additional DateExtraction records with confidence < 1.0.

    return winners


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_date(year: int, month: int, day: int) -> Optional[datetime.date]:
    """Return a ``datetime.date`` or ``None`` if the triple is invalid.

    Cheap wrapper around the constructor — the regex layer already
    admits some out-of-range triples (e.g. month=13 from US-slash) and
    it's simpler to let the constructor reject than to sanity-gate
    upstream.
    """
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


__all__ = ["DateExtraction", "extract_dates"]
