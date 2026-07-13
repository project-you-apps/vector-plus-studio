"""Currency extraction from raw passage text.

Wave-1 uses pure regex for the four most-common invoice shapes:

- ``$`` prefix (``$12.34``, ``$1,234.56``, ``$12``)
- Parenthesized refund/negative notation (``($12.34)``)
- Trailing currency codes (``12.34 USD``)
- Named-symbol currencies (``€ £``) — limited wave 1 support

Uses :class:`decimal.Decimal` throughout so upstream rollups (Financial
Report §6, Trend Report §3) don't accumulate floating-point error
across thousands of invoice lines.

See ``Report Types Design 2026-07-10.md`` §0.3 and §3 (Trend fuel-
surcharge preset story) for the shared-extractor contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional


# ---------------------------------------------------------------------------
# MoneyExtraction — the record every currency hit produces
# ---------------------------------------------------------------------------

@dataclass
class MoneyExtraction:
    """A single extracted currency amount.

    ``value`` is always the positive magnitude; ``is_negative`` carries
    the sign so downstream aggregators can decide whether to net or to
    keep refunds separate (§6 edge case: "refunds / negatives (mark
    separately, don't net silently)").

    ``context_hint`` is the nearest word within 20 chars of the amount —
    "total", "surcharge", "tax", "fuel". Trend + Financial reports use
    it to route amounts to their canonical bucket without a second pass.
    """

    value: Decimal               # exact — use Decimal not float
    currency: str                # "USD" (default), "EUR", "GBP", ...
    raw_span: str
    start: int
    end: int
    is_negative: bool            # for parenthesized "( $12.34 )" refund notation
    confidence: float
    context_hint: Optional[str] = None  # nearby word — routes to trend/financial buckets


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Amount body — digits with optional thousands separators + optional
# decimals. Requires at least one integer digit. Kept as a raw fragment
# so the outer patterns can splice it in around currency markers.
_AMOUNT_BODY = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

# $12.34 / $1,234.56 / $12 — plain USD prefix. No negative sign here;
# use the parenthesized form for accounting-style refunds.
_USD_PREFIX_RE = re.compile(rf"\$\s*({_AMOUNT_BODY})")

# ($12.34) — parenthesized refund notation. Allows optional whitespace.
_USD_PAREN_RE = re.compile(rf"\(\s*\$\s*({_AMOUNT_BODY})\s*\)")

# 12.34 USD / 1,234.56 EUR — trailing currency-code notation.
# Restrict to a small allow-list so "10 EA" (each) doesn't false-match.
_TRAILING_CODE_RE = re.compile(
    rf"(?<![\w\d])({_AMOUNT_BODY})\s*(USD|EUR|GBP|CAD|AUD|JPY|CHF)\b"
)

# €12.34 / £1,234.56 — named-symbol currencies.
_EUR_PREFIX_RE = re.compile(rf"€\s*({_AMOUNT_BODY})")
_GBP_PREFIX_RE = re.compile(rf"£\s*({_AMOUNT_BODY})")

# Context-hint word matcher — walks backward from the match, then
# forward, grabbing the closest alphabetic word within 20 chars. Used
# to route hits to a bucket for Trend / Financial reports.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z_-]*")
_CONTEXT_WINDOW = 20

# Preset-name → normalized-hint-tokens map. When ``patterns=`` is
# supplied to :func:`extract_currency`, each preset name is compared
# against the extracted ``context_hint`` case-insensitively; a hit
# means the amount belongs to that bucket. This is intentionally
# lightweight in wave 1 — the canonical preset library ships in
# ``presets/trend_presets.yaml`` in wave 2 (§3 design doc).
_PRESET_TOKENS: dict[str, tuple[str, ...]] = {
    "fuel_surcharge": ("fuel", "surcharge"),
    "invoice_total": ("total", "invoice"),
    "state_fee": ("state", "fee"),
    "tax": ("tax", "sales"),
    "delivery": ("delivery",),
    "tip": ("tip", "gratuity"),
}


# ---------------------------------------------------------------------------
# extract_currency — public entry point
# ---------------------------------------------------------------------------

def extract_currency(
    text: str,
    patterns: Optional[list[str]] = None,
) -> list[MoneyExtraction]:
    """Extract currency amounts from text.

    Default patterns handle:

    - ``$`` prefix: ``$12.34``, ``$1,234.56``, ``$12``
    - Parenthesized refunds: ``($12.34)``
    - Trailing currency codes: ``12.34 USD``
    - Named currencies: ``€ £`` (limited wave 1 support)

    If ``patterns`` is provided, additionally filters to matches whose
    ``context_hint`` (nearby word within 20 chars) matches one of the
    preset names (see ``_PRESET_TOKENS``). E.g.
    ``patterns=["fuel_surcharge", "invoice_total"]`` restricts to those
    contexts. See ``presets/trend_presets.yaml`` in wave 2 for the
    canonical preset library.
    """
    if not text:
        return []

    hits: list[MoneyExtraction] = []

    # -- USD parenthesized (negative) --------------------------------------
    # Run BEFORE plain USD-prefix so the $ inside the parens doesn't get
    # double-matched by _USD_PREFIX_RE.
    for m in _USD_PAREN_RE.finditer(text):
        val = _parse_amount(m.group(1))
        if val is None:
            continue
        hits.append(MoneyExtraction(
            value=val, currency="USD",
            raw_span=m.group(0),
            start=m.start(), end=m.end(),
            is_negative=True,
            confidence=1.0,
            context_hint=_extract_context_hint(text, m.start(), m.end()),
        ))

    # -- USD prefix --------------------------------------------------------
    # Skip any that overlap with an already-accepted paren hit (the paren
    # matcher already covered them).
    paren_ranges = [(h.start, h.end) for h in hits]
    for m in _USD_PREFIX_RE.finditer(text):
        if _overlaps(m.start(), m.end(), paren_ranges):
            continue
        val = _parse_amount(m.group(1))
        if val is None:
            continue
        hits.append(MoneyExtraction(
            value=val, currency="USD",
            raw_span=m.group(0),
            start=m.start(), end=m.end(),
            is_negative=False,
            confidence=1.0,
            context_hint=_extract_context_hint(text, m.start(), m.end()),
        ))

    # -- Trailing currency code ("12.34 USD") ------------------------------
    prior_ranges = [(h.start, h.end) for h in hits]
    for m in _TRAILING_CODE_RE.finditer(text):
        if _overlaps(m.start(), m.end(), prior_ranges):
            continue
        val = _parse_amount(m.group(1))
        if val is None:
            continue
        hits.append(MoneyExtraction(
            value=val, currency=m.group(2).upper(),
            raw_span=m.group(0),
            start=m.start(), end=m.end(),
            is_negative=False,
            confidence=0.95,
            context_hint=_extract_context_hint(text, m.start(), m.end()),
        ))

    # -- Named-symbol currencies ------------------------------------------
    prior_ranges = [(h.start, h.end) for h in hits]
    for m in _EUR_PREFIX_RE.finditer(text):
        if _overlaps(m.start(), m.end(), prior_ranges):
            continue
        val = _parse_amount(m.group(1))
        if val is None:
            continue
        hits.append(MoneyExtraction(
            value=val, currency="EUR",
            raw_span=m.group(0),
            start=m.start(), end=m.end(),
            is_negative=False,
            confidence=0.9,
            context_hint=_extract_context_hint(text, m.start(), m.end()),
        ))
    prior_ranges = [(h.start, h.end) for h in hits]
    for m in _GBP_PREFIX_RE.finditer(text):
        if _overlaps(m.start(), m.end(), prior_ranges):
            continue
        val = _parse_amount(m.group(1))
        if val is None:
            continue
        hits.append(MoneyExtraction(
            value=val, currency="GBP",
            raw_span=m.group(0),
            start=m.start(), end=m.end(),
            is_negative=False,
            confidence=0.9,
            context_hint=_extract_context_hint(text, m.start(), m.end()),
        ))

    # -- Apply preset filter ----------------------------------------------
    if patterns:
        wanted_tokens: set[str] = set()
        # Unknown preset names still get their own name as a token —
        # so passing patterns=["fuel"] filters on the literal word.
        for name in patterns:
            tokens = _PRESET_TOKENS.get(name.lower())
            if tokens:
                wanted_tokens.update(tokens)
            else:
                wanted_tokens.add(name.lower())
        hits = [h for h in hits if _hint_matches(h.context_hint, wanted_tokens)]

    # -- Sort by offset for stable downstream consumption -----------------
    hits.sort(key=lambda h: h.start)

    # TODO(wave-2): LLM-fallback for ambiguous currency shapes ("twelve
    # ninety-five", "USD twelve fifty", handwritten "8.95" without a
    # dollar sign in a context clearly about money). Design in §C.1 —
    # plug an optional callable that receives (text, prior_hits) and
    # returns additional MoneyExtraction records with confidence < 1.0.

    return hits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_amount(raw: str) -> Optional[Decimal]:
    """Strip commas, coerce to Decimal, or return None on failure."""
    clean = raw.replace(",", "").strip()
    if not clean:
        return None
    try:
        return Decimal(clean)
    except (InvalidOperation, ValueError):
        return None


def _extract_context_hint(text: str, start: int, end: int) -> Optional[str]:
    """Nearest alphabetic word within ``_CONTEXT_WINDOW`` chars of the match.

    Priority: nearest word overall, preferring the one to the LEFT (that's
    where invoice descriptors like "fuel surcharge $8.95" put the label).
    Falls back to the right-hand side if nothing on the left. Lowercased.
    """
    left_start = max(0, start - _CONTEXT_WINDOW)
    left_slice = text[left_start:start]
    right_slice = text[end:end + _CONTEXT_WINDOW]

    # Scan left-side words in reverse order so the closest wins.
    left_words = list(_WORD_RE.finditer(left_slice))
    if left_words:
        return left_words[-1].group(0).lower()

    right_words = _WORD_RE.search(right_slice)
    if right_words:
        return right_words.group(0).lower()
    return None


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    """True iff ``[start, end)`` intersects any range in ``ranges``."""
    for s, e in ranges:
        if start < e and end > s:
            return True
    return False


def _hint_matches(hint: Optional[str], wanted: set[str]) -> bool:
    """True iff any wanted token appears in / equals the hint."""
    if not hint:
        return False
    h = hint.lower()
    for w in wanted:
        if w in h or h in w:
            return True
    return False


__all__ = ["MoneyExtraction", "extract_currency"]
