"""Entity-mention extraction from raw passage text.

Case-insensitive matching with alias support. Context extraction is
sentence-boundary aware — reports (Entity Rollup §5 in particular) want
the mention plus surrounding context that reads like a sentence, not a
sliced-mid-word char window.

See ``Report Types Design 2026-07-10.md`` §0.3 for the shared-extractor
contract and §5 (Entity Rollup) for the primary consumer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# MentionSpan — the record every entity match produces
# ---------------------------------------------------------------------------

@dataclass
class MentionSpan:
    """A single entity-name mention in text.

    ``entity_name`` is the canonical name the caller searched for;
    ``matched_text`` is what actually appeared in the source (may
    differ by casing or match one of the ``aliases``). Entity Rollup
    keys on ``entity_name`` for grouping while displaying ``matched_text``
    to preserve source flavor.

    ``context`` is expanded to the nearest sentence boundary within
    ``context_chars`` on each side of the match; falls back to a plain
    char slice if no boundary is found in range.
    """

    entity_name: str             # canonical name the caller searched for
    matched_text: str            # what actually matched (surface form)
    start: int                   # char offset in source text (inclusive)
    end: int                     # char offset in source text (exclusive)
    context: str                 # sentence-aware ±context_chars around match


# ---------------------------------------------------------------------------
# Sentence boundary characters — used for context expansion.
# ---------------------------------------------------------------------------

# Matches any single sentence terminator OR a newline. We look for the
# NEAREST-to-the-match boundary within the expansion window, so the
# regex flavor is simple: character class scanned by hand, not a
# compiled pattern (a single-char search is faster than re.finditer over
# short slices).
_SENTENCE_BOUNDARIES = frozenset(".!?\n")


# ---------------------------------------------------------------------------
# extract_entity_mentions — public entry point
# ---------------------------------------------------------------------------

def extract_entity_mentions(
    text: str,
    entity_name: str,
    aliases: Optional[list[str]] = None,
    context_chars: int = 100,
) -> list[MentionSpan]:
    """Find all mentions of an entity in text.

    Case-insensitive. ``aliases`` are additional strings that also count
    as matches for the same entity (e.g. ``entity_name="Sysco Portland"``,
    ``aliases=["Sysco", "SYSCO PORTLAND", "sysco-portland"]``).

    Context extraction tries to expand out to the nearest sentence
    boundary (``. ! ?`` or newline) within ``context_chars`` of the
    match on each side. Falls back to raw char slice if no boundary
    found in range.

    Returns matches sorted by ``start`` offset. Overlapping matches
    (e.g. "Sysco Portland" and "Sysco" would both fire on "Sysco
    Portland Inc") are deduplicated: longer surface form wins so the
    caller sees the most-specific mention.
    """
    if not text or not entity_name:
        return []

    # Build the search-term list: canonical + aliases, all lower-cased
    # for case-insensitive matching. Preserve the ORIGINAL cased values
    # so we can report the raw surface form back in ``matched_text``.
    # Longer terms first so "Sysco Portland" wins over "Sysco" in the
    # dedup step below.
    search_terms: list[str] = [entity_name]
    if aliases:
        search_terms.extend(a for a in aliases if a)
    # De-dup terms case-insensitively, longest first.
    seen: set[str] = set()
    ordered: list[str] = []
    for t in sorted(search_terms, key=len, reverse=True):
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(t)

    hits: list[MentionSpan] = []
    lowered = text.lower()

    for term in ordered:
        term_lower = term.lower()
        term_len = len(term_lower)
        if term_len == 0:
            continue
        start = 0
        while True:
            idx = lowered.find(term_lower, start)
            if idx < 0:
                break
            end = idx + term_len
            # Skip if this hit overlaps a previously accepted, longer
            # match (order-of-iteration means longer terms come first).
            if _overlaps(idx, end, hits):
                start = end
                continue
            matched_text = text[idx:end]
            context = _sentence_aware_context(text, idx, end, context_chars)
            hits.append(MentionSpan(
                entity_name=entity_name,
                matched_text=matched_text,
                start=idx,
                end=end,
                context=context,
            ))
            start = end

    hits.sort(key=lambda h: h.start)
    return hits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlaps(start: int, end: int, hits: list[MentionSpan]) -> bool:
    """True iff the range intersects any accepted hit."""
    for h in hits:
        if start < h.end and end > h.start:
            return True
    return False


def _sentence_aware_context(
    text: str,
    start: int,
    end: int,
    context_chars: int,
) -> str:
    """Return context around ``[start, end)`` expanded to sentence boundaries.

    Walks left from ``start`` looking for the NEAREST sentence terminator
    within ``context_chars``; steps one position past it so the boundary
    char itself doesn't leak into the context. Walks right similarly.
    Falls back to a raw ``[start - context_chars, end + context_chars)``
    slice if no boundary is present in the window.
    """
    n = len(text)
    left_min = max(0, start - context_chars)
    right_max = min(n, end + context_chars)

    # Left expansion: search backward from start-1 through left_min for
    # the closest boundary. If found, the context begins AFTER the
    # boundary (drop the terminator + trailing whitespace).
    left_bound = left_min
    for i in range(start - 1, left_min - 1, -1):
        if text[i] in _SENTENCE_BOUNDARIES:
            left_bound = i + 1
            # Skip whitespace immediately after the terminator so the
            # context starts on the next sentence word.
            while left_bound < start and text[left_bound].isspace():
                left_bound += 1
            break

    # Right expansion: search forward from end through right_max for
    # the closest boundary. If found, INCLUDE the terminator in the
    # context (readers expect complete sentences).
    right_bound = right_max
    for i in range(end, right_max):
        if text[i] in _SENTENCE_BOUNDARIES:
            right_bound = i + 1  # include the terminator itself
            break

    return text[left_bound:right_bound].strip()


__all__ = ["MentionSpan", "extract_entity_mentions"]
