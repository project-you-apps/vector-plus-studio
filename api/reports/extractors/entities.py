"""Entity-mention extraction from raw passage text.

Two shapes are exported:

- :func:`extract_entity_mentions` — targeted-lookup: given a canonical
  entity name (and optional aliases), find every mention in the text.
  Case-insensitive with sentence-aware context. Entity Rollup is the
  primary consumer.
- :func:`discover_entities` — discovery: given text WITHOUT a target
  list, surface proper-noun candidates using capitalization heuristics
  + a first-token stopword filter. Coverage Report's orphan-entity
  detection is the primary consumer; any future report that needs
  discovery rather than lookup reuses it here.
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


# ---------------------------------------------------------------------------
# discover_entities — discovery-shape companion (no target list)
# ---------------------------------------------------------------------------

# Default stopword set for entity discovery. First-token filter — if the
# first capitalized token in a proper-noun candidate lower-cases to a
# member of this set, the candidate is dropped. Covers standard English
# function words that occasionally start sentences and weekday / month
# names that naturally capitalize in prose without being entities.
_DISCOVERY_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "in", "on",
    "at", "to", "for", "with", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "i", "we", "you", "they", "he",
    "she", "him", "her", "them", "us", "my", "your", "our", "their", "his",
    "hers", "not", "no", "yes", "so", "than", "from", "into", "onto",
    "up", "down", "out", "over", "under", "about", "just", "very", "also",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "there", "here", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "any", "some", "all", "each", "every", "one",
    "two", "three", "am", "pm",
    # Weekdays / months — capitalize naturally in prose.
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})


def discover_entities(
    text: str,
    min_length: int = 2,
    extra_stopwords: Optional[frozenset[str]] = None,
) -> list[str]:
    """Discover proper-noun entity mentions in ``text`` without a target list.

    Companion to :func:`extract_entity_mentions` (targeted-lookup shape).
    Returns discovered entity surface forms deduplicated
    case-insensitively (first-seen surface form wins) in first-seen
    offset order — the shape Coverage Report's orphan-entity detection
    wants, and the shape any future report needing discovery-not-lookup
    will reuse.

    Heuristic: capture 1-3 consecutive capitalized tokens where each
    token starts with an uppercase ASCII letter followed by
    ``min_length - 1`` or more alphanumerics. Reject when the first
    token (lowercased) matches the built-in stopword set OR the
    ``extra_stopwords`` frozenset (a caller-supplied domain-specific
    tightening).

    Args:
        text: Raw passage text.
        min_length: Minimum characters per capitalized token. Default 2
            lets 2-char acronyms like ``US`` through. Callers wanting to
            filter shorter sentence openers should pass ``3`` — Coverage
            Report's orphan detection does exactly that.
        extra_stopwords: Additional first-token stopwords (lowercased)
            to merge into the default set.

    Returns:
        List of discovered entity strings in first-seen order.
    """
    if not text:
        return []
    if min_length < 1:
        min_length = 1
    tail = min_length - 1  # ``{tail,}`` on the trailing char class
    pattern = re.compile(
        r"\b([A-Z][a-zA-Z0-9]{" + str(tail) + r",}"
        r"(?:\s+[A-Z][a-zA-Z0-9]{" + str(tail) + r",}){0,2})\b"
    )
    stopwords = _DISCOVERY_STOPWORDS
    if extra_stopwords:
        stopwords = stopwords | extra_stopwords

    seen: set[str] = set()
    ordered: list[str] = []
    for m in pattern.finditer(text):
        candidate = m.group(1).strip()
        if not candidate:
            continue
        first_tok = candidate.split()[0].lower()
        if first_tok in stopwords:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


__all__ = ["MentionSpan", "extract_entity_mentions", "discover_entities"]
