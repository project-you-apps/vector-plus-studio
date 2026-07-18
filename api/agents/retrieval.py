"""Shared retrieval helper — top-N pattern selection for LLM context.

Q&A and Cart Curator both need "give me the N patterns most relevant to
this query" so the LLM prompt carries actual cart substrate rather than
an unfounded synthesis. This module centralizes that retrieval so both
agents (and future ones) share one implementation.

Design decisions
----------------
- **No embeddings for v1.** The MVP uses lexical bigram+token overlap
  against a normalized query. Reasons:
    - The LLM adapter route is thread-pool-driven and returns text.
      Threading in an embed call would need a second adapter interface
      and add end-to-end latency we don't need for the pitch surface.
    - Every registered cart on the demo droplet has passages a
      keyword-overlap heuristic can find given a well-formed question.
    - Adds no new dependencies.
  When we outgrow this — bilingual carts, image-heavy carts, agents
  running against non-English cartridges — swap the ranker for a real
  embedding call (probably reusing the existing ``/api/embed`` server-
  side path). The signature (``retrieve_top_patterns``) stays stable.

- **Tombstone-aware.** Same rule as reports: hippocampus row byte 28,
  bit 0 = tombstoned. Skipped silently.

- **Bounded prefix.** Each candidate passage is truncated to a prefix
  before both scoring AND context assembly. Bounds compute + prompt
  token count; a 200-char prefix carries enough signal to score and
  enough content to answer most questions.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

from api.cartridge_io import (
    TRUTH_STATUS_ACTIVE,
    TRUTH_STATUS_SUPERSEDED,
)
from api.reports.cart_reader import CartHandle


# ---------------------------------------------------------------------------
# Tokenizer constants — mirror api/reports/summary.py.
# (Tombstone check lives on CartHandle.is_tombstoned; kept single-sourced.)
# ---------------------------------------------------------------------------

# Same stopword set as summary.py; kept local to avoid a cross-module
# import of a module-private helper (matches the Reports Wave 1b idiom
# of every module being self-contained).
_STOPWORDS: frozenset[str] = frozenset({
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
})

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")

# Prefix length used for scoring + context. Long enough to carry meaning;
# short enough to keep the LLM context bounded on top-8 retrieval.
CANDIDATE_PREFIX_CHARS = 400


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPattern:
    """One hit from the top-N retrieval pass.

    Callers assemble the prompt context from these + save the ``idx``
    list into ``AgentOutput.cited_patterns`` so the frontend can render
    ``vps://source/{slug}`` links back to the source file.
    """

    idx: int
    source: str
    snippet: str
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pattern-filter mode strings — mirror ``AgentInput.pattern_filter``. Defined
# here (rather than imported from base) so retrieval.py has no dep back into
# the agent runtime; base.py's Literal type is the source of truth for the
# API surface, this string tuple is just the value space.
PatternFilterMode = Literal["active_only", "include_superseded", "all"]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _tokens_no_stop(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2]


def _should_include(
    cart: CartHandle,
    idx: int,
    pattern_filter: PatternFilterMode,
) -> bool:
    """Return True if pattern ``idx`` passes the given ``pattern_filter``.

    Tombstoned patterns are ALWAYS excluded — tombstone is a delete flag,
    orthogonal to lifecycle. Everything else is filtered per mode:

    * ``"active_only"`` — only truth_status == ACTIVE (the default; matches
      the "current relevant patterns" semantics agents want by default).
    * ``"include_superseded"`` — ACTIVE + SUPERSEDED (useful for
      diff-style questions where superseded history is relevant).
    * ``"all"`` — every non-tombstoned pattern, regardless of lifecycle.

    Unknown ``pattern_filter`` values fall through to ``"active_only"``,
    matching the safest possible default.
    """
    if cart.is_tombstoned(idx):
        return False
    if pattern_filter == "all":
        return True
    status = cart.truth_status(idx)
    if pattern_filter == "include_superseded":
        return status in (TRUTH_STATUS_ACTIVE, TRUTH_STATUS_SUPERSEDED)
    # Safe default: active_only.
    return status == TRUTH_STATUS_ACTIVE


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def retrieve_top_patterns(
    cart: CartHandle,
    query: str,
    top_n: int,
    *,
    prefix_chars: int = CANDIDATE_PREFIX_CHARS,
    pattern_filter: PatternFilterMode = "active_only",
) -> list[RetrievedPattern]:
    """Return the ``top_n`` most query-relevant live patterns.

    Ranks by weighted token+bigram overlap against ``query``. Empty
    ``query`` short-circuits to the FIRST ``top_n`` non-tombstoned
    patterns (useful for Auto Briefing / Cart Curator when the user
    hasn't supplied a specific ask).

    Empty cart → empty list. Tombstoned patterns are always skipped.
    Lifecycle filtering follows ``pattern_filter``:

    * ``"active_only"`` (default) — truth_status ACTIVE only. Matches
      the "current relevant patterns" semantics agents want by default.
    * ``"include_superseded"`` — ACTIVE + SUPERSEDED. Useful for
      diff / history queries.
    * ``"all"`` — every non-tombstoned pattern.

    Existing carts (byte 30 == 0) read as ACTIVE, so behavior is
    unchanged from the pre-2026-07-17 tombstone-only version.
    """
    if top_n <= 0:
        return []

    q_tokens = _tokens_no_stop(query or "")

    # Empty query fast path: return the first N live patterns in cart order.
    # Auto Briefing uses this when the user gives no specific focus.
    if not q_tokens:
        out: list[RetrievedPattern] = []
        for idx in range(cart.count):
            if not _should_include(cart, idx, pattern_filter):
                continue
            passage = cart.get_passage(idx)
            source = cart.get_source(idx) or ""
            out.append(RetrievedPattern(
                idx=idx,
                source=source,
                snippet=passage[:prefix_chars],
                score=0.0,
            ))
            if len(out) >= top_n:
                break
        return out

    # Build query token set + bigram set for weighted scoring.
    q_token_set = set(q_tokens)
    q_bigrams: set[str] = set()
    for i in range(len(q_tokens) - 1):
        q_bigrams.add(f"{q_tokens[i]} {q_tokens[i + 1]}")

    # Score every live pattern once. O(N * prefix_chars) which is fine
    # for cart sizes we care about (< 10k patterns typical).
    scored: list[tuple[float, int, str, str]] = []
    for idx in range(cart.count):
        if not _should_include(cart, idx, pattern_filter):
            continue
        passage = cart.get_passage(idx)
        prefix = passage[:prefix_chars]
        p_tokens = _tokens_no_stop(prefix)
        if not p_tokens:
            continue
        # Token overlap (weighted by query token count in passage) plus
        # a bigram bonus that rewards phrases matching exactly.
        p_token_counts = Counter(p_tokens)
        token_score = sum(p_token_counts[t] for t in q_token_set)
        bigram_score = 0.0
        for i in range(len(p_tokens) - 1):
            bg = f"{p_tokens[i]} {p_tokens[i + 1]}"
            if bg in q_bigrams:
                bigram_score += 3.0  # bigrams outweigh single-token hits
        # Length normalization so long passages don't dominate purely
        # by having more chances to match. Cap at 1 to avoid dividing
        # by tiny denominators.
        norm = 1.0 + (len(p_tokens) ** 0.5) / 10.0
        total = (token_score + bigram_score) / norm
        if total <= 0:
            continue
        source = cart.get_source(idx) or ""
        scored.append((total, idx, source, prefix))

    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:top_n]
    return [
        RetrievedPattern(idx=idx, source=source, snippet=snippet, score=score)
        for (score, idx, source, snippet) in top
    ]


def format_context_block(patterns: Iterable[RetrievedPattern]) -> str:
    """Format retrieved patterns as a context block for an LLM prompt.

    Structure the model likes: a numbered list of ``[N] source: text``
    blocks, blank-line separated. Numbers become the "cite [3]" hook
    for the model's answer.
    """
    lines: list[str] = []
    for i, p in enumerate(patterns, start=1):
        src = p.source or "unknown source"
        # Collapse internal newlines so each passage is one paragraph in
        # the prompt — LLMs handle that shape better than embedded
        # newlines mid-block.
        snippet = " ".join(p.snippet.split())
        lines.append(f"[{i}] ({src}) {snippet}")
    return "\n\n".join(lines)


__all__ = [
    "RetrievedPattern",
    "retrieve_top_patterns",
    "format_context_block",
    "CANDIDATE_PREFIX_CHARS",
]
