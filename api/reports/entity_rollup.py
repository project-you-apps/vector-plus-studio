"""Entity Rollup report — "show me all mentions of X" (Report Types Design §5).

Wave-1 report, no LLM dependency. Answers "show me every mention of Sysco
Portland" (or any vendor / person / project / concept name) across a cart,
chronologically, with per-mention source + optional context excerpt.

Design source: ``docs/vps-internal/Report Types Design 2026-07-10.md``
section 5. Frontend mirror: ``frontend/src/reports/report-definitions.ts``
(``entity_rollup`` entry — slug uses underscore, not hyphen).

Query strategy (per §5):

1. Load the cart via :class:`~api.reports.cart_reader.CartHandle`.
2. Walk every non-tombstoned passage via ``cart.iter_passages()``.
3. Run ``extract_entity_mentions()`` (lazy imported from
   :mod:`api.reports.extractors` when available; local regex fallback
   until the parallel Wave-1a dispatch lands) against each passage.
4. For each mention, capture (pattern_idx, source_path, span, date).
   Date comes from ``extract_dates`` on the same passage (or on the
   source path, which for Cart Builder invoices tends to carry
   ``-YYYY-MM-DD.`` in the filename).
5. Sort chronologically; undated mentions bucketed last.

Wave-1 explicitly excludes the §5 co-occurrence aggregation (which
vendors appear alongside this one). That's a Wave-2 add.

Slug discipline: ``name = "entity_rollup"`` matches the frontend
underscore-slug in ``report-definitions.ts``. The registry rejects
duplicate slugs, so if this file gets imported twice the import will
fail loudly rather than silently double-register.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .base import Report, ReportInput, ReportOptions, ReportOutput
from .cart_reader import CartHandle
from .registry import register_report


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hippocampus flags byte lives at offset 28 in the 64-byte row (see
# ``api/cartridge_io.py::HIPPO_FORMAT`` and ``cartbuilder/cartridge_builder.py``
# for the layout). Bit 0 = tombstone.
HIPPO_FLAGS_OFFSET = 28
FLAG_TOMBSTONE = 0x01

# Threshold above which we warn the user that the search was too generic.
# Grant's 6-month Sysco cart is expected to produce ~30-99 mentions for
# a proper entity_name; >100 usually means someone typed "invoice" or "Portland".
GENERIC_MATCH_WARNING_THRESHOLD = 100

# Context window per §5: "surrounding sentences" ~= ±100 chars unless a
# sentence terminator is closer.
CONTEXT_RADIUS = 100

# Sentence-terminator characters we prefer as excerpt boundaries when
# they fall inside the CONTEXT_RADIUS window.
_SENTENCE_TERMINATORS = ".!?\n"


# ---------------------------------------------------------------------------
# Internal shape for a single mention
# ---------------------------------------------------------------------------

@dataclass
class _Mention:
    """One entity mention inside one passage. Not exported."""
    pattern_idx: int
    source: str
    match_start: int
    match_end: int
    matched_text: str
    date_str: Optional[str]          # ISO "YYYY-MM-DD" or None
    context_excerpt: str             # markdown, already bold-highlighted


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@register_report
class EntityRollupReport(Report):
    """All-mentions-of-X rollup, chronological.

    See module docstring for design source + query strategy. This class
    is the Wave-1 implementation — regex-only entity + date extraction
    with a graceful local fallback if the shared extractors module
    hasn't landed yet.
    """

    # Slug MUST match the frontend entry (underscore, not hyphen). The
    # registry decorator asserts this by refusing an empty ``name``, and
    # a Wave-1b smoke test asserts backend+frontend equality.
    name = "entity_rollup"
    display_name = "Entity Rollup"
    description = (
        "All mentions of a specific entity — vendor, person, project, concept."
    )

    # Mirrors the frontend ``entity_rollup`` inputSchema
    # (``report-definitions.ts``). ``include_context`` is authored on
    # the backend per design doc §5 default = True; the frontend surface
    # doesn't render a toggle for it yet, but the executor will pass it
    # through untouched once the form gains the field. Wave-1a agent
    # brief 2026-07-11 called this out.
    input_schema: list[dict[str, Any]] = [
        {
            "name": "entity_name",
            "label": "Entity name",
            "type": "text",
            "required": True,
            "placeholder": "e.g. Sysco Portland",
        },
        {
            "name": "aliases",
            "label": "Aliases",
            "type": "text",
            "required": False,
            "placeholder": "comma-separated alternate spellings",
            "helpText": "Optional — e.g. \"SP, Sysco PDX, Sysco-Portland\".",
        },
        {
            # Frontend enum doesn't include "checkbox"; render as a
            # two-option select for now. Executor treats the string
            # payload via ``ReportInput.get_bool``.
            "name": "include_context",
            "label": "Include context excerpts",
            "type": "select",
            "required": False,
            "default": "true",
            "options": ["true", "false"],
            "helpText": (
                "Show ±100 chars of surrounding text under each mention "
                "(default on)."
            ),
        },
    ]

    llm_dependency = False
    # Entity rollups make sense as recurring briefs — "weekly Sysco
    # summary" is the canonical Hot Stack composition example in §C.4.
    supports_scheduling = True

    # -- interface --------------------------------------------------------
    def generate(
        self,
        cart_path: str,
        inputs: ReportInput,
        options: ReportOptions,
    ) -> ReportOutput:
        cart = CartHandle(cart_path)

        entity_name = inputs.get_str("entity_name")
        if not entity_name:
            # Executor wraps the message with report + cart context; keep
            # this one narrow so it points at the actual missing field.
            raise ValueError(
                "entity_rollup requires 'entity_name' (see input_schema)."
            )
        aliases: list[str] = [
            str(a).strip() for a in inputs.get_list("aliases") if str(a).strip()
        ]
        include_context = inputs.get_bool("include_context", default=True)

        # Extractors are dispatched in parallel; if the shared module
        # hasn't landed yet we fall back to the local regex impls.
        extract_entity_mentions, extract_dates = _load_extractors()

        # Walk every passage; skip tombstones; collect mentions.
        mentions: list[_Mention] = []
        for idx, passage_text, source in cart.iter_passages():
            if _is_tombstoned(cart, idx):
                continue
            if not passage_text:
                continue
            spans = extract_entity_mentions(
                passage_text, entity_name, aliases=aliases
            )
            if not spans:
                continue
            # One date per passage — mentions on the same passage share it.
            date_str = _extract_first_date(extract_dates, passage_text, source)
            for span in spans:
                start, end, matched, precomputed_ctx = _span_bounds(span)
                excerpt = _build_excerpt(
                    passage_text,
                    start,
                    end,
                    matched,
                    precomputed_ctx,
                    include_context,
                )
                mentions.append(
                    _Mention(
                        pattern_idx=idx,
                        source=source,
                        match_start=start,
                        match_end=end,
                        matched_text=matched,
                        date_str=date_str,
                        context_excerpt=excerpt,
                    )
                )

        # Warnings from the cart handle propagate to the ReportOutput so
        # the UI's audit footer can surface any shape mismatches.
        warnings: list[str] = list(cart.length_warnings)

        # No matches → short, friendly message. Skip the header + table
        # fluff since there's nothing to describe.
        if not mentions:
            md = (
                f"# Entity: {entity_name} — {cart.cart_name}\n\n"
                "No mentions found — check spelling or try aliases."
            )
            return ReportOutput(
                markdown=md,
                metadata={
                    "total_mentions": 0,
                    "unique_sources": 0,
                    "entity_name": entity_name,
                    "aliases_used": aliases,
                },
                warnings=warnings,
            )

        # Generic-word warning — surface both in ReportOutput.warnings
        # (audit surface) and inline in the markdown (user-facing).
        generic_warning: Optional[str] = None
        if len(mentions) > GENERIC_MATCH_WARNING_THRESHOLD:
            generic_warning = (
                f"Your search matched {len(mentions)} common results; "
                f"consider a more specific entity_name or add aliases to constrain."
            )
            warnings.append(generic_warning)

        # Chronological sort: undated last. Secondary key = pattern_idx
        # so mentions inside the same passage stay stable + reproducible.
        mentions.sort(
            key=lambda m: (m.date_str is None, m.date_str or "", m.pattern_idx)
        )

        unique_sources = len({m.source for m in mentions if m.source})
        dated = [m.date_str for m in mentions if m.date_str]
        first_date = min(dated) if dated else None
        last_date = max(dated) if dated else None

        markdown = _render_markdown(
            entity_name=entity_name,
            cart_name=cart.cart_name,
            mentions=mentions,
            first_date=first_date,
            last_date=last_date,
            unique_sources=unique_sources,
            generic_warning=generic_warning,
            include_context=include_context,
        )

        metadata: dict[str, Any] = {
            "total_mentions": len(mentions),
            "unique_sources": unique_sources,
            "first_date": first_date,
            "last_date": last_date,
            "entity_name": entity_name,
            "aliases_used": aliases,
            "include_context": include_context,
        }
        # Source refs are the "click through to passage" surface for the
        # UI; suppress when the caller has flipped them off (email / Slack
        # briefs, where payload size matters).
        if options.include_source_refs:
            metadata["source_refs"] = [
                {
                    "pattern_idx": m.pattern_idx,
                    "source": m.source,
                    "date": m.date_str,
                    "matched": m.matched_text,
                }
                for m in mentions
            ]

        return ReportOutput(
            markdown=markdown,
            metadata=metadata,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(
    *,
    entity_name: str,
    cart_name: str,
    mentions: list[_Mention],
    first_date: Optional[str],
    last_date: Optional[str],
    unique_sources: int,
    generic_warning: Optional[str],
    include_context: bool,
) -> str:
    """Render the report body per §5's output shape."""
    lines: list[str] = []
    lines.append(f"# Entity: {entity_name} — {cart_name}")
    lines.append("")
    lines.append(
        f"**{len(mentions)}** mentions across "
        f"**{unique_sources}** files"
    )
    if first_date and last_date:
        lines.append(f"**Coverage**: {first_date} → {last_date}")
    else:
        # Design doc §5 says "drop coverage line" when no dates found —
        # but this is Entity Rollup, not Summary. Keep the line but flag
        # it so the user sees why the report couldn't date-sort.
        lines.append("**Coverage**: (no datable passages)")
    lines.append("")

    if generic_warning:
        # Blockquote so it visually separates from the report body.
        lines.append(f"> **WARNING**: {generic_warning}")
        lines.append("")

    lines.append("## Chronological mentions")
    lines.append("")

    # Group by date bucket (or "undated"). ``_sentinel`` guarantees the
    # first iteration always emits a header.
    _sentinel = object()
    current_bucket: Any = _sentinel
    for m in mentions:
        bucket = m.date_str or "undated"
        if bucket != current_bucket:
            current_bucket = bucket
            lines.append(f"### {bucket}")
        if include_context:
            lines.append(f"> {m.context_excerpt}")
        source_label = m.source or "(unknown source)"
        lines.append(f"Source: {source_label} (pattern #{m.pattern_idx})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Extractor loader — lazy import, local fallback
# ---------------------------------------------------------------------------

def _load_extractors() -> tuple[Callable[..., list], Callable[..., list]]:
    """Try the shared ``api.reports.extractors`` module; fall back to the
    local regex impls if it hasn't landed yet.

    The parallel Wave-1a dispatch may not have landed by the time this
    module is imported; the fallback keeps Entity Rollup shippable
    today. When the real extractors module lands, this call resolves to
    it automatically — no code change to this file.
    """
    try:  # noqa: SIM105 — keep the intent explicit
        from .extractors import (  # type: ignore[attr-defined]
            extract_entity_mentions,
            extract_dates,
        )
        return extract_entity_mentions, extract_dates
    except ImportError:
        return _fallback_extract_entity_mentions, _fallback_extract_dates


# ---------------------------------------------------------------------------
# Fallback extractor implementations (used until extractors module lands)
# ---------------------------------------------------------------------------

def _fallback_extract_entity_mentions(
    text: str,
    entity_name: str,
    aliases: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Case-insensitive, word-boundary-preferring substring matcher.

    Returns a list of ``{"start", "end", "matched"}`` dicts. The shape
    is what the real extractor is expected to return (or something
    equivalent — ``_span_bounds`` accepts dicts + duck-typed objects).
    """
    aliases = aliases or []
    targets = [entity_name] + [a for a in aliases if a]
    # De-dup while preserving order (short names could shadow longer aliases).
    seen: dict[str, None] = {}
    for t in targets:
        if t and t not in seen:
            seen[t] = None
    ordered = list(seen.keys())
    if not ordered:
        return []
    # Sort by length descending so "Sysco Portland" matches before "Sysco"
    # steals the prefix. re.finditer returns non-overlapping matches
    # per alternation branch, so this ordering matters.
    ordered.sort(key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in ordered)
    hits: list[dict[str, Any]] = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        hits.append({
            "start": m.start(),
            "end": m.end(),
            "matched": m.group(0),
        })
    return hits


# Regex date extractors, wave-1 fallback. Order matters — ISO first so
# "2026-05-16" doesn't get misread by the compact-US pattern.
_DATE_REGEXES: list[tuple[re.Pattern[str], str]] = [
    # ISO YYYY-MM-DD or YYYY/MM/DD
    (re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"), "iso"),
    # "May 16, 2026" / "May 16 2026"
    (
        re.compile(
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* "
            r"(\d{1,2}),? (\d{4})\b",
            re.IGNORECASE,
        ),
        "month_name",
    ),
    # US MM/DD/YYYY or MM-DD-YYYY
    (re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b"), "us"),
    # Sysco-style compact MMDDYYYY in filenames (per Timeline §2 brief)
    (re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)"), "compact_us"),
]

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _fallback_extract_dates(text: str) -> list[dict[str, Any]]:
    """Return list of ``{"date": "YYYY-MM-DD", "start", "end"}`` matches."""
    hits: list[dict[str, Any]] = []
    for regex, kind in _DATE_REGEXES:
        for m in regex.finditer(text):
            iso = _to_iso(m.groups(), kind)
            if iso:
                hits.append({"date": iso, "start": m.start(), "end": m.end()})
    return hits


def _to_iso(groups: tuple[str, ...], kind: str) -> Optional[str]:
    """Normalize a regex-group tuple into an ISO date string, or None
    if the numeric components are out of range."""
    try:
        if kind == "iso":
            y, mo, d = int(groups[0]), int(groups[1]), int(groups[2])
        elif kind == "us":
            mo, d, y = int(groups[0]), int(groups[1]), int(groups[2])
        elif kind == "month_name":
            mo = _MONTHS.get(groups[0][:3].lower(), 0)
            if not mo:
                return None
            d = int(groups[1])
            y = int(groups[2])
        elif kind == "compact_us":
            mo, d, y = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            return None
    except (ValueError, TypeError):
        return None
    if not (1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# Shared helpers (used with either fallback or real extractors)
# ---------------------------------------------------------------------------

def _extract_first_date(
    extract_dates: Callable[[str], list[Any]],
    passage_text: str,
    source: str,
) -> Optional[str]:
    """Pick the best date for a passage.

    Strategy: prefer a date inside the passage body (that's the invoice /
    email / doc date). Fall back to the source filename — Grant's Sysco
    carts encode ``-YYYY-MM-DD.`` in the filename which is often the
    only structured date on the passage.
    """
    for target in (passage_text, source):
        if not target:
            continue
        try:
            hits = extract_dates(target)
        except Exception:
            # Defensive: an extractor bug on one passage shouldn't nuke
            # the whole report.
            hits = []
        iso_dates = [_pick_iso(h) for h in hits]
        iso_dates = [d for d in iso_dates if d]
        if iso_dates:
            # Earliest wins — heuristic assumes the "document date" is
            # the leftmost / first-mentioned date.
            return sorted(iso_dates)[0]
    return None


def _pick_iso(entry: Any) -> Optional[str]:
    """Pull the ISO date string off an extractor result.

    Handles both shapes:

    - Shared extractors module (``api.reports.extractors.dates``) returns
      :class:`DateExtraction` dataclasses with a ``.date`` attribute typed
      as :class:`datetime.date` — ``.isoformat()`` gives us the YYYY-MM-DD
      string.
    - Local fallback returns plain dicts with a ``"date"`` string.
    """
    if isinstance(entry, dict):
        # ``date`` is our fallback convention; ``iso`` / ``value`` are
        # safe aliases in case an alternative extractor picks a different
        # name.
        val = entry.get("date") or entry.get("iso") or entry.get("value")
        if val is None:
            return None
        # Guard against dict returning a date object anyway.
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return str(val)
    val = getattr(entry, "date", None) or getattr(entry, "iso", None)
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _span_bounds(span: Any) -> tuple[int, int, str, Optional[str]]:
    """Coerce a mention span (dict or duck-typed object) to
    ``(start, end, matched_text, precomputed_context)``.

    Accepts:

    - Shared extractors' :class:`MentionSpan` — has ``matched_text``,
      ``start``, ``end``, ``context`` (sentence-aware, already computed).
    - Local fallback dict — has ``matched``, ``start``, ``end`` (no
      precomputed context; ``None`` returned so caller falls back to
      :func:`_context_excerpt`).
    """
    if isinstance(span, dict):
        matched = span.get("matched_text") or span.get("matched") or span.get("text", "")
        return (
            int(span.get("start", 0)),
            int(span.get("end", 0)),
            str(matched),
            span.get("context"),
        )
    # Duck-typed object (MentionSpan dataclass, most likely).
    matched = (
        getattr(span, "matched_text", None)
        or getattr(span, "matched", None)
        or getattr(span, "text", "")
    )
    return (
        int(getattr(span, "start", 0)),
        int(getattr(span, "end", 0)),
        str(matched),
        getattr(span, "context", None),
    )


def _build_excerpt(
    text: str,
    start: int,
    end: int,
    matched: str,
    precomputed_context: Optional[str],
    include_context: bool,
) -> str:
    """Return the markdown-formatted excerpt for one mention.

    Prefers the extractor's precomputed sentence-aware context (the
    shared ``MentionSpan.context`` field); falls back to the local
    ±CONTEXT_RADIUS char slice with sentence-terminator walk when the
    extractor didn't provide one (fallback path). Either way, the
    matched surface form is bold-highlighted and whitespace is
    collapsed so the excerpt fits one blockquote line.
    """
    if not include_context:
        # Caller opted out of context — still return the highlighted
        # bare mention so the source line has something to reference.
        return f"**{matched}**"

    if precomputed_context:
        return _highlight_and_collapse(precomputed_context, matched)

    # Fallback path: derive from the raw passage using our own sentence
    # walk. Only reached when the fallback extractor is in play.
    left = _walk_left_to_sentence_start(text, start)
    right = _walk_right_to_sentence_end(text, end)
    prefix = text[left:start]
    suffix = text[end:right]
    lead = "..." if left > 0 else ""
    trail = "..." if right < len(text) else ""
    excerpt = f"{lead}{prefix}**{matched}**{suffix}{trail}"
    return re.sub(r"\s+", " ", excerpt).strip()


def _highlight_and_collapse(context: str, matched: str) -> str:
    """Bold-highlight the first case-insensitive ``matched`` occurrence
    inside ``context``, collapse whitespace.

    If ``matched`` isn't literally present (unusual — extractor should
    have kept it) we still collapse whitespace and append a bare-bold
    marker so the user sees something.
    """
    if not matched:
        return re.sub(r"\s+", " ", context).strip()
    lowered = context.lower()
    idx = lowered.find(matched.lower())
    if idx < 0:
        # Defensive: extractor's context didn't include the match. Fall
        # back to appending the bold mention so the reader still sees it.
        collapsed = re.sub(r"\s+", " ", context).strip()
        return f"{collapsed} (**{matched}**)"
    highlighted = (
        f"{context[:idx]}**{context[idx:idx + len(matched)]}**"
        f"{context[idx + len(matched):]}"
    )
    return re.sub(r"\s+", " ", highlighted).strip()


def _walk_left_to_sentence_start(text: str, start: int) -> int:
    """Return the leftmost offset within ``CONTEXT_RADIUS`` of ``start``
    that follows a sentence terminator, or ``start - CONTEXT_RADIUS`` if
    no terminator is inside the window."""
    hard_left = max(0, start - CONTEXT_RADIUS)
    for i in range(start - 1, hard_left - 1, -1):
        if text[i] in _SENTENCE_TERMINATORS:
            # Skip the terminator itself + any trailing whitespace.
            j = i + 1
            while j < start and text[j].isspace():
                j += 1
            return j
    return hard_left


def _walk_right_to_sentence_end(text: str, end: int) -> int:
    """Return the rightmost offset within ``CONTEXT_RADIUS`` of ``end``
    that includes a sentence terminator, or ``end + CONTEXT_RADIUS`` if
    no terminator is inside the window."""
    hard_right = min(len(text), end + CONTEXT_RADIUS)
    for i in range(end, hard_right):
        if text[i] in _SENTENCE_TERMINATORS:
            return i + 1  # inclusive of the terminator
    return hard_right


def _is_tombstoned(cart: CartHandle, idx: int) -> bool:
    """True iff pattern ``idx``'s hippocampus flags byte has bit 0
    (``FLAG_TOMBSTONE``) set.

    Returns False for carts without a hippocampus array — treating
    "no hippo" as "everything live" matches how the wider engine
    handles legacy carts (see ``api/search.py``).
    """
    row = cart.get_hippocampus_row(idx)
    if row is None:
        return False
    try:
        return bool(int(row[HIPPO_FLAGS_OFFSET]) & FLAG_TOMBSTONE)
    except (IndexError, TypeError):
        return False


__all__ = ["EntityRollupReport"]
