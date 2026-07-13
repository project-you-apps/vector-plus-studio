"""Summary Report — "what's in this cart?" cart orientation.

Wave-1 report (no LLM dependency). Implements Section 1 of
``docs/vps-internal/Report Types Design 2026-07-10.md``.

Answers the orientation question without a search: pattern count, unique
source files, date span across passages, extracted-content counts
(graphics + tables), a dumb-but-works bigram-clustered top-themes list,
and a per-file table with first/last-seen dates.

Design decisions worth flagging for later waves:

- **Slug**: ``"summary"`` (underscore convention). Matches
  ``frontend/src/reports/report-definitions.ts``.
- **Themes**: this wave does NOT hit Membot — post-Wave-1a. Instead we
  cluster by shared bigrams (first 200 chars per passage). If fewer than
  3 distinct themes surface, we fall back to Pattern-0 description
  tokens. Wave-2 replaces this with high-diversity Membot search.
- **Tombstones**: skipped from counts + coverage + themes. Bit 0 of
  byte 28 in each hippocampus row (per
  ``api/cartbuilder/cartridge_builder.py::FLAG_TOMBSTONE``). If any
  tombstones were skipped we surface a warning noting the count.
- **Extractors**: imported lazily inside ``generate()`` so the module
  loads even if ``api.reports.extractors`` hasn't shipped yet. When
  the module IS present, we assume ``extract_dates(text) -> iterable``
  where each element has either a ``date`` attribute
  (datetime / date / ISO string) or is itself a datetime / date /
  ISO string. Best-effort coercion; anything unparseable is dropped.
- **Pattern-0 missing**: warn + fall back to derived stats (cart name
  from filename stem, no description, no graphic/table counts).
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime
from typing import Any, Iterable, Optional

from .base import Report, ReportInput, ReportOptions, ReportOutput
from .cart_reader import CartHandle
from .registry import register_report
from .source_link import source_link


# ---------------------------------------------------------------------------
# Tombstone detection — mirrors api/cartbuilder/cartridge_builder.py
# ---------------------------------------------------------------------------

# HIPPO_FORMAT flags byte lives at offset 28 in the 64-byte row.
# Bit 0 = tombstone (see FLAG_TOMBSTONE in cartridge_builder.py).
_HIPPO_FLAGS_OFFSET = 28
_FLAG_TOMBSTONE = 0x01


# ---------------------------------------------------------------------------
# Bigram theme extractor — the dumb-but-works Wave-1 approach
# ---------------------------------------------------------------------------

# Very small English stopword set — enough to filter the worst bigram noise
# without pulling in an NLP dep. If we grow this we should move to a shared
# helper; for now local + narrow keeps the module standalone.
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

# Word tokenizer: keep alphanumerics + hyphen; treat everything else as boundary.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")

# Passage prefix used for theme mining — keeps compute bounded on large carts.
_THEME_PASSAGE_PREFIX = 200


def _bigrams(text: str) -> list[str]:
    """Yield lowercase (non-stopword) 2-word bigrams from ``text``.

    Bigram = two consecutive kept tokens. Any token that lands in
    ``_STOPWORDS`` is skipped — which also means bigrams straddling a
    stopword don't survive. This mirrors the "collocation" flavor of
    theme extraction (adjective+noun, noun+noun) without needing POS
    tagging.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return [f"{kept[i]} {kept[i + 1]}" for i in range(len(kept) - 1)]


def _fallback_theme_tokens(description: str, top_n: int) -> list[tuple[str, int]]:
    """When bigram clustering yields fewer than 3 themes, mine Pattern-0
    description tokens instead. Returns (token, weight=1) pairs so the
    output shape stays uniform.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(description or "")]
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) > 3]
    # De-dupe preserving order (description tokens are already meaningful;
    # no need to count them).
    seen: dict[str, None] = {}
    for t in kept:
        if t not in seen:
            seen[t] = None
    return [(t, 1) for t in list(seen.keys())[:top_n]]


# ---------------------------------------------------------------------------
# Date helpers — coerce whatever the extractors module gives us
# ---------------------------------------------------------------------------

# Fallback date-parser for ISO-shaped strings when the extractors module
# hands us plain strings instead of parsed objects. We only need YYYY-MM-DD
# precision for Coverage; anything richer is silently truncated.
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _coerce_date(value: Any) -> Optional[date]:
    """Best-effort convert whatever the extractor returned to a ``date``.

    Accepts datetime / date / ISO string / an object with ``.date``
    attribute (the shape hinted at in the design doc). Returns ``None``
    for anything we can't confidently parse — the caller drops it.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # DateExtraction-shaped object with a .date attribute — the extractors
    # module's likely public shape per Section 0.3 of the design doc.
    if hasattr(value, "date"):
        inner = value.date  # type: ignore[attr-defined]
        # Some libs expose .date as a method; call if so.
        if callable(inner):
            try:
                inner = inner()
            except Exception:
                return None
        return _coerce_date(inner)
    # Fall back to string parsing.
    if isinstance(value, str):
        m = _ISO_DATE_RE.search(value)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def _try_extract_dates(text: str) -> list[date]:
    """Lazy-import the extractors module and pull dates from ``text``.

    Returns ``[]`` if the module isn't importable yet (Wave-1 parallel
    dispatch) or if extraction yields nothing usable. Never raises.
    """
    try:
        from api.reports import extractors  # type: ignore
    except Exception:
        return []
    extract_dates = getattr(extractors, "extract_dates", None)
    if extract_dates is None:
        return []
    try:
        raw = extract_dates(text)
    except Exception:
        return []
    if not raw:
        return []
    out: list[date] = []
    for item in raw:
        d = _coerce_date(item)
        if d is not None:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@register_report
class SummaryReport(Report):
    """Cart orientation report — "what's in this cart?"."""

    # ---- class-level metadata (mirrors report-definitions.ts) -----------
    name = "summary"
    display_name = "Summary"
    description = "Cart orientation — what's in this cart, top themes, sources."
    llm_dependency = False
    supports_scheduling = True

    # Mirror the frontend's summary FieldSchema entries verbatim. Keeping
    # this in sync is the discipline the Wave-1b smoke test will assert.
    input_schema: list[dict[str, Any]] = [
        {
            "name": "top_themes",
            "label": "Top themes",
            "type": "number",
            "required": False,
            "default": 5,
            "helpText": "How many top themes to surface (default 5).",
        },
        {
            "name": "date_range",
            "label": "Date range",
            "type": "date-range",
            "required": False,
            "helpText": "Optional — restrict summary to a date window.",
        },
    ]

    # ---- generate --------------------------------------------------------
    def generate(
        self,
        cart_path: str,
        inputs: ReportInput,
        options: ReportOptions,
    ) -> ReportOutput:
        cart = CartHandle(cart_path)
        warnings: list[str] = list(cart.length_warnings)
        metadata: dict[str, Any] = {}

        # ---- inputs ------------------------------------------------------
        top_themes_n = inputs.get_int("top_themes", 5) or 5
        top_themes_n = max(1, top_themes_n)

        # ---- Pattern-0 metadata (with graceful fallback) -----------------
        p0 = cart.pattern0 or {}
        if not cart.pattern0:
            warnings.append(
                "Pattern-0 metadata is missing or in a binary-header shape "
                "we can't parse; falling back to derived stats."
            )
        cart_name = str(p0.get("cart_name") or cart.cart_name)
        p0_description = str(p0.get("description") or "")
        graphic_count = int(p0.get("graphic_count") or 0)
        table_count = int(p0.get("table_count") or 0)

        # ---- Empty-cart short circuit ------------------------------------
        if cart.count == 0:
            md = f"# {cart_name}\n\nThis cart is empty.\n"
            metadata["pattern_count"] = 0
            return ReportOutput(markdown=md, metadata=metadata, warnings=warnings)

        # ---- Tombstone-aware pattern enumeration -------------------------
        # We walk every pattern once, gathering:
        #   - live pattern indices (for later theme mining + counts)
        #   - per-source pattern counts (skipping tombstoned rows)
        #   - concatenated text sample for date extraction (bounded per-passage)
        live_indices: list[int] = []
        tombstoned = 0
        per_source_count: Counter[str] = Counter()

        for idx, _text, source in cart.iter_passages():
            row = cart.get_hippocampus_row(idx)
            if row is not None and len(row) > _HIPPO_FLAGS_OFFSET:
                if int(row[_HIPPO_FLAGS_OFFSET]) & _FLAG_TOMBSTONE:
                    tombstoned += 1
                    continue
            live_indices.append(idx)
            # Empty source label = "unknown"; keeps the source table honest.
            per_source_count[source or ""] += 1

        if tombstoned:
            warnings.append(
                f"Skipped {tombstoned} tombstoned pattern"
                f"{'s' if tombstoned != 1 else ''} from counts / coverage / themes."
            )
        metadata["pattern_count"] = len(live_indices)
        metadata["tombstoned_skipped"] = tombstoned

        live_count = len(live_indices)
        if live_count == 0:
            # All patterns tombstoned — treat as empty for reporting purposes.
            md = (
                f"# {cart_name}\n\nThis cart has no live patterns "
                f"({tombstoned} tombstoned).\n"
            )
            return ReportOutput(markdown=md, metadata=metadata, warnings=warnings)

        unique_sources = [s for s in per_source_count.keys() if s]
        metadata["unique_source_count"] = len(unique_sources)

        # ---- Date extraction across live passages ------------------------
        # Per-source first/last-seen tracking piggybacks on the same scan.
        source_dates: dict[str, list[date]] = {}
        all_dates: list[date] = []
        for idx in live_indices:
            text = cart.get_passage(idx)
            src = cart.get_source(idx) or ""
            dates = _try_extract_dates(text)
            if dates:
                all_dates.extend(dates)
                source_dates.setdefault(src, []).extend(dates)

        min_date = min(all_dates) if all_dates else None
        max_date = max(all_dates) if all_dates else None

        # ---- Theme mining -----------------------------------------------
        # Global bigram frequency across the first _THEME_PASSAGE_PREFIX chars
        # of every LIVE passage. Cheap and reasonable for orientation.
        bigram_counts: Counter[str] = Counter()
        for idx in live_indices:
            text = cart.get_passage(idx)[:_THEME_PASSAGE_PREFIX]
            bigram_counts.update(_bigrams(text))

        top_themes: list[tuple[str, int]] = bigram_counts.most_common(top_themes_n)
        # Design doc: if fewer than 3 distinct themes emerge, fall back to
        # Pattern-0 description tokens.
        if len(top_themes) < 3:
            fallback = _fallback_theme_tokens(p0_description, top_themes_n)
            if fallback:
                top_themes = fallback
                warnings.append(
                    "Bigram theme clustering produced fewer than 3 themes; "
                    "falling back to Pattern-0 description tokens."
                )

        # ---- Markdown assembly ------------------------------------------
        lines: list[str] = []
        lines.append(f"# {cart_name}")
        lines.append("")

        # Count line + coverage line + extracted line. Coverage is dropped
        # when we couldn't extract any dates (per design-doc edge case).
        count_line = (
            f"**{live_count}** pattern{'s' if live_count != 1 else ''} "
            f"across **{len(unique_sources)}** source "
            f"file{'s' if len(unique_sources) != 1 else ''}"
        )
        lines.append(count_line)

        if min_date and max_date:
            lines.append(f"**Coverage**: {min_date.isoformat()} → {max_date.isoformat()}")

        # Only surface the Extracted line if we have counts to show; if
        # both are zero and Pattern-0 was missing, skip it.
        if graphic_count or table_count or cart.pattern0:
            lines.append(
                f"**Extracted**: {graphic_count} graphic"
                f"{'s' if graphic_count != 1 else ''}, "
                f"{table_count} table{'s' if table_count != 1 else ''}"
            )
        lines.append("")

        # ---- Top themes section -----------------------------------------
        if top_themes:
            lines.append("## Top themes")
            for i, (label, cnt) in enumerate(top_themes, start=1):
                lines.append(f"{i}. {label} ({cnt} pattern{'s' if cnt != 1 else ''})")
            lines.append("")

        # ---- Source files section ---------------------------------------
        # Design-doc edge case: single-source cart skips the table
        # (redundant with the count line).
        if len(unique_sources) > 1:
            lines.append("## Source files")
            lines.append("| File | Patterns | First seen | Last seen |")
            lines.append("|---|---|---|---|")
            # Order by descending pattern count then filename for stability.
            for src in sorted(
                unique_sources,
                key=lambda s: (-per_source_count[s], s),
            ):
                dates_for_src = source_dates.get(src) or []
                first_seen = (
                    min(dates_for_src).isoformat() if dates_for_src else "—"
                )
                last_seen = (
                    max(dates_for_src).isoformat() if dates_for_src else "—"
                )
                lines.append(
                    f"| {source_link(src)} | {per_source_count[src]} | {first_seen} | {last_seen} |"
                )
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"

        # ---- Metadata for the audit footer ------------------------------
        metadata["date_coverage"] = {
            "min": min_date.isoformat() if min_date else None,
            "max": max_date.isoformat() if max_date else None,
            "count": len(all_dates),
        }
        metadata["graphic_count"] = graphic_count
        metadata["table_count"] = table_count
        metadata["top_themes"] = [
            {"label": label, "count": cnt} for label, cnt in top_themes
        ]
        # Source refs — the executor's include_source_refs toggle lets us
        # skip payload weight when bandwidth-sensitive (email, Slack).
        if options.include_source_refs:
            metadata["source_files"] = [
                {"name": s, "patterns": per_source_count[s]}
                for s in unique_sources
            ]

        return ReportOutput(
            markdown=markdown,
            metadata=metadata,
            warnings=warnings,
        )


__all__ = ["SummaryReport"]
