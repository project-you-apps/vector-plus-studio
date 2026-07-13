"""Coverage Report — "what did we miss?" gap diagnosis.

Wave-1c report (no LLM dependency). The 9th report type — added after the
Wave-1a foundation + Wave-1b's four reports to close the pitch
discrepancy ("9 different built-in reports" vs 8 shipped) and deliver a
substrate-composed diagnostic report with no LLM dependency.

Where the other reports answer "what's here?" Coverage answers the
inverse: **what's missing / underrepresented?** Sections rendered in
order:

1. **Coverage Overview** — item counts, source count, date range,
   extraction densities (percentage of items with dates / entities /
   currency).
2. **Underrepresented Themes** — bigram-clustered themes with fewer than
   ``min_theme_items`` items.
3. **Orphan Entities** — proper-noun candidates appearing in exactly one
   passage across the whole cart.
4. **Source Coverage** — per-source item counts + flag for sources
   contributing fewer than ``source_coverage_min`` items.
5. **Time Gaps** — gaps larger than ``gap_threshold_days`` between
   consecutive extracted dates.
6. **Context-Poor Items** — items lacking date + entity + currency
   (enrichment candidates).

Design decisions worth flagging:

- **Slug**: ``"coverage"`` — single word, no underscore. Matches the
  frontend entry in ``report-definitions.ts``.
- **Extractors**: ``extract_dates`` + ``extract_currency`` imported
  lazily inside ``generate()``. Coverage does NOT reuse
  ``extract_entity_mentions`` because that helper takes a
  ``entity_name`` argument (targeted lookup) whereas Coverage needs to
  DISCOVER entities to find orphans. A local proper-noun regex serves
  as the orphan-discovery scanner — flagged here so a future Wave adds
  a shared ``discover_entities()`` helper to ``extractors/entities.py``
  and Coverage swaps to that. The additive-only constraint on Wave 1c
  ruled out modifying the extractor module today.
- **Themes**: bigram approach mirroring ``summary.py``. Duplicated
  locally (matching the Wave-1b idiom — every report is self-contained
  even when there is overlap) rather than importing the module-private
  ``_bigrams`` helper.
- **Tombstones**: inline flag byte check
  (``row[28] & 0x01 != 0``) — the same pattern used across
  ``change_log.py``, ``entity_rollup.py``, and ``comparison.py`` until
  a shared ``is_tombstoned`` helper lands.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Optional

from .base import Report, ReportInput, ReportOptions, ReportOutput
from .cart_reader import CartHandle
from .registry import register_report
from .source_link import source_link


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tombstone flag = bit 0 of the flags byte at offset 28 of the 64-byte
# hippocampus row. Matches the pattern in change_log.py / entity_rollup.py /
# comparison.py and the FLAG_TOMBSTONE constant in
# ``api/cartbuilder/cartridge_builder.py``.
_HIPPO_FLAGS_OFFSET = 28
_FLAG_TOMBSTONE = 0x01

# Passage prefix used for theme bigram mining — matches ``summary.py`` so
# the two reports agree on what a "theme" is at the passage level.
_THEME_PASSAGE_PREFIX = 200

# Cap on how many candidate themes we consider for the underrepresented
# scan. Bigger than summary's top-N because we need enough of a tail to
# actually surface underrepresented themes (top-5 all-time-common bigrams
# would almost never be underrepresented in a real cart).
_THEME_CANDIDATE_POOL = 50

# Cap on how many underrepresented themes we display, to keep the section
# readable even on messy carts.
_UNDERREP_DISPLAY_CAP = 15

# Sample cap on the Context-Poor Items section — the design brief calls
# for at most 10 sample items with the full count reported separately.
_CONTEXT_POOR_SAMPLE_CAP = 10

# Snippet length used for one-line item samples in the theme / context-
# poor sections.
_SAMPLE_SNIPPET_CHARS = 100

# Word tokenizer for bigram mining — same shape as summary.py's tokenizer.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")

# Proper-noun candidate regex — 1-3 consecutive capitalized tokens of at
# least 3 chars each. The 3-char minimum trims most sentence-openers
# ("The", "But", "Our"); the stopword filter downstream catches the rest.
# 3-word cap keeps us from swallowing "The Board Of Directors Voted" as
# one entity.
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,}){0,2})\b"
)

# English stopword set for bigram theme mining + proper-noun filtering.
# Wider than summary.py's because orphan discovery needs to reject more
# sentence-openers ("There", "These", weekday names, months) that
# capitalize but are not entities.
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
    # Proper-noun-adjacent sentence openers that shouldn't count as
    # entities (they naturally capitalize in prose).
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_tombstoned_row(row: Optional[Any]) -> bool:
    """Return True if the raw hippocampus row's flags byte has the
    tombstone bit set.

    ``row`` is a numpy uint8 array of length 64 (or ``None`` if the cart
    has no hippocampus). Kept intentionally tolerant — missing / short
    rows read as "live" so legacy Cart Builder GUI carts without an
    H-block still surface in Coverage.
    """
    if row is None:
        return False
    if len(row) <= _HIPPO_FLAGS_OFFSET:
        return False
    try:
        return bool(int(row[_HIPPO_FLAGS_OFFSET]) & _FLAG_TOMBSTONE)
    except (TypeError, ValueError):
        return False


def _bigrams(text: str) -> list[str]:
    """Yield lowercase 2-word bigrams from ``text``, skipping stopwords.

    Mirrors ``summary.py::_bigrams`` — kept local rather than imported
    because summary.py's helper is module-private and the Wave-1b idiom
    is "each report is self-contained."
    """
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return [f"{kept[i]} {kept[i + 1]}" for i in range(len(kept) - 1)]


def _snippet(text: str, max_chars: int = _SAMPLE_SNIPPET_CHARS) -> str:
    """Trim ``text`` to ``max_chars`` at a word boundary for sample display.

    Collapses whitespace so the snippet fits on one bullet line. Returns
    "(empty passage)" for whitespace-only text so the section stays
    readable when the source is a brain-only cart.
    """
    if not text:
        return "(empty passage)"
    flat = " ".join(text.split())
    if not flat:
        return "(empty passage)"
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    space = cut.rfind(" ")
    if space >= max_chars // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _pct(numerator: int, denominator: int) -> str:
    """Format an integer ratio as a rounded percentage string.

    Returns ``"0%"`` when the denominator is zero (an empty cart) — the
    caller has already surfaced that as "no items" so the exact value
    doesn't matter, we just avoid a divide-by-zero.
    """
    if denominator <= 0:
        return "0%"
    return f"{round(100 * numerator / denominator)}%"


# ---------------------------------------------------------------------------
# Extracted-context types for one live item
# ---------------------------------------------------------------------------

# Kept as a plain dict per-item (not a dataclass) so the item builder can
# read/write fields incrementally without ceremony. Shape:
#
#   {
#     "idx":        int,             # cart pattern index
#     "text":       str,             # raw passage text
#     "source":     str,             # source label ("" if unknown)
#     "dates":      list[date],      # extracted calendar dates
#     "entities":   list[str],       # proper-noun candidates
#     "currency":   list[Any],       # extractor MoneyExtraction records
#   }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@register_report
class CoverageReport(Report):
    """Diagnose gaps in a cart — the 9th report type.

    Composed entirely from Wave-1a foundation extractors + the shared
    ``CartHandle`` reader. No LLM dependency.
    """

    # ---- class-level metadata (mirrors report-definitions.ts) -----------
    name = "coverage"
    display_name = "Coverage Report"
    description = (
        "Diagnose gaps in the cart — underrepresented themes, orphan "
        "entities, source imbalance, and time-range holes."
    )
    llm_dependency = False
    # Weekly "what did the last 7 days miss?" is a natural scheduled
    # brief so this stays True (matches Summary + Entity Rollup).
    supports_scheduling = True

    input_schema: list[dict[str, Any]] = [
        {
            "name": "min_theme_items",
            "label": "Min items per theme",
            "type": "number",
            "required": False,
            "default": 3,
            "helpText": (
                "Themes represented by fewer than this many items are "
                "flagged as underrepresented (default 3)."
            ),
        },
        {
            "name": "gap_threshold_days",
            "label": "Gap threshold (days)",
            "type": "number",
            "required": False,
            "default": 30,
            "helpText": (
                "Time gaps longer than this between consecutive dates "
                "are surfaced (default 30)."
            ),
        },
        {
            "name": "max_orphan_entities",
            "label": "Max orphan entities",
            "type": "number",
            "required": False,
            "default": 20,
            "helpText": (
                "Cap on how many orphan entities to display (default 20)."
            ),
        },
        {
            "name": "source_coverage_min",
            "label": "Source coverage min",
            "type": "number",
            "required": False,
            "default": 5,
            "helpText": (
                "Sources contributing fewer than this many items are "
                "flagged as under-utilized (default 5)."
            ),
        },
    ]

    # ---- generate --------------------------------------------------------
    def generate(
        self,
        cart_path: str,
        inputs: ReportInput,
        options: ReportOptions,
    ) -> ReportOutput:
        # Lazy extractor imports — matches the Wave-1b constraint so
        # coverage.py stays loadable even if extractors were dispatched
        # in a different agent branch.
        try:
            from .extractors.dates import extract_dates as _extract_dates
        except ImportError:  # pragma: no cover
            _extract_dates = None
        try:
            from .extractors.currency import extract_currency as _extract_currency
        except ImportError:  # pragma: no cover
            _extract_currency = None

        cart = CartHandle(cart_path)
        warnings: list[str] = list(cart.length_warnings)
        metadata: dict[str, Any] = {}

        # ---- inputs ------------------------------------------------------
        min_theme_items = inputs.get_int("min_theme_items", 3) or 3
        gap_threshold_days = inputs.get_int("gap_threshold_days", 30) or 30
        max_orphan_entities = inputs.get_int("max_orphan_entities", 20) or 20
        source_coverage_min = inputs.get_int("source_coverage_min", 5) or 5
        min_theme_items = max(1, min_theme_items)
        gap_threshold_days = max(1, gap_threshold_days)
        max_orphan_entities = max(1, max_orphan_entities)
        source_coverage_min = max(1, source_coverage_min)

        # ---- Walk cart once, gather everything --------------------------
        live_items: list[dict[str, Any]] = []
        tombstoned = 0
        for idx in range(cart.count):
            row = cart.get_hippocampus_row(idx)
            if _is_tombstoned_row(row):
                tombstoned += 1
                continue
            text = cart.get_passage(idx)
            source = cart.get_source(idx) or ""

            dates_here: list[date] = []
            if _extract_dates is not None and text:
                try:
                    for hit in _extract_dates(text):
                        d = getattr(hit, "date", None)
                        if isinstance(d, date):
                            dates_here.append(d)
                except Exception:
                    # Defensive: one buggy passage should not tank the
                    # whole report. Silently drop the passage's dates.
                    dates_here = []

            currency_here: list[Any] = []
            if _extract_currency is not None and text:
                try:
                    currency_here = list(_extract_currency(text))
                except Exception:
                    currency_here = []

            entities_here: list[str] = []
            if text:
                seen_in_item: set[str] = set()
                for m in _PROPER_NOUN_RE.finditer(text):
                    candidate = m.group(1).strip()
                    # Skip if the first token is a stopword (weekdays,
                    # month names, sentence openers) or the whole
                    # candidate is too short after strip.
                    if not candidate:
                        continue
                    first_tok = candidate.split()[0].lower()
                    if first_tok in _STOPWORDS:
                        continue
                    key = candidate.lower()
                    if key in seen_in_item:
                        continue
                    seen_in_item.add(key)
                    entities_here.append(candidate)

            live_items.append({
                "idx": idx,
                "text": text,
                "source": source,
                "dates": dates_here,
                "entities": entities_here,
                "currency": currency_here,
            })

        if tombstoned:
            warnings.append(
                f"Skipped {tombstoned} tombstoned pattern"
                f"{'s' if tombstoned != 1 else ''} from coverage analysis."
            )
        metadata["tombstoned_skipped"] = tombstoned
        metadata["live_count"] = len(live_items)

        cart_name = cart.cart_name
        if cart.pattern0:
            cart_name = str(cart.pattern0.get("cart_name") or cart_name)

        # ---- Empty-cart short circuit -----------------------------------
        # Even in the empty case we still render all 6 section headers so
        # the report shape is stable — the design brief calls out
        # "sections render 'No items found' gracefully" as a requirement.
        if not live_items:
            markdown = _render_empty_report(cart_name, tombstoned)
            metadata["empty"] = True
            return ReportOutput(
                markdown=markdown,
                metadata=metadata,
                warnings=warnings,
            )

        # ---- Section 1: Coverage Overview ------------------------------
        n_live = len(live_items)
        # Group sources for the overview (unique labels count).
        unique_sources = sorted({
            item["source"] for item in live_items if item["source"]
        })
        with_dates = sum(1 for i in live_items if i["dates"])
        with_entities = sum(1 for i in live_items if i["entities"])
        with_currency = sum(1 for i in live_items if i["currency"])
        all_dates = sorted({d for i in live_items for d in i["dates"]})
        date_min = all_dates[0] if all_dates else None
        date_max = all_dates[-1] if all_dates else None
        metadata["overview"] = {
            "live_count": n_live,
            "unique_sources": len(unique_sources),
            "date_range": {
                "min": date_min.isoformat() if date_min else None,
                "max": date_max.isoformat() if date_max else None,
            },
            "with_dates_pct": _pct(with_dates, n_live),
            "with_entities_pct": _pct(with_entities, n_live),
            "with_currency_pct": _pct(with_currency, n_live),
        }

        # ---- Section 2: Underrepresented Themes ------------------------
        # Per-theme item counts, using set membership so a bigram
        # appearing 5x inside one passage counts as 1 item, not 5.
        theme_to_items: dict[str, set[int]] = defaultdict(set)
        for i, item in enumerate(live_items):
            prefix_text = item["text"][:_THEME_PASSAGE_PREFIX]
            for bg in set(_bigrams(prefix_text)):
                theme_to_items[bg].add(i)
        # Rank candidate themes by descending item count; take top N as
        # the candidate pool so single-mention noise doesn't dominate.
        ranked = sorted(
            theme_to_items.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )[:_THEME_CANDIDATE_POOL]
        underrep_themes: list[tuple[str, list[int]]] = [
            (theme, sorted(items))
            for theme, items in ranked
            if 0 < len(items) < min_theme_items
        ]
        # Cap the display so we don't dump a wall of noisy 1-item themes.
        capped_underrep = underrep_themes[:_UNDERREP_DISPLAY_CAP]
        metadata["underrepresented_themes"] = [
            {"theme": theme, "item_count": len(items)}
            for theme, items in capped_underrep
        ]

        # ---- Section 3: Orphan Entities --------------------------------
        # An entity is an orphan if it appears in exactly ONE live item.
        # Case-insensitive dedup keeps "Sysco" and "SYSCO" from being
        # counted as two different entities.
        entity_to_items: dict[str, set[int]] = defaultdict(set)
        entity_display: dict[str, str] = {}
        for i, item in enumerate(live_items):
            for ent in item["entities"]:
                key = ent.lower()
                entity_to_items[key].add(i)
                # First-seen surface form wins as the display label.
                entity_display.setdefault(key, ent)
        orphans: list[tuple[str, int]] = [
            (entity_display[key], next(iter(items)))
            for key, items in entity_to_items.items()
            if len(items) == 1
        ]
        # Sort alphabetically by display label for stable output. Cap to
        # the user-supplied max.
        orphans.sort(key=lambda pair: pair[0].lower())
        total_orphans = len(orphans)
        orphans = orphans[:max_orphan_entities]
        metadata["orphan_entities"] = {
            "total": total_orphans,
            "displayed": len(orphans),
        }

        # ---- Section 4: Source Coverage --------------------------------
        source_counts: Counter[str] = Counter()
        for item in live_items:
            source_counts[item["source"] or "(no source)"] += 1
        # Sort descending by count so the biggest contributors show
        # first; alpha secondary key keeps ties stable.
        source_rows = sorted(
            source_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        under_utilized_sources = [
            (src, count) for src, count in source_rows
            if count < source_coverage_min
        ]
        metadata["source_coverage"] = {
            "unique_sources": len(source_counts),
            "under_utilized_count": len(under_utilized_sources),
        }

        # ---- Section 5: Time Gaps --------------------------------------
        # Only meaningful if we have >= 2 dates. all_dates is already
        # sorted + deduplicated; find gaps between consecutive entries.
        time_gaps: list[tuple[date, date, int]] = []
        if len(all_dates) >= 2:
            for prev, nxt in zip(all_dates, all_dates[1:]):
                delta = (nxt - prev).days
                if delta > gap_threshold_days:
                    time_gaps.append((prev, nxt, delta))
        metadata["time_gaps"] = [
            {
                "from": prev.isoformat(),
                "to": nxt.isoformat(),
                "days": days,
            }
            for prev, nxt, days in time_gaps
        ]

        # ---- Section 6: Context-Poor Items -----------------------------
        # Item is "context-poor" iff it has NO extracted dates AND NO
        # extracted entities AND NO extracted currency.
        context_poor = [
            item for item in live_items
            if not item["dates"]
            and not item["entities"]
            and not item["currency"]
        ]
        metadata["context_poor"] = {
            "total": len(context_poor),
            "sampled": min(len(context_poor), _CONTEXT_POOR_SAMPLE_CAP),
        }

        # ---- Assemble markdown -----------------------------------------
        lines: list[str] = []
        lines.append(f"# Coverage Report: {cart_name}")
        lines.append("")

        # -- Section 1 --
        lines.append("## Coverage Overview")
        lines.append("")
        lines.append(
            f"- **Items**: {n_live} live pattern"
            f"{'s' if n_live != 1 else ''}"
            + (f" ({tombstoned} tombstoned, skipped)" if tombstoned else "")
        )
        lines.append(
            f"- **Sources**: {len(unique_sources)} unique"
            f"{' file' if len(unique_sources) == 1 else ' files'}"
        )
        if date_min and date_max:
            lines.append(
                f"- **Date range**: {date_min.isoformat()} → "
                f"{date_max.isoformat()}"
            )
        else:
            lines.append("- **Date range**: (no dates extracted)")
        lines.append(
            f"- **Extraction density**: "
            f"dates {_pct(with_dates, n_live)} · "
            f"entities {_pct(with_entities, n_live)} · "
            f"currency {_pct(with_currency, n_live)}"
        )
        lines.append("")

        # -- Section 2 --
        lines.append("## Underrepresented Themes")
        lines.append("")
        if not capped_underrep:
            lines.append(
                f"_All top themes have at least {min_theme_items} items — "
                f"no gaps detected._"
            )
        else:
            for theme, items in capped_underrep:
                sample_idx = items[0]
                sample = live_items[sample_idx]
                # 2026-07-13 (Phase A): source-file reference is now a
                # markdown link the frontend intercepts. Prior code
                # wrapped the source name in decorative [ ] brackets;
                # source_link() emits its own [display](vps://…) syntax.
                lines.append(
                    f"- **{theme}** — {len(items)} item"
                    f"{'s' if len(items) != 1 else ''} "
                    f"(sample: pattern #{sample['idx']} "
                    f"{source_link(sample['source'])}: "
                    f"{_snippet(sample['text'])})"
                )
            if len(underrep_themes) > _UNDERREP_DISPLAY_CAP:
                lines.append("")
                lines.append(
                    f"_… and {len(underrep_themes) - _UNDERREP_DISPLAY_CAP}"
                    f" more underrepresented themes not shown._"
                )
        lines.append("")

        # -- Section 3 --
        lines.append("## Orphan Entities")
        lines.append("")
        if not orphans:
            lines.append(
                "_No orphan entities found — every proper noun surfaces "
                "in at least 2 items._"
            )
        else:
            for ent, item_idx in orphans:
                item = live_items[item_idx]
                # Phase A: source is now a real markdown link (see
                # source_link.py). Decorative [ ] wrapper dropped —
                # source_link() emits its own [display](vps://…) syntax.
                lines.append(
                    f"- **{ent}** — pattern #{item['idx']} "
                    f"{source_link(item['source'])}"
                )
            if total_orphans > len(orphans):
                lines.append("")
                lines.append(
                    f"_… and {total_orphans - len(orphans)} more orphan "
                    f"entities not shown (raise `max_orphan_entities` "
                    f"to see them)._"
                )
        lines.append("")

        # -- Section 4 --
        lines.append("## Source Coverage")
        lines.append("")
        if not source_rows:
            lines.append("_No sources identified._")
        else:
            lines.append("| Source | Items | Status |")
            lines.append("|---|---|---|")
            for src, count in source_rows:
                status = (
                    "under-utilized" if count < source_coverage_min else "OK"
                )
                lines.append(f"| {source_link(src)} | {count} | {status} |")
            if under_utilized_sources:
                n_under = len(under_utilized_sources)
                lines.append("")
                lines.append(
                    f"_{n_under} source"
                    f"{'s' if n_under != 1 else ''} "
                    f"{'contribute' if n_under != 1 else 'contributes'} "
                    f"fewer than {source_coverage_min} items._"
                )
        lines.append("")

        # -- Section 5 --
        lines.append("## Time Gaps")
        lines.append("")
        if not all_dates:
            lines.append(
                "_No dates were extracted from the cart — time-gap "
                "analysis skipped._"
            )
        elif len(all_dates) < 2:
            lines.append(
                "_Only one distinct date extracted — no gaps to measure._"
            )
        elif not time_gaps:
            lines.append(
                f"_No gaps larger than {gap_threshold_days} day"
                f"{'s' if gap_threshold_days != 1 else ''} between "
                f"consecutive extracted dates._"
            )
        else:
            for prev, nxt, days in time_gaps:
                lines.append(
                    f"- **{days} days**: {prev.isoformat()} → "
                    f"{nxt.isoformat()}"
                )
        lines.append("")

        # -- Section 6 --
        lines.append("## Context-Poor Items")
        lines.append("")
        if not context_poor:
            lines.append(
                "_Every live item has at least one extracted date, "
                "entity, or currency reference._"
            )
        else:
            n_poor = len(context_poor)
            lines.append(
                f"**{n_poor}** item"
                f"{'s' if n_poor != 1 else ''} "
                f"{'lack' if n_poor != 1 else 'lacks'} extracted dates, "
                f"entities, and currency (enrichment candidates)."
            )
            lines.append("")
            for item in context_poor[:_CONTEXT_POOR_SAMPLE_CAP]:
                # Phase A: source-file reference is now a markdown link.
                # See source_link.py for the slug convention.
                lines.append(
                    f"- pattern #{item['idx']} "
                    f"{source_link(item['source'])}: "
                    f"{_snippet(item['text'])}"
                )
            if len(context_poor) > _CONTEXT_POOR_SAMPLE_CAP:
                lines.append("")
                lines.append(
                    f"_… and {len(context_poor) - _CONTEXT_POOR_SAMPLE_CAP}"
                    f" more context-poor items not shown._"
                )

        markdown = "\n".join(lines).rstrip() + "\n"

        # ---- Source refs for the audit footer ---------------------------
        if options.include_source_refs:
            metadata["orphan_source_refs"] = [
                {"entity": ent, "pattern_idx": live_items[i]["idx"]}
                for ent, i in orphans
            ]
            metadata["context_poor_source_refs"] = [
                item["idx"] for item in context_poor[:_CONTEXT_POOR_SAMPLE_CAP]
            ]

        return ReportOutput(
            markdown=markdown,
            metadata=metadata,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Empty-cart renderer
# ---------------------------------------------------------------------------

def _render_empty_report(cart_name: str, tombstoned: int) -> str:
    """Render a valid 6-section report body for an empty (or fully
    tombstoned) cart.

    Called out explicitly by the design brief: "Empty cart doesn't
    crash — sections render 'No items found' gracefully." Keeping the
    six section headers present so downstream tooling (Wave-2 table-of-
    contents / anchor linking) doesn't need to special-case the empty
    shape.
    """
    tomb_note = (
        f" ({tombstoned} tombstoned pattern"
        f"{'s' if tombstoned != 1 else ''} skipped)"
        if tombstoned else ""
    )
    lines = [
        f"# Coverage Report: {cart_name}",
        "",
        "## Coverage Overview",
        "",
        f"No live items found in this cart{tomb_note}.",
        "",
        "## Underrepresented Themes",
        "",
        "_No items found._",
        "",
        "## Orphan Entities",
        "",
        "_No items found._",
        "",
        "## Source Coverage",
        "",
        "_No items found._",
        "",
        "## Time Gaps",
        "",
        "_No items found._",
        "",
        "## Context-Poor Items",
        "",
        "_No items found._",
    ]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["CoverageReport"]
