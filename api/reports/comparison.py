"""Comparison Report — two subsets of a cart side-by-side.

Answers "Q1 invoices vs Q2 invoices", "vendor A vs vendor B",
"May mentions vs June mentions". Wave-1 has no LLM dependency.

Spec: ``docs/vps-internal/Report Types Design 2026-07-10.md`` §4.

Query strategy — wave-1 note
----------------------------
The design doc's Query Strategy calls for Membot ``memory_search`` on
each subset's query string. **Semantic search is a Wave-2 concern** —
Wave-1 lands without a hosted-Membot dependency so the reports engine
can be dispatched, wired, and demoed without waiting on the search
service.

For Wave-1 we implement subset membership as **plain substring
matching**: a passage belongs to a subset iff the query (lowercased)
appears as a substring of the passage text or its source_path
(lowercased). This is honest and functional — office managers doing
"vendor A vs vendor B" invoice comparisons routinely have the vendor
name as a literal in the passage. When Wave-2 wires Membot semantic
search, the swap point is ``_matches_subset`` below; the report body,
overlap math, and aggregate metrics stay identical.

Tombstones
----------
Skips patterns whose hippocampus ``flags`` byte (offset 28 in the H-row,
per ``api/cartridge_io.py::HIPPO_FORMAT``) has bit 0 set. This is the
membot tombstone bit — tombstoned patterns are the H-row equivalent of
soft-deletes and must not surface in reports.

Extractor dependency
--------------------
Uses ``extract_dates`` for the date-span metric and ``extract_currency``
for the total-currency metric. Both are imported **lazily** inside
``generate()`` so import order between the parallel Wave-1a dispatches
doesn't matter — if extractors haven't landed at import time we fail at
call time with a clear message instead of on module load.
"""
from __future__ import annotations

from typing import Optional

from .base import Report, ReportInput, ReportOptions, ReportOutput
from .cart_reader import CartHandle
from .registry import register_report
from .source_link import source_link


# Offset of membot's ``flags`` byte within the 64-byte hippocampus row.
# Bit 0 = tombstone. Matches ``api/cartridge_io.py::HIPPO_FORMAT``
# (fields: pattern_id I, format_version B, cartridge_type B,
# parent_ptr I, child_ptr I, sibling_ptr I, source_hash I,
# sequence_num H, timestamp I, flags B  ← this one).
_HIPPO_FLAGS_OFFSET = 28
_HIPPO_TOMBSTONE_BIT = 0x01

# Cap on how many "distinctive" bullets we emit per side. Spec §4 says
# "first 5". Keeping the number a constant so it's easy to tune later
# without hunting through render code.
_DISTINCTIVE_LIMIT = 5

# Length cap for the passage summary snippet in each distinctive bullet.
# Short enough to fit on one visible line in a preview; long enough to
# convey what the passage is about. Not spec'd — chosen for readability.
_SUMMARY_MAX_CHARS = 120


@register_report
class ComparisonReport(Report):
    """Two subsets of a cart side-by-side.

    See module docstring + Report Types Design §4.
    """

    # NB: slug matches the frontend definition in
    # ``frontend/src/reports/report-definitions.ts`` — literal ``comparison``
    # (no hyphens, no suffix). Verified 2026-07-11.
    name = "comparison"
    display_name = "Comparison"
    description = "Two subsets of a cart side-by-side."

    llm_dependency = False
    # Recurring "Q4 vs Q3 monthly comparison" is a natural scheduled
    # brief so this stays True even though it needs two query strings.
    supports_scheduling = True

    # Mirrors ``report-definitions.ts`` field order. All four inputs are
    # required text — subset_a_name / subset_a_query / subset_b_name /
    # subset_b_query. See frontend for placeholder + help strings.
    input_schema = [
        {
            "name": "subset_a_name",
            "label": "Subset A name",
            "type": "text",
            "required": True,
            "placeholder": "e.g. May invoices",
        },
        {
            "name": "subset_a_query",
            "label": "Subset A query",
            "type": "text",
            "required": True,
            "placeholder": "Membot query string that defines subset A",
        },
        {
            "name": "subset_b_name",
            "label": "Subset B name",
            "type": "text",
            "required": True,
            "placeholder": "e.g. June invoices",
        },
        {
            "name": "subset_b_query",
            "label": "Subset B query",
            "type": "text",
            "required": True,
            "placeholder": "Membot query string that defines subset B",
        },
    ]

    # -- generate ---------------------------------------------------------
    def generate(
        self,
        cart_path: str,
        inputs: ReportInput,
        options: ReportOptions,
    ) -> ReportOutput:
        # -- extractors (lazy import) -------------------------------------
        # See module docstring: extractors ship as a parallel Wave-1a
        # dispatch. Import at call time so this report is loadable even
        # if extractors haven't landed yet — the failure surfaces as a
        # helpful runtime message rather than an ImportError on module
        # import that breaks the whole registry.
        try:
            from .extractors.dates import extract_dates
            from .extractors.currency import extract_currency
        except ImportError as exc:  # pragma: no cover — soft-fail path
            raise ImportError(
                "ComparisonReport requires api.reports.extractors "
                "(dates + currency). Ensure the Wave-1a extractors "
                f"dispatch has landed. Underlying error: {exc}"
            ) from exc

        # -- read inputs ---------------------------------------------------
        subset_a_name = inputs.get_str("subset_a_name") or "Subset A"
        subset_a_query = inputs.get_str("subset_a_query") or ""
        subset_b_name = inputs.get_str("subset_b_name") or "Subset B"
        subset_b_query = inputs.get_str("subset_b_query") or ""

        # Required-input guard. Empty query is legal only via the "fail
        # loud" path further down (we still surface a helpful markdown
        # body for an empty subset rather than raising).
        if not subset_a_query or not subset_b_query:
            return self._empty_query_response(
                subset_a_name, subset_a_query,
                subset_b_name, subset_b_query,
                cart_path,
            )

        # -- open cart -----------------------------------------------------
        cart = CartHandle(cart_path)
        warnings: list[str] = list(cart.length_warnings)

        # -- walk patterns, build subsets ---------------------------------
        # Iterate ONCE and dispatch each passage into both filters (a
        # passage can belong to both — that's the overlap set). Skip
        # tombstoned patterns so soft-deleted content doesn't count.
        set_a: set[int] = set()
        set_b: set[int] = set()
        tombstone_count = 0

        query_a_lc = subset_a_query.lower()
        query_b_lc = subset_b_query.lower()

        for idx, text, source in cart.iter_passages():
            if _is_tombstoned(cart, idx):
                tombstone_count += 1
                continue
            text_lc = text.lower()
            source_lc = source.lower()
            if _matches_subset(query_a_lc, text_lc, source_lc):
                set_a.add(idx)
            if _matches_subset(query_b_lc, text_lc, source_lc):
                set_b.add(idx)

        # -- edge case: empty subset --------------------------------------
        # Fail loud (spec §4). A misleading empty comparison is worse than
        # a clear "no match — check your query" markdown.
        if not set_a or not set_b:
            return self._empty_subset_response(
                cart, subset_a_name, subset_a_query, set_a,
                subset_b_name, subset_b_query, set_b,
                tombstone_count, warnings,
            )

        overlap = set_a & set_b
        only_a = set_a - overlap
        only_b = set_b - overlap

        # -- aggregate metrics per side -----------------------------------
        metrics_a = _aggregate_metrics(
            cart, set_a, extract_dates, extract_currency,
        )
        metrics_b = _aggregate_metrics(
            cart, set_b, extract_dates, extract_currency,
        )

        # -- render markdown ----------------------------------------------
        cart_display_name = _cart_display_name(cart)

        md_parts: list[str] = []
        md_parts.append(
            f"# Comparison: {subset_a_name} vs {subset_b_name} "
            f"— {cart_display_name}"
        )
        md_parts.append("")

        # -- Summary table ------------------------------------------------
        md_parts.append("## Summary")
        md_parts.append("")
        md_parts.append(f"| Metric | {subset_a_name} | {subset_b_name} | Δ |")
        md_parts.append("|---|---|---|---|")
        md_parts.append(
            _row(
                "Passages matched",
                metrics_a["count"], metrics_b["count"],
                # Larger-count field — spec §4 example shows "+11%".
                _delta_pct_or_abs(metrics_a["count"], metrics_b["count"]),
            )
        )
        md_parts.append(
            _row(
                "Unique sources",
                metrics_a["unique_sources"], metrics_b["unique_sources"],
                # Small-count field — spec §4 example ("Unique vendors
                # | 3 | 4 | +1") shows absolute delta. force_abs=True.
                _delta_pct_or_abs(
                    metrics_a["unique_sources"], metrics_b["unique_sources"],
                    force_abs=True,
                ),
            )
        )
        md_parts.append(
            _row(
                "Date span",
                metrics_a["date_span"], metrics_b["date_span"],
                _date_span_delta(metrics_a, metrics_b),
            )
        )
        md_parts.append(
            _row(
                "Total currency",
                metrics_a["currency_display"], metrics_b["currency_display"],
                _currency_delta(metrics_a, metrics_b),
            )
        )
        md_parts.append("")

        # -- Overlap section (skip when empty per spec §4) ----------------
        if overlap:
            if not only_a and not only_b:
                # Full overlap — subsets are identical (spec §4 edge case).
                md_parts.append("## Overlap")
                md_parts.append("")
                md_parts.append(
                    f"{len(overlap)} patterns matched both subsets — "
                    "no meaningful distinction, subsets are identical."
                )
                md_parts.append("")
            else:
                md_parts.append("## Overlap")
                md_parts.append("")
                md_parts.append(
                    f"{len(overlap)} patterns matched both subsets."
                )
                md_parts.append("")
        # else: spec §4 says "no overlap → skip the Overlap section." Done.

        # -- Distinctive bullets ------------------------------------------
        # Full-overlap case: distinctive lists are empty by definition, so
        # skip both distinctive sections when only_a and only_b are empty.
        if only_a or only_b:
            md_parts.append(
                f"## Distinctive to {subset_a_name} (first {_DISTINCTIVE_LIMIT})"
            )
            md_parts.append("")
            md_parts.extend(
                _distinctive_bullets(cart, sorted(only_a)[:_DISTINCTIVE_LIMIT])
            )
            md_parts.append("")

            md_parts.append(
                f"## Distinctive to {subset_b_name} (first {_DISTINCTIVE_LIMIT})"
            )
            md_parts.append("")
            md_parts.extend(
                _distinctive_bullets(cart, sorted(only_b)[:_DISTINCTIVE_LIMIT])
            )
            md_parts.append("")

        # -- metadata (audit surface) -------------------------------------
        metadata: dict = {
            "subset_a": {
                "name": subset_a_name,
                "query": subset_a_query,
                "count": metrics_a["count"],
                "unique_sources": metrics_a["unique_sources"],
                "total_currency_usd": str(metrics_a["currency_total_usd"]),
            },
            "subset_b": {
                "name": subset_b_name,
                "query": subset_b_query,
                "count": metrics_b["count"],
                "unique_sources": metrics_b["unique_sources"],
                "total_currency_usd": str(metrics_b["currency_total_usd"]),
            },
            "overlap_count": len(overlap),
            "tombstoned_skipped": tombstone_count,
            "match_strategy": "substring",  # swap to "membot_semantic" in Wave-2
        }
        if options.include_source_refs:
            metadata["subset_a"]["pattern_indices"] = sorted(set_a)
            metadata["subset_b"]["pattern_indices"] = sorted(set_b)
            metadata["overlap_pattern_indices"] = sorted(overlap)

        return ReportOutput(
            markdown="\n".join(md_parts).rstrip() + "\n",
            metadata=metadata,
            warnings=warnings,
        )

    # -- soft-fail response builders ---------------------------------------
    def _empty_query_response(
        self,
        subset_a_name: str, subset_a_query: str,
        subset_b_name: str, subset_b_query: str,
        cart_path: str,
    ) -> ReportOutput:
        """Both queries must be non-empty — spec §4 "fail loud"."""
        missing = []
        if not subset_a_query:
            missing.append(f"{subset_a_name!r} (subset_a_query)")
        if not subset_b_query:
            missing.append(f"{subset_b_name!r} (subset_b_query)")
        msg = (
            f"# Comparison — no query supplied\n\n"
            f"Cannot compare without a query string for: "
            f"{', '.join(missing)}. Supply a non-empty query for each side."
        )
        return ReportOutput(
            markdown=msg,
            warnings=[
                "Empty subset query — comparison requires both subset_a_query "
                "and subset_b_query to be non-empty."
            ],
            metadata={"cart_path": cart_path, "error": "empty_query"},
        )

    def _empty_subset_response(
        self,
        cart: CartHandle,
        subset_a_name: str, subset_a_query: str, set_a: set,
        subset_b_name: str, subset_b_query: str, set_b: set,
        tombstone_count: int, warnings: list[str],
    ) -> ReportOutput:
        """Spec §4 edge case: empty subset → don't produce a misleading
        empty comparison; return a helpful markdown explaining what went
        wrong."""
        empties = []
        if not set_a:
            empties.append(f"{subset_a_name!r} (query: {subset_a_query!r})")
        if not set_b:
            empties.append(f"{subset_b_name!r} (query: {subset_b_query!r})")

        cart_display_name = _cart_display_name(cart)

        body = [
            f"# Comparison: {subset_a_name} vs {subset_b_name} "
            f"— {cart_display_name}",
            "",
            "## No comparison produced",
            "",
            "One or both subsets matched zero passages, so a side-by-side "
            "comparison would be misleading.",
            "",
            "**Empty subset(s)**:",
        ]
        for e in empties:
            body.append(f"- {e}")
        body += [
            "",
            f"- {subset_a_name}: {len(set_a)} matched",
            f"- {subset_b_name}: {len(set_b)} matched",
            "",
            "**Try**:",
            "- Broaden or reword the query for the empty side",
            "- Confirm the cart contains the terms you expect "
            "(this report uses substring matching in wave 1 — "
            "semantic search lands in wave 2).",
        ]
        warnings.append(
            "Empty subset(s) prevented a comparison: "
            + ", ".join(empties)
        )
        return ReportOutput(
            markdown="\n".join(body) + "\n",
            warnings=warnings,
            metadata={
                "cart_path": cart.cart_path,
                "error": "empty_subset",
                "subset_a": {
                    "name": subset_a_name, "query": subset_a_query,
                    "count": len(set_a),
                },
                "subset_b": {
                    "name": subset_b_name, "query": subset_b_query,
                    "count": len(set_b),
                },
                "tombstoned_skipped": tombstone_count,
                "match_strategy": "substring",
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_tombstoned(cart: CartHandle, idx: int) -> bool:
    """True iff the hippocampus flags byte at bit 0 is set for ``idx``.

    Falls back to False when the cart has no hippocampus array — legacy
    /brain-only carts pre-date the tombstone convention so we treat them
    as fully live.
    """
    row = cart.get_hippocampus_row(idx)
    if row is None or len(row) <= _HIPPO_FLAGS_OFFSET:
        return False
    return bool(int(row[_HIPPO_FLAGS_OFFSET]) & _HIPPO_TOMBSTONE_BIT)


def _matches_subset(query_lc: str, text_lc: str, source_lc: str) -> bool:
    """Wave-1 subset membership: pure substring containment.

    A passage is in the subset iff ``query_lc`` is a substring of the
    passage text OR its source_path. This is the honest-and-functional
    wave-1 replacement for Membot semantic search; wave-2 swaps this out
    for real embedding-cosine search without touching the report body.
    """
    if not query_lc:
        return False
    return (query_lc in text_lc) or (query_lc in source_lc)


def _aggregate_metrics(
    cart: CartHandle,
    idx_set: set[int],
    extract_dates,           # callable: (text) -> list[DateExtraction]
    extract_currency,        # callable: (text, patterns=None) -> list[MoneyExtraction]
) -> dict:
    """Per-side aggregate metrics for the Summary table.

    Returned dict shape (all keys always present):

    - ``count``: int passage count
    - ``unique_sources``: int unique source-path count
    - ``date_min`` / ``date_max``: ``datetime.date`` or None
    - ``date_span``: human-readable string ("2026-05-01 → 2026-06-30")
      or "—" when no dates found
    - ``currency_total_usd``: :class:`Decimal` USD total across passages
      (non-USD amounts are skipped — mixed-currency handling is a Wave-2
      concern; matches Financial Rollup §6 edge case)
    - ``currency_display``: formatted "$X,XXX.YY" or "—"
    """
    from decimal import Decimal

    count = len(idx_set)
    sources: set[str] = set()
    date_min = None
    date_max = None
    usd_total = Decimal("0")

    for idx in idx_set:
        text = cart.get_passage(idx)
        src = cart.get_source(idx)
        if src:
            sources.add(src)

        # Dates: scan passage text AND source path (Sysco-style filenames
        # embed YYYYMMDD which the compact-format extractor picks up).
        combined = text + "\n" + src if src else text
        for hit in extract_dates(combined):
            if date_min is None or hit.date < date_min:
                date_min = hit.date
            if date_max is None or hit.date > date_max:
                date_max = hit.date

        # Currency: sum USD-denominated amounts, treating parenthesized
        # (negative) amounts per the extractor's is_negative flag. Non-USD
        # is skipped rather than merged (spec §6 "don't merge mixed
        # currencies silently").
        for money in extract_currency(text):
            if money.currency != "USD":
                continue
            if money.is_negative:
                usd_total -= money.value
            else:
                usd_total += money.value

    return {
        "count": count,
        "unique_sources": len(sources),
        "date_min": date_min,
        "date_max": date_max,
        "date_span": _fmt_date_span(date_min, date_max),
        "currency_total_usd": usd_total,
        "currency_display": _fmt_currency(usd_total),
    }


def _fmt_date_span(date_min, date_max) -> str:
    """Render "YYYY-MM-DD → YYYY-MM-DD" or a single date, or em-dash."""
    if date_min is None or date_max is None:
        return "—"
    if date_min == date_max:
        return date_min.isoformat()
    return f"{date_min.isoformat()} → {date_max.isoformat()}"


def _fmt_currency(total) -> str:
    """Render a Decimal USD total as ``$X,XXX.YY``. Zero renders as em-dash
    so the Summary table doesn't scream "$0.00" for text-only carts."""
    from decimal import Decimal
    if total == Decimal("0"):
        return "—"
    # Normalize to 2 decimal places for display; keep the full precision
    # in metadata for downstream consumption.
    sign = "-" if total < 0 else ""
    magnitude = abs(total)
    return f"{sign}${magnitude:,.2f}"


def _cart_display_name(cart: CartHandle) -> str:
    """Prefer Pattern-0's ``cart_name`` if present, otherwise filename stem.

    Matches the discipline noted in
    ``CartHandle.cart_name``'s docstring: filename stem is a fallback for
    when Pattern-0 metadata is absent or binary-encoded.
    """
    if cart.pattern0 and isinstance(cart.pattern0, dict):
        pn = cart.pattern0.get("cart_name")
        if isinstance(pn, str) and pn.strip():
            return pn.strip()
    return cart.cart_name


def _delta_pct_or_abs(a, b, force_abs: bool = False) -> str:
    """Signed percentage change B vs A, or absolute delta when % is
    meaningless (either side zero) or the caller wants absolute
    (``force_abs=True``, used for small-count fields per spec §4's
    "Unique vendors | 3 | 4 | +1" example).

    Per spec §4: signed pct if both sides > 0; otherwise abs delta.
    Format like "+11%", "-3.2%", "+1".
    """
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return "—"

    if not force_abs and af > 0 and bf > 0:
        pct = (bf - af) / af * 100.0
        return _fmt_signed_pct(pct)

    # Absolute delta (works for counts, zeros, and forced small-count).
    delta = bf - af
    if delta == int(delta):
        return f"{'+' if delta >= 0 else ''}{int(delta)}"
    return f"{'+' if delta >= 0 else ''}{delta:.1f}"


def _fmt_signed_pct(pct: float) -> str:
    """Format a percentage with a sign, dropping trailing .0 on whole
    numbers (matches spec examples: +11%, -3.2%)."""
    sign = "+" if pct >= 0 else ""
    if abs(pct) >= 10 and abs(pct - round(pct)) < 0.05:
        return f"{sign}{int(round(pct))}%"
    return f"{sign}{pct:.1f}%"


def _date_span_delta(metrics_a: dict, metrics_b: dict) -> str:
    """Delta for the Date span row.

    Percentages don't apply to date ranges — surface a span-days
    difference instead (positive means B covers a longer window).
    """
    a_min, a_max = metrics_a["date_min"], metrics_a["date_max"]
    b_min, b_max = metrics_b["date_min"], metrics_b["date_max"]
    if a_min is None or a_max is None or b_min is None or b_max is None:
        return "—"
    a_days = (a_max - a_min).days
    b_days = (b_max - b_min).days
    delta = b_days - a_days
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta}d"


def _currency_delta(metrics_a: dict, metrics_b: dict) -> str:
    """Delta for the Total currency row — percent when both sides > 0."""
    from decimal import Decimal
    a = metrics_a["currency_total_usd"]
    b = metrics_b["currency_total_usd"]
    if a == Decimal("0") and b == Decimal("0"):
        return "—"
    if a > 0 and b > 0:
        pct = float((b - a) / a) * 100.0
        return _fmt_signed_pct(pct)
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{sign}${abs(delta):,.2f}" if delta < 0 else f"{sign}${delta:,.2f}"


def _row(label: str, a, b, delta: str) -> str:
    """Assemble one Summary-table row."""
    return f"| {label} | {a} | {b} | {delta} |"


def _distinctive_bullets(cart: CartHandle, indices: list[int]) -> list[str]:
    """Render bullet lines for the "Distinctive to X" section.

    Each bullet is ``- {source}: {short_summary}``. When indices is empty
    (full overlap edge case, filtered by caller) we still emit a
    placeholder so the section isn't visually empty.
    """
    if not indices:
        return ["- (none)"]
    bullets = []
    for idx in indices:
        raw_src = cart.get_source(idx)
        # 2026-07-13 (Phase A): source names emit as vps://source/{slug}
        # links so the frontend can drill down. When the cart has no
        # source for a pattern, fall back to the bare "pattern #N" text
        # (no link — there's no source to focus on).
        src_display = source_link(raw_src) if raw_src else f"pattern #{idx}"
        summary = _short_summary(cart.get_passage(idx))
        bullets.append(f"- {src_display}: {summary}")
    return bullets


def _short_summary(text: str) -> str:
    """First-line-ish excerpt of a passage, capped at _SUMMARY_MAX_CHARS.

    Skips the Cart-Builder label line (first line of GUI-built cart
    passages is often ``<filename>``) by preferring the first non-empty
    subsequent line when the passage has multiple lines. Falls back to
    the whole first line when there's only one.
    """
    if not text:
        return "(empty passage)"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "(empty passage)"
    # If the first line looks like a bare filename or filename+part
    # (Cart Builder GUI convention), prefer the second line for the
    # human-facing summary.
    first = lines[0]
    picked = first
    if len(lines) > 1 and _looks_like_filename_label(first):
        picked = lines[1]
    if len(picked) > _SUMMARY_MAX_CHARS:
        picked = picked[: _SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return picked


def _looks_like_filename_label(line: str) -> bool:
    """Heuristic: true if the line reads as a bare Cart-Builder label
    line (short, no spaces except in "(part N/M)")."""
    if len(line) > 120:
        return False
    if line.startswith("[[["):
        # e.g. "[[[Timestamp: ...]]]" — treat as label-ish so the summary
        # doesn't lead with a bare timestamp.
        return True
    # Filename-ish: no whitespace, or ends with a common extension.
    lower = line.lower()
    exts = (".md", ".txt", ".pdf", ".docx", ".html", ".json", ".csv", ".log")
    return lower.endswith(exts) or (" " not in line and "." in line)


__all__ = ["ComparisonReport"]
