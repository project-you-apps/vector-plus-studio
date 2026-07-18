"""Cart Curator Agent — coverage-style recommendations for a cart.

"What's missing from this cart?" — runs a lightweight Coverage-style
analysis (unique sources, per-source counts, orphan-ish sources) and
hands the LLM a structured summary to synthesize into concrete
recommendations: "add more X", "your Y is thin", "consider Z".

Design decisions (2026-07-13 v1)
--------------------------------
- **Deterministic core + LLM narrative.** The Coverage Report engine
  runs deterministic gap analysis — we borrow the SAME shape here (source
  distribution + orphan detection) but wrap it in an LLM synthesis step
  that reads like a colleague's advice rather than a bare data dump.

- **Cites patterns.** Under-represented sources / orphan-y sources get
  their pattern indices attached to ``cited_patterns`` so a user can
  click through to see exactly which patterns the recommendation is
  built on.

- **Doesn't call Coverage Report.** We don't reuse ``CoverageReport``
  directly — that report is a heavy build with a lot of sections
  we don't need here. Instead we compute the two signals we care about
  (per-source counts, under-represented sources) inline. Cheap, focused,
  and doesn't create a runtime dependency between Agents and Reports
  beyond ``CartHandle``.

- **Tombstone-aware.** Same rule as everywhere else — hippocampus row
  byte 28, bit 0 = tombstoned. Skipped from all counts.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from api.reports.cart_reader import CartHandle
from api.reports.source_link import source_link

from .base import Agent, AgentInput, AgentOptions, AgentOutput
from .prompt import wrap_llama3_instruct
from .registry import register_agent


_HIPPO_FLAGS_OFFSET = 28
_FLAG_TOMBSTONE = 0x01

# Sources with fewer than this many patterns are flagged as
# "under-represented" — worth deepening or removing. Matches the default
# on Coverage Report's source_coverage_min.
_UNDERREPRESENTED_SOURCE_THRESHOLD = 5

# Cap on how many under-represented sources to feed to the LLM. Too many
# and the prompt drifts into a list-dump; a bounded set keeps the
# recommendations focused.
_MAX_UNDERREP_TO_CITE = 8


def _is_tombstoned(cart: CartHandle, idx: int) -> bool:
    row = cart.get_hippocampus_row(idx)
    if row is None or len(row) <= _HIPPO_FLAGS_OFFSET:
        return False
    return bool(int(row[_HIPPO_FLAGS_OFFSET]) & _FLAG_TOMBSTONE)


@register_agent
class CartCuratorAgent(Agent):
    """Cart Curator recipe — coverage-style recommendations via LLM."""

    name = "cart_curator"
    display_name = "Cart Curator"
    description = (
        "Recommendations for improving this cart — under-represented "
        "sources, thin coverage, gaps. Runs a lightweight coverage "
        "analysis + LLM synthesis on top."
    )
    llm_dependency = True

    input_schema: list[dict[str, Any]] = [
        {
            "name": "focus_area",
            "label": "Focus area (optional)",
            "type": "text",
            "required": False,
            "placeholder": "e.g. financial docs, vendor invoices, meeting notes",
            "helpText": "Optional — tell the curator what you're building this cart for.",
        },
        {
            "name": "source_min",
            "label": "Under-represented source threshold",
            "type": "number",
            "required": False,
            "default": _UNDERREPRESENTED_SOURCE_THRESHOLD,
            "helpText": "Sources with fewer than this many patterns are flagged.",
        },
    ]

    def execute(
        self,
        cart_path: str,
        inputs: AgentInput,
        options: AgentOptions,
    ) -> AgentOutput:
        from api.llm import get_llm_adapter

        cart = CartHandle(cart_path)
        warnings: list[str] = list(cart.length_warnings)

        focus_area = inputs.get_str("focus_area", "") or ""
        source_min = inputs.get_int(
            "source_min", _UNDERREPRESENTED_SOURCE_THRESHOLD,
        ) or _UNDERREPRESENTED_SOURCE_THRESHOLD
        source_min = max(1, source_min)

        if cart.count == 0:
            return AgentOutput(
                markdown=(
                    f"# Curator: {cart.cart_name}\n\n"
                    "This cart is empty — nothing to curate yet.\n"
                ),
                warnings=warnings,
                metadata={"pattern_count": 0},
            )

        # -- Deterministic pass: per-source counts + under-represented ----
        per_source_count: Counter[str] = Counter()
        per_source_first_idx: dict[str, int] = {}
        live_count = 0
        tombstoned = 0
        for idx in range(cart.count):
            if _is_tombstoned(cart, idx):
                tombstoned += 1
                continue
            live_count += 1
            src = cart.get_source(idx) or ""
            if not src:
                continue
            per_source_count[src] += 1
            per_source_first_idx.setdefault(src, idx)

        if tombstoned:
            warnings.append(
                f"Skipped {tombstoned} tombstoned pattern"
                f"{'s' if tombstoned != 1 else ''}."
            )

        unique_sources = list(per_source_count.keys())
        under_rep = [
            (src, cnt) for src, cnt in per_source_count.most_common()
            if cnt < source_min
        ]
        # Order under-represented: fewest patterns first (thinnest = most
        # actionable). Break ties by source name for deterministic output.
        under_rep.sort(key=lambda t: (t[1], t[0].lower()))
        under_rep_display = under_rep[:_MAX_UNDERREP_TO_CITE]
        cited_pattern_ids: list[int] = [
            per_source_first_idx[src] for (src, _) in under_rep_display
            if src in per_source_first_idx
        ]

        # -- Compose the structured summary the LLM sees ------------------
        summary_lines: list[str] = []
        summary_lines.append(f"Cart name: {cart.cart_name}")
        summary_lines.append(f"Live patterns: {live_count}")
        summary_lines.append(f"Unique sources: {len(unique_sources)}")
        if unique_sources:
            top_sources = per_source_count.most_common(5)
            summary_lines.append("Top sources by pattern count:")
            for src, cnt in top_sources:
                summary_lines.append(f"  - {src}: {cnt}")
        if under_rep_display:
            summary_lines.append(
                f"Under-represented sources (< {source_min} patterns):"
            )
            for src, cnt in under_rep_display:
                summary_lines.append(f"  - {src}: {cnt}")
        else:
            summary_lines.append(
                "No under-represented sources at the current threshold."
            )
        structured_summary = "\n".join(summary_lines)

        focus_line = (
            f'The user says the cart is for: "{focus_area}".\n\n'
            if focus_area else ""
        )
        system_prompt = (
            "You are a knowledge-cart curator. Given a coverage summary, "
            "produce actionable recommendations. Rules: (1) Structure "
            "output as: a one-sentence overall assessment, then 3-5 "
            "bulleted recommendations. (2) Each recommendation should be "
            "specific and actionable — 'add more X' or 'consider "
            "deepening Y' or 'these sources look orphaned'. (3) Do not "
            "invent statistics not in the summary. (4) If everything "
            "looks balanced, say so plainly and suggest ways to keep it "
            "that way rather than manufacturing problems."
        )
        user_prompt = (
            f"{focus_line}"
            f"Coverage summary:\n\n{structured_summary}\n\n"
            f"Write your recommendations now."
        )

        prompt = wrap_llama3_instruct(system_prompt, user_prompt)
        adapter = get_llm_adapter()
        result = adapter.synthesize(
            prompt,
            model_hint=options.llm_model_hint or "default",
            max_tokens=options.max_tokens,
        )
        if result.error:
            warnings.append(f"LLM call reported an error: {result.error}")
        body = result.text.strip() or (
            "(LLM returned no content. The structured summary above is "
            "the deterministic view — recommendations are best-effort.)"
        )

        # -- Assemble markdown output -------------------------------------
        lines: list[str] = []
        lines.append(f"# Curator: {cart.cart_name}")
        lines.append("")
        lines.append(
            f"**{live_count}** live pattern"
            f"{'s' if live_count != 1 else ''} across "
            f"**{len(unique_sources)}** source"
            f"{'s' if len(unique_sources) != 1 else ''}."
        )
        lines.append("")

        # LLM-synthesized recommendations block.
        lines.append("## Recommendations")
        lines.append("")
        lines.append(body)
        lines.append("")

        # Deterministic under-represented block (with vps:// source
        # links, so users can drill straight into the thin material).
        if under_rep_display and options.include_source_refs:
            lines.append("## Under-represented sources")
            lines.append("")
            for src, cnt in under_rep_display:
                link = source_link(src) if src else "(no source)"
                lines.append(
                    f"- {link} — {cnt} pattern{'s' if cnt != 1 else ''}"
                )
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"

        llm_usage = {
            "calls_made": 1,
            "tokens_used": result.tokens_used,
            "neurons_used": result.neurons_used,
            "cost_usd": result.cost_usd,
            "provider": result.provider,
            "model": result.model,
        }
        metadata = {
            "cart_name": cart.cart_name,
            "pattern_count": cart.count,
            "live_pattern_count": live_count,
            "unique_source_count": len(unique_sources),
            "under_represented_count": len(under_rep),
            "focus_area": focus_area or None,
            "source_min": source_min,
            "llm_provider": result.provider,
            "llm_model": result.model,
            "llm_elapsed_ms": result.elapsed_ms,
        }

        return AgentOutput(
            markdown=markdown,
            cited_patterns=cited_pattern_ids,
            metadata=metadata,
            warnings=warnings,
            llm_usage=llm_usage,
        )


__all__ = ["CartCuratorAgent"]
