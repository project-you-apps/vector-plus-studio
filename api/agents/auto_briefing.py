"""Auto-Briefing Agent — daily briefing over a cart.

"Give me a briefing on this cart" — surfaces the shape of the cart
(size, spread of sources, recent-ish material) and an LLM-synthesized
narrative that reads like a good analyst's day-one summary.

Design decisions (2026-07-13 v1)
--------------------------------
- **One-shot.** No scheduling infra in v1; the scheduled variant is a
  v1.5 dispatch (design doc). This agent produces a fresh briefing every
  time it's run — differentiation from "yesterday's briefing" requires
  persistence we don't have yet.

- **No citations.** Auto Briefing summarizes the whole cart rather than
  answering a specific question, so per-pattern citations would be
  noise. ``cited_patterns`` returned empty by design. Cart Curator's
  recommendations DO cite specific patterns; that's the distinction.

- **Focus input optional.** Users can supply a ``focus`` string (e.g.
  "compliance", "new hires") to narrow the briefing. Empty = broad
  overview. When populated, focus is retrieved against and folded into
  the prompt as a "pay special attention to" hint.

- **Sample budget.** We sample the first N passages (bounded by
  options.max_context_patterns * 2) and feed their prefixes to the LLM.
  Larger carts get a wider sample than smaller ones without blowing the
  prompt budget.
"""
from __future__ import annotations

from typing import Any

from api.reports.cart_reader import CartHandle

from .base import Agent, AgentInput, AgentOptions, AgentOutput
from .prompt import wrap_llama3_instruct
from .registry import register_agent
from .retrieval import retrieve_top_patterns, format_context_block


# Cap on how many patterns feed the LLM prompt. Larger than Q&A's
# retrieval size because a briefing wants breadth not depth.
_BRIEFING_SAMPLE_CAP_MULTIPLIER = 2


@register_agent
class AutoBriefingAgent(Agent):
    """Auto-briefing recipe — one-shot cart briefing via LLM synthesis."""

    name = "auto_briefing"
    display_name = "Auto-Briefing"
    description = (
        "Daily briefing on this cart — recent additions, key themes, "
        "notable material. LLM-synthesized narrative summary."
    )
    llm_dependency = True

    input_schema: list[dict[str, Any]] = [
        {
            "name": "focus",
            "label": "Focus (optional)",
            "type": "text",
            "required": False,
            "placeholder": "e.g. compliance, new hires, product launches",
            "helpText": "Optional — narrow the briefing to a specific topic.",
        },
        {
            "name": "tone",
            "label": "Tone",
            "type": "select",
            "required": True,
            "default": "executive",
            "options": ["executive", "casual", "technical"],
            "helpText": "Voice register for the briefing narrative.",
        },
    ]

    def execute(
        self,
        cart_path: str,
        inputs: AgentInput,
        options: AgentOptions,
    ) -> AgentOutput:
        # Lazy import — matches the Wave-1 report convention. Keeps the
        # module importable without the LLM adapter being configured
        # (registry lookups don't trigger LLM boot).
        from api.llm import get_llm_adapter

        cart = CartHandle(cart_path)
        warnings: list[str] = list(cart.length_warnings)

        focus = inputs.get_str("focus", "") or ""
        tone = inputs.get_str("tone", "executive") or "executive"

        if cart.count == 0:
            return AgentOutput(
                markdown=f"# Briefing: {cart.cart_name}\n\nThis cart is empty.\n",
                warnings=warnings,
                metadata={"pattern_count": 0},
            )

        sample_n = max(
            options.max_context_patterns,
            options.max_context_patterns * _BRIEFING_SAMPLE_CAP_MULTIPLIER,
        )
        # Empty query = first-N sample path. Focus query = ranked retrieval.
        patterns = retrieve_top_patterns(cart, focus, sample_n)
        if not patterns:
            warnings.append(
                "No live patterns matched the focus query — falling back to "
                "an unfocused sample of the first patterns."
            )
            patterns = retrieve_top_patterns(cart, "", sample_n)

        context_block = format_context_block(patterns)
        cart_name = cart.cart_name
        p0 = cart.pattern0 or {}
        p0_description = str(p0.get("description") or "").strip()

        system_prompt = (
            "You are a briefing analyst. Given a set of numbered passages "
            "from a knowledge cart, produce a concise briefing memo. "
            "Structure: (1) a two-sentence overview, (2) 3-5 bulleted "
            "themes with one-line context each, (3) one closing 'what to "
            "watch' recommendation. Never invent facts not in the "
            "passages. Never say 'I cannot' — if the material is thin, "
            f"say the briefing is preliminary. Voice register: {tone}."
        )
        focus_line = (
            f'Focus this briefing on: "{focus}".\n\n' if focus else ""
        )
        desc_line = (
            f'Cart description (from Pattern-0): "{p0_description}".\n\n'
            if p0_description else ""
        )
        user_prompt = (
            f"Cart name: {cart_name}\n"
            f"Live patterns sampled: {len(patterns)} of {cart.count} total.\n\n"
            f"{focus_line}"
            f"{desc_line}"
            f"Passages:\n\n{context_block}\n\n"
            f"Write the briefing now."
        )

        prompt = wrap_llama3_instruct(system_prompt, user_prompt)
        adapter = get_llm_adapter()
        result = adapter.synthesize(
            prompt,
            model_hint=options.llm_model_hint or "default",
            max_tokens=options.max_tokens,
        )

        if result.error:
            warnings.append(
                f"LLM call reported an error: {result.error}"
            )
        body_text = result.text.strip() or (
            "(LLM returned no content. This can happen when the provider "
            "throttles or the model produced only stop tokens. Try again.)"
        )

        # Compose markdown — heading + body. Auto Briefing doesn't cite
        # specific patterns (design decision above) so no source links.
        markdown = f"# Briefing: {cart_name}\n\n{body_text}\n"

        llm_usage = {
            "calls_made": 1,
            "tokens_used": result.tokens_used,
            "neurons_used": result.neurons_used,
            "cost_usd": result.cost_usd,
            "provider": result.provider,
            "model": result.model,
        }
        metadata = {
            "cart_name": cart_name,
            "pattern_count": cart.count,
            "patterns_sampled": len(patterns),
            "retrieved_source_count": len({p.source for p in patterns if p.source}),
            "focus": focus or None,
            "tone": tone,
            "llm_provider": result.provider,
            "llm_model": result.model,
            "llm_elapsed_ms": result.elapsed_ms,
        }

        return AgentOutput(
            markdown=markdown,
            cited_patterns=[],  # briefing = no citations by design
            metadata=metadata,
            warnings=warnings,
            llm_usage=llm_usage,
        )


__all__ = ["AutoBriefingAgent"]
