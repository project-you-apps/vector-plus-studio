"""Q&A Agent — natural-language question over a cart.

The "chatbot over your cart" — but scoped. User types a question, agent
retrieves the top-N passages that best match, LLM synthesizes an answer
grounded in those passages, output includes ``vps://source/{slug}``
links back to the cited sources.

Design decisions (2026-07-13 v1)
--------------------------------
- **Retrieval + synthesis.** Two-step: rank passages, then prompt the
  LLM with the ranked block + the user's question. The prompt system
  role tells the model to cite sources by number; the numbered
  citations become the pattern-idx list in ``AgentOutput.cited_patterns``.

- **Cites patterns.** Unlike Auto Briefing, Q&A cites specific patterns
  it answered from. The frontend renders these as ``vps://source/{slug}``
  links so a user can click through to the underlying passage.

- **No web search fallback.** If the cart doesn't have the answer, we
  don't reach out — we say so. That's the whole promise of "cart as
  substrate": answers are grounded in your own memory. Web-fetch is a
  v1.5 agent, not a v1 fallback here.
"""
from __future__ import annotations

from typing import Any

from api.reports.cart_reader import CartHandle
from api.reports.source_link import source_link

from .base import Agent, AgentInput, AgentOptions, AgentOutput
from .prompt import wrap_llama3_instruct
from .registry import register_agent
from .retrieval import retrieve_top_patterns, format_context_block


@register_agent
class QAAgent(Agent):
    """Free-form Q&A recipe — natural-language question over a cart."""

    name = "qa"
    display_name = "Q&A"
    description = (
        "Ask a natural-language question. The agent retrieves the "
        "most relevant passages and synthesizes an answer with citations."
    )
    llm_dependency = True

    input_schema: list[dict[str, Any]] = [
        {
            "name": "question",
            "label": "Your question",
            "type": "textarea",
            "required": True,
            "placeholder": "e.g. What themes appear across the poems in this cart?",
            "helpText": "Ask anything the cart's material could answer.",
        },
        {
            "name": "answer_style",
            "label": "Answer style",
            "type": "select",
            "required": True,
            "default": "concise",
            "options": ["concise", "detailed", "bulleted"],
            "helpText": "Shape of the response.",
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

        question = inputs.get_str("question", "") or ""
        style = inputs.get_str("answer_style", "concise") or "concise"

        if not question.strip():
            raise ValueError("Q&A requires a non-empty 'question' input.")

        if cart.count == 0:
            return AgentOutput(
                markdown=(
                    f"# Q&A: {cart.cart_name}\n\n"
                    f"**Question.** {question}\n\n"
                    "This cart is empty — nothing to answer from.\n"
                ),
                warnings=warnings,
                metadata={"pattern_count": 0},
            )

        patterns = retrieve_top_patterns(
            cart, question, options.max_context_patterns,
        )
        if not patterns:
            warnings.append(
                "No passages scored above zero against the question — the "
                "answer below is unlikely to be grounded."
            )
            # Fall back to first-N so the LLM still gets substrate rather
            # than a completely empty context (which triggers hallucination).
            patterns = retrieve_top_patterns(cart, "", options.max_context_patterns)

        context_block = format_context_block(patterns)

        # Style directive folded into system prompt so the model doesn't
        # need to guess from the user prompt.
        style_directive = {
            "concise": "Answer in 2-4 sentences.",
            "detailed": "Answer in 2-3 paragraphs with concrete detail.",
            "bulleted": "Answer as a short bulleted list of key points.",
        }.get(style, "Answer in 2-4 sentences.")

        system_prompt = (
            "You are a research assistant answering questions grounded in "
            "the passages provided by the user. Rules: (1) Only use "
            "information present in the passages. (2) Cite passage "
            "numbers inline like [1] or [2] where a claim comes from a "
            "specific passage. (3) If the passages don't contain the "
            "answer, say so plainly — do not guess. "
            f"(4) Style: {style_directive}"
        )
        user_prompt = (
            f"Passages from the cart:\n\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Write your answer now, with inline [N] citations."
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

        answer_text = result.text.strip() or (
            "(LLM returned no content. Try rewording the question.)"
        )

        # Assemble the markdown output — question + answer + a Sources
        # section that renders as clickable vps://source/{slug} links so
        # users can drill down.
        lines: list[str] = []
        lines.append(f"# Q&A: {cart.cart_name}")
        lines.append("")
        lines.append(f"**Question.** {question}")
        lines.append("")
        lines.append("## Answer")
        lines.append("")
        lines.append(answer_text)
        lines.append("")

        if options.include_source_refs and patterns:
            lines.append("## Sources")
            lines.append("")
            # Emit unique sources in retrieval order — dedup so a single
            # source that showed up in multiple retrieved patterns doesn't
            # appear three times in the list.
            seen: set[str] = set()
            for i, p in enumerate(patterns, start=1):
                src = p.source or ""
                key = src.lower()
                if key in seen:
                    continue
                seen.add(key)
                link = source_link(src) if src else "(no source)"
                lines.append(f"- [{i}] {link}")
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
            "patterns_retrieved": len(patterns),
            "question": question,
            "answer_style": style,
            "llm_provider": result.provider,
            "llm_model": result.model,
            "llm_elapsed_ms": result.elapsed_ms,
        }

        return AgentOutput(
            markdown=markdown,
            cited_patterns=[p.idx for p in patterns],
            metadata=metadata,
            warnings=warnings,
            llm_usage=llm_usage,
        )


__all__ = ["QAAgent"]
