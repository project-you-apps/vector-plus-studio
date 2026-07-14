"""Free Agent — catch-all recipe for tasks that don't fit the specialized four.

Positioned LAST in the recipe grid so the specialized agents lead: when
none of the scoped recipes fit the user's task, Free Agent is the
minimum-friction fallback. Same cart-aware retrieval as Q&A, but the
prompt template drops question-framing and treats the input as a
general task/prompt.

Design decisions (2026-07-14 v1, baseball pun during All-Star week —
name may change if it doesn't stick)
--------------------------------------------------------------------
- **Cart-aware retrieval reused.** Same ``retrieve_top_patterns`` +
  ``format_context_block`` pipeline as Q&A. Rebuilding a parallel
  retriever would drift over time; this way any future ranker upgrade
  lifts both surfaces together.

- **No question-framing.** Q&A's prompt says "Question: ..." + "Write
  your answer now, with inline [N] citations." Free Agent's user prompt
  says "User task: ..." — the LLM is free to interpret the input as a
  question, a rewrite request, a summarization brief, a compare-two-
  things ask, etc. The specialized recipes exist BECAUSE narrow framing
  produces better output; Free Agent is deliberately broad.

- **Single required input.** No answer-style select, no filters, no
  tone toggle — the whole point of the catch-all is minimum friction.
  Users who want more structure pick a specialized agent instead.

- **Still cites.** Even though the task isn't a "question," source
  attribution is a load-bearing VPS value prop — the answer is grounded
  in the user's own cart. So the prompt still asks the model to cite
  passage numbers inline, and the markdown still emits a Sources
  section with ``vps://source/{slug}`` links.
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
class FreeAgent(Agent):
    """Catch-all recipe — describe any task, use the cart as context."""

    name = "free_agent"
    display_name = "Free Agent"
    description = (
        "When none of the specialized recipes fit — describe your task "
        "or question, and Free Agent will do its best with your cart as "
        "context."
    )
    llm_dependency = True

    input_schema: list[dict[str, Any]] = [
        {
            "name": "user_input",
            "label": "What would you like the agent to do?",
            "type": "textarea",
            "required": True,
            "placeholder": (
                "Summarize the last month of additions... rewrite this "
                "passage for a general audience... compare the two "
                "most-cited papers..."
            ),
            "helpText": (
                "Any task or question the cart's material could help with."
            ),
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

        user_input = inputs.get_str("user_input", "") or ""

        if not user_input.strip():
            raise ValueError("Free Agent requires a non-empty 'user_input'.")

        if cart.count == 0:
            return AgentOutput(
                markdown=(
                    f"# Free Agent: {cart.cart_name}\n\n"
                    f"**Task.** {user_input}\n\n"
                    "This cart is empty — there's no substrate to work with.\n"
                ),
                warnings=warnings,
                metadata={"pattern_count": 0},
            )

        patterns = retrieve_top_patterns(
            cart, user_input, options.max_context_patterns,
        )
        if not patterns:
            warnings.append(
                "No passages scored above zero against the task input — "
                "the response below is unlikely to be grounded."
            )
            # Fall back to first-N so the LLM still gets substrate rather
            # than a completely empty context (which triggers hallucination).
            patterns = retrieve_top_patterns(
                cart, "", options.max_context_patterns,
            )

        context_block = format_context_block(patterns)

        system_prompt = (
            "You are a helpful assistant with access to a knowledge base. "
            "The user will describe a task or ask a question. Use the "
            "retrieved context to complete their task or answer their "
            "question thoroughly and accurately. Cite sources with [N] "
            "where relevant. If the context doesn't contain enough "
            "information, say so honestly."
        )
        user_prompt = (
            f"Retrieved context:\n\n{context_block}\n\n"
            f"---\n\n"
            f"User task: {user_input}"
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
            "(LLM returned no content. Try rewording the task.)"
        )

        # Assemble markdown output — task + response + a Sources section
        # with vps://source/{slug} links so users can drill down.
        lines: list[str] = []
        lines.append(f"# Free Agent: {cart.cart_name}")
        lines.append("")
        lines.append(f"**Task.** {user_input}")
        lines.append("")
        lines.append("## Response")
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
            "user_input": user_input,
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


__all__ = ["FreeAgent"]
