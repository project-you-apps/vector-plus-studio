"""Professor Agent — quiz generator over a cart.

"Give me a quiz on this cart" — the onboarding / study surface. Users
pick a difficulty + question count + optional topic; agent samples
passages from the cart, LLM synthesizes N questions with answers, output
is a study-worthy quiz block.

Design decisions (v1)
---------------------
- **Question format.** Short-answer + explanation. NOT multiple choice —
  MC would require the LLM to generate plausible distractors, which is a
  quality problem that hurts on the free-tier Llama models. Short-answer
  + explanation is a well-behaved LLM shape and reads like a good tutor.

- **Difficulty maps to sample width AND prompt directive.** Easy = 3
  patterns sampled, questions ask about surface facts. Medium = 5
  patterns, blend of surface + inference. Hard = 8 patterns, questions
  test cross-passage connections.

- **Topic filter is a retrieval query.** Same retrieval helper as Q&A;
  empty topic = broad sample. Topic-filtered quizzes let a user drill
  down on one theme (e.g. "give me a hard quiz on entity Sysco Portland").

- **Cites patterns.** ``cited_patterns`` = the pattern indices the LLM's
  questions were drawn from, so a user could click through to the
  source. Same convention as Q&A.
"""
from __future__ import annotations

from typing import Any

from api.reports.cart_reader import CartHandle
from api.reports.source_link import source_link

from .base import Agent, AgentInput, AgentOptions, AgentOutput
from .prompt import wrap_llama3_instruct
from .registry import register_agent
from .retrieval import retrieve_top_patterns, format_context_block


# Difficulty → (sample_patterns, prompt_directive) map. Sample width
# scales with difficulty because harder questions need more substrate to
# draw cross-connections from.
_DIFFICULTY_PROFILE: dict[str, tuple[int, str]] = {
    "easy": (
        3,
        "Ask surface-level factual questions any careful reader could "
        "answer from a single passage. Answers should be one sentence.",
    ),
    "medium": (
        5,
        "Blend surface facts with light inference. Answers can span "
        "1-2 sentences with a short explanation.",
    ),
    "hard": (
        8,
        "Ask questions that require synthesizing across multiple "
        "passages, or that test nuance rather than facts. Provide "
        "answers with a paragraph of explanation each.",
    ),
}

# Cap on question count — prevents pathological requests from burning
# tokens or producing unrenderable outputs.
_MAX_QUESTIONS = 20
_DEFAULT_QUESTIONS = 5


@register_agent
class ProfessorAgent(Agent):
    """Professor recipe — LLM-generated quiz over a cart."""

    name = "professor"
    display_name = "Professor"
    description = (
        "Generate a study quiz from the cart. Pick difficulty + question "
        "count + an optional topic. Great for onboarding new hires."
    )
    llm_dependency = True

    input_schema: list[dict[str, Any]] = [
        {
            "name": "num_questions",
            "label": "Number of questions",
            "type": "number",
            "required": True,
            "default": _DEFAULT_QUESTIONS,
            "helpText": f"How many quiz questions to generate (1-{_MAX_QUESTIONS}).",
        },
        {
            "name": "difficulty",
            "label": "Difficulty",
            "type": "select",
            "required": True,
            "default": "medium",
            "options": ["easy", "medium", "hard"],
            "helpText": "Shapes both retrieval width and question style.",
        },
        {
            "name": "topic",
            "label": "Topic filter (optional)",
            "type": "text",
            "required": False,
            "placeholder": "e.g. Sysco Portland, fuel surcharges, poetry themes",
            "helpText": "Optional — narrow the quiz to a specific topic.",
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

        num_q = inputs.get_int("num_questions", _DEFAULT_QUESTIONS) or _DEFAULT_QUESTIONS
        num_q = max(1, min(num_q, _MAX_QUESTIONS))
        difficulty = (inputs.get_str("difficulty", "medium") or "medium").lower()
        if difficulty not in _DIFFICULTY_PROFILE:
            warnings.append(
                f"Unknown difficulty {difficulty!r} — falling back to 'medium'."
            )
            difficulty = "medium"
        topic = inputs.get_str("topic", "") or ""

        if cart.count == 0:
            return AgentOutput(
                markdown=(
                    f"# Quiz: {cart.cart_name}\n\n"
                    "This cart is empty — no material to quiz on.\n"
                ),
                warnings=warnings,
                metadata={"pattern_count": 0},
            )

        sample_n, style_directive = _DIFFICULTY_PROFILE[difficulty]
        # Give the LLM a bit more substrate than the difficulty base so
        # question variety doesn't collapse when the retrieval hits are
        # thin. Bounded by max_context_patterns to respect the caller's
        # neuron budget.
        sample_n = min(max(sample_n, num_q), options.max_context_patterns)
        patterns = retrieve_top_patterns(cart, topic, sample_n)
        if not patterns:
            warnings.append(
                "No passages matched the topic filter — quiz will draw from "
                "an unfocused sample of the cart."
            )
            patterns = retrieve_top_patterns(cart, "", sample_n)

        context_block = format_context_block(patterns)

        system_prompt = (
            "You are a professor writing a study quiz over material the "
            "user has provided. Rules: (1) Only use information present "
            "in the passages. (2) Format each question as: "
            "'**Q1.** <question>' on its own line, then '**A1.** "
            "<answer>' on the next line, blank line between questions. "
            "(3) Cite passage numbers inline in the answer like [1] or "
            "[2] where a fact comes from a specific passage. (4) Do not "
            "generate multiple-choice; use short-answer questions with "
            "written answers. "
            f"(5) Difficulty guidance: {style_directive}"
        )
        topic_line = f'Topic focus: "{topic}".\n\n' if topic else ""
        user_prompt = (
            f"Passages from the cart:\n\n{context_block}\n\n"
            f"{topic_line}"
            f"Generate exactly {num_q} question(s), numbered starting at 1. "
            f"Write them now."
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
            "(LLM returned no content. Try adjusting the topic or "
            "difficulty and running again.)"
        )

        # Compose markdown output. Header + metadata line + LLM body +
        # source list (numbered by retrieval order to match the [N]
        # citations the model was told to emit).
        lines: list[str] = []
        lines.append(f"# Quiz: {cart.cart_name}")
        lines.append("")
        difficulty_label = difficulty.capitalize()
        topic_display = topic or "any topic"
        lines.append(
            f"**{num_q}** {difficulty_label}-difficulty questions on *{topic_display}*."
        )
        lines.append("")
        lines.append(body)
        lines.append("")

        if options.include_source_refs and patterns:
            lines.append("## Sources")
            lines.append("")
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
            "retrieved_source_count": len({p.source for p in patterns if p.source}),
            "num_questions": num_q,
            "difficulty": difficulty,
            "topic": topic or None,
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


__all__ = ["ProfessorAgent"]
