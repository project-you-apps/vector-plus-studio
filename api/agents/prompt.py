"""Prompt-engineering helper — Llama 3 instruct-template wrapper.

**LOAD-BEARING.** Every LLM call from every agent MUST wrap its prompt
in this template before hitting :func:`api.llm.get_llm_adapter`.

Rationale
---------
The Cloudflare Workers AI runtime, the default provider, hosts
open-source Llama 3 instruct models. Those models were fine-tuned on
the ChatML-like Llama 3 header format::

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    …system prompt…<|eot_id|><|start_header_id|>user<|end_header_id|>
    …user prompt…<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Feed the model a bare user string and it falls back to base-model
completion — a Q&A prompt about a poetry cart produced a rambling
non-sequitur completion instead of an answer in that failure mode.

Wrapping the prompt in the instruct headers puts the model firmly in
chat-completion mode, and outputs read as actual answers to the
question. The wrap costs a few dozen tokens; the failure mode costs
the entire response quality.

Provider portability
--------------------
Anthropic + Heartbeat adapters don't need the wrap — but the extra
tokens are harmless (Anthropic strips leading whitespace, Claude
handles the header tokens as ordinary text). Keeping ALL agents on
one wrap keeps agent code portable across providers without a
per-provider conditional.

If a future adapter proves the wrap actively hurts on its provider,
extend the CF Worker to detect + strip.

Historical note
---------------
The initial dispatch tried raw prompts. The completion-mode failure
surfaced quickly and the wrap landed as the fix in the same commit as
the initial scaffold, so no test in this codebase exists showing the
failure mode — it lives in the LLM's training-time behavior, not ours.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Constants — the raw Llama 3 header + separator tokens
# ---------------------------------------------------------------------------

# These strings are load-bearing — do NOT edit without verifying against
# the Llama 3 tokenizer. Any drift breaks the template silently (model
# reverts to base completion).
_BOS = "<|begin_of_text|>"
_HDR_START = "<|start_header_id|>"
_HDR_END = "<|end_header_id|>"
_EOT = "<|eot_id|>"


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def wrap_llama3_instruct(system: str, user: str) -> str:
    """Wrap ``system`` + ``user`` prompts in the Llama 3 instruct template.

    Parameters
    ----------
    system
        System role message — sets the agent's persona / rules /
        output-shape expectations. Kept concise; long system prompts
        eat into the context budget without helping quality.
    user
        User role message — the actual question / task / retrieval
        context. Cited-passage blocks go here (not in system) so the
        model treats them as part of the user's ask.

    Returns
    -------
    str
        A single string ready to hand to
        :meth:`api.llm.adapter.LLMAdapter.synthesize`. Do NOT further
        modify — the trailing header token positions the model to start
        generating in the assistant role.

    Notes
    -----
    Empty strings are permitted (some agents want an empty system) but
    the wrap still emits both header blocks. The model tolerates empty
    role content; downstream token counting stays predictable.
    """
    return (
        f"{_BOS}"
        f"{_HDR_START}system{_HDR_END}\n\n"
        f"{system}{_EOT}"
        f"{_HDR_START}user{_HDR_END}\n\n"
        f"{user}{_EOT}"
        f"{_HDR_START}assistant{_HDR_END}\n\n"
    )


__all__ = ["wrap_llama3_instruct"]
