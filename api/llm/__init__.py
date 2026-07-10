"""Public surface for the VPS LLM adapter layer.

Report modules and other consumers should only depend on these four
symbols. The provider classes themselves are import-visible via
:mod:`api.llm.providers`, but callers should reach for
:func:`get_llm_adapter` first — that's the composition point.

Usage::

    from api.llm import get_llm_adapter, SynthesisResult

    llm = get_llm_adapter()
    result: SynthesisResult = llm.synthesize(
        "Summarize this cart in 5 bullets.",
        model_hint="large",
        max_tokens=1024,
    )
    if result.error:
        # transient failure — retry or fall back
        ...
    else:
        print(result.text)

See ``api/llm/README.md`` for the full story (three-tier LLM
architecture, provider configuration, adding a new provider).
"""
from .adapter import LLMAdapter, LLMError, SynthesisResult
from .registry import get_llm_adapter

__all__ = ["LLMAdapter", "LLMError", "SynthesisResult", "get_llm_adapter"]
