"""Anthropic Claude API adapter (placeholder).

VPS-hosted Claude access: the droplet holds an Anthropic API key, calls
``POST https://api.anthropic.com/v1/messages`` on the user's behalf, and
bills the customer per report / per token. This is the enterprise-tier
default. Power users prefer the Heartbeat BYO-Claude adapter, and
free-tier users default to the Cloudflare Workers AI adapter.

**Not yet implemented.** The class exists so the registry can enumerate
all three providers and so future integration is a straight file-swap.
When wiring the real implementation:

- Endpoint: ``POST https://api.anthropic.com/v1/messages``
- Auth: ``x-api-key: {ANTHROPIC_API_KEY}``, plus
  ``anthropic-version: 2023-06-01`` (or whatever the current stable is).
- Body shape: ``{"model": <id>, "max_tokens": N,
  "messages": [{"role": "user", "content": <prompt>}]}``.
- Model hint mapping:

    ================  ==================================
    model_hint        Claude model id
    ================  ==================================
    default           ``claude-haiku-4-5``
    small             ``claude-haiku-4-5``
    large             ``claude-sonnet-5``
    vision            ``claude-opus-4-8``
    ================  ==================================

- Cost tracking: the response body carries ``usage.input_tokens`` and
  ``usage.output_tokens``; multiply by the current model rate to
  populate ``SynthesisResult.cost_usd``.
- Rate limits: 429 → return ``SynthesisResult(error="rate limited")``,
  same shape as :class:`~api.llm.providers.cloudflare.CloudflareAdapter`.

Keeping the file structure symmetric with the Cloudflare adapter now
means the eventual implementation is a mechanical port, not an
architecture decision.
"""
from __future__ import annotations

from ..adapter import LLMAdapter, LLMError, SynthesisResult


class AnthropicAdapter(LLMAdapter):
    """Placeholder for the Anthropic Claude adapter.

    :meth:`synthesize` currently raises :class:`LLMError`. The class
    still exists so :func:`api.llm.registry.get_llm_adapter` can select
    it and produce a helpful error, and so downstream code can import
    it without ImportError once the real implementation ships.
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def synthesize(
        self,
        prompt: str,
        *,
        model_hint: str = "default",
        max_tokens: int = 2048,
    ) -> SynthesisResult:
        raise LLMError(
            "Anthropic adapter not yet implemented — set "
            "VECTOR_PLUS_LLM_PROVIDER=cloudflare or install Heartbeat."
        )
