"""Heartbeat / BYO-Claude adapter — Track A (placeholder).

Track A is the "power user brings their own Claude" path: the user
already has a Claude Pro / Team subscription and the Heartbeat browser
extension installed, and VPS hands off the prompt through that surface.
Free to us; user-supplied quality (typically the best of the three
tracks); requires user setup.

**Not yet implemented.** The likely shape when we ship it:

- **Option 1 — local REST endpoint.** Heartbeat exposes a loopback HTTP
  endpoint (similar to Image Builder's ``127.0.0.1:7879``) that accepts
  ``{prompt, model_hint}`` and returns the same
  :class:`~api.llm.adapter.SynthesisResult` shape. Adapter POSTs to it
  with a shared bearer token at ``~/.vector-plus/heartbeat-token``.
- **Option 2 — MCP handoff via Membot.** Heartbeat and Membot both
  speak MCP; the adapter emits an MCP tool call that Heartbeat's MCP
  server dispatches to the user's Claude session, and the response
  round-trips back through the same channel. Higher fidelity, more
  moving parts.

Option 1 is simpler and matches the Image Builder client pattern
we've already validated. Option 2 composes better with the rest of
the Membot / MCP surface — probably wins long-term. Decision is
deferred until Heartbeat exposes one or both endpoints.

The class exists now so the registry stays enumerable and the
symmetry with the Cloudflare and Anthropic adapters is preserved.
"""
from __future__ import annotations

from ..adapter import LLMAdapter, LLMError, SynthesisResult


class HeartbeatAdapter(LLMAdapter):
    """Placeholder for the Track A Heartbeat / BYO-Claude adapter.

    :meth:`synthesize` raises :class:`LLMError` with an actionable
    message pointing the operator at Heartbeat setup. Downstream code
    can import + instantiate the class without breaking.
    """

    @property
    def provider_name(self) -> str:
        return "heartbeat"

    def synthesize(
        self,
        prompt: str,
        *,
        model_hint: str = "default",
        max_tokens: int = 2048,
    ) -> SynthesisResult:
        raise LLMError(
            "Heartbeat adapter not yet implemented — install Heartbeat "
            "browser extension + configure MCP for Track A BYO-Claude "
            "flow."
        )
