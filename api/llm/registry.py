"""Provider selection for the VPS LLM adapter layer.

Reads ``VECTOR_PLUS_LLM_PROVIDER`` from the environment and returns a
constructed :class:`~api.llm.adapter.LLMAdapter`. Defaults to Cloudflare
Workers AI, the free-tier default.

Callers should hold onto the returned adapter for the lifetime of the
request (or the process) rather than calling :func:`get_llm_adapter`
per synthesis — provider ``__init__`` methods can do env-var validation
that shouldn't run in a tight loop.
"""
from __future__ import annotations

import os

from .adapter import LLMAdapter, LLMError
from .providers import AnthropicAdapter, CloudflareAdapter, HeartbeatAdapter


# Env var name is deliberately verbose (``VECTOR_PLUS_LLM_PROVIDER``
# rather than ``LLM_PROVIDER``) so it can't collide with any other
# tenant on the droplet — VPS shares an env namespace with the cart
# builder + image builder + membot on the same box.
_ENV_VAR = "VECTOR_PLUS_LLM_PROVIDER"
_DEFAULT_PROVIDER = "cloudflare"


def get_llm_adapter() -> LLMAdapter:
    """Return the configured LLM adapter.

    Reads ``VECTOR_PLUS_LLM_PROVIDER`` and constructs the matching
    provider class. Unknown provider names raise :class:`LLMError` so
    the caller sees the misconfiguration immediately instead of at
    first synthesis attempt.
    """
    provider = os.environ.get(_ENV_VAR, _DEFAULT_PROVIDER).strip().lower()
    if provider == "cloudflare":
        return CloudflareAdapter()
    elif provider == "anthropic":
        return AnthropicAdapter()
    elif provider == "heartbeat":
        return HeartbeatAdapter()
    else:
        raise LLMError(
            f"Unknown provider: {provider!r}. Expected one of: "
            f"cloudflare, anthropic, heartbeat."
        )
