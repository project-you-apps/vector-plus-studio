"""Provider implementations for the LLM adapter layer.

Each provider implements :class:`~api.llm.adapter.LLMAdapter`. The
active provider is chosen at runtime by
:func:`~api.llm.registry.get_llm_adapter` based on the
``VECTOR_PLUS_LLM_PROVIDER`` environment variable.

Adding a new provider:

1. Drop a ``newprovider.py`` in this folder with a class implementing
   :class:`LLMAdapter`.
2. Re-export it below.
3. Register the string key in :mod:`api.llm.registry`.
"""
from .anthropic import AnthropicAdapter
from .cloudflare import CloudflareAdapter
from .heartbeat import HeartbeatAdapter

__all__ = ["AnthropicAdapter", "CloudflareAdapter", "HeartbeatAdapter"]
