"""LLM adapter base interface for VPS.

Defines the pluggable provider surface that report modules (and, later,
the SQL-interpreter and "Ask this cart" flows) call to get text out of
whichever LLM backend is configured.

This adapter has three pluggable backends: browser-relayed (user brings
their own LLM session through a browser extension), hosted-paid (VPS
pays per token via a first-party LLM API), and hosted-free (open-source
models on a hosted inference platform).

Every provider implements :class:`LLMAdapter` and returns
:class:`SynthesisResult`. Callers should NOT catch provider-specific
exceptions — providers translate transient errors into a populated
``error`` field on the result, and reserve :class:`LLMError` for
configuration problems the caller cannot recover from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SynthesisResult:
    """Result of a single LLM synthesis call.

    Populated fields depend on the provider:

    - Cloudflare: ``model``, ``tokens_used``, ``neurons_used``, ``elapsed_ms``
    - Anthropic (later): ``model``, ``tokens_used``, ``cost_usd``, ``elapsed_ms``
    - Heartbeat (later): ``model``, ``tokens_used``
    - All: ``text``, ``provider``

    ``error`` is populated (and ``text`` is empty) if the call failed.
    Callers should check ``error`` first, then use ``text``.
    """

    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    neurons_used: Optional[int] = None   # CF-specific
    cost_usd: Optional[float] = None     # Anthropic-specific
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None


class LLMError(Exception):
    """Raised by adapters on unrecoverable errors (auth failure, missing
    config, unimplemented provider, etc). Transient errors (rate limit,
    timeout, network blip) should be surfaced via
    ``SynthesisResult.error`` instead so callers can retry or fall back
    without a try/except."""

    pass


class LLMAdapter(ABC):
    """Abstract base class every provider implements.

    Concrete adapters live under ``api/llm/providers/`` and get selected
    at runtime via :func:`api.llm.registry.get_llm_adapter` based on the
    ``VECTOR_PLUS_LLM_PROVIDER`` environment variable.
    """

    @abstractmethod
    def synthesize(
        self,
        prompt: str,
        *,
        model_hint: str = "default",
        max_tokens: int = 2048,
    ) -> SynthesisResult:
        """Send ``prompt`` to the underlying LLM, return the synthesis result.

        ``model_hint`` is provider-specific; providers translate
        ``'default'``, ``'small'``, ``'large'``, ``'vision'`` into their
        own model names. Unknown hints fall back to ``'default'``.

        Should NOT raise on transient errors (rate limit, timeout) —
        return a :class:`SynthesisResult` with ``error`` populated
        instead. :class:`LLMError` is reserved for config problems the
        caller can't recover from (missing API token, unimplemented
        provider selected).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short slug identifying the provider (``"cloudflare"``,
        ``"anthropic"``, ``"heartbeat"``). Used in logs, in Sentry
        tagging, and stamped into :attr:`SynthesisResult.provider`."""
        ...
