"""Top-level runner for the Agents engine.

Composition point wired to the FastAPI route (``/api/agents/run``).
Callers pass an agent slug + cart path + raw form inputs and get back
a fully-populated :class:`AgentOutput`.

Guardrails enforced here (so no agent module has to re-implement them):

- Unknown agent slug → :class:`ValueError` with an ``'unknown agent'``
  message. The FastAPI route converts this to a 400.
- Agent has ``llm_dependency=True`` but ``options.max_llm_calls == 0``
  → :class:`ValueError` pointing at the field. Neuron-cap enforcement
  in the route sets ``max_llm_calls`` based on remaining quota.
- Any exception inside ``execute()`` is re-raised with agent name + cart
  path context added to the message. Original traceback stays attached
  via ``raise X from exc``.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .base import AgentInput, AgentOptions, AgentOutput
from .registry import get_agent_by_name


def run_agent(
    agent_name: str,
    cart_path: str,
    raw_inputs: dict[str, Any],
    options: Optional[AgentOptions] = None,
) -> AgentOutput:
    """Look up ``agent_name``, instantiate, execute, return output.

    Parameters
    ----------
    agent_name
        Slug matching a registered :class:`Agent`. Case-sensitive.
    cart_path
        Absolute path to the ``.cart.npz`` file the agent reads from.
    raw_inputs
        Untyped dict of user-supplied form values. Wrapped in an
        :class:`AgentInput` for the agent to read via ``get_*``
        accessors.
    options
        Execution options (LLM budget, verbosity, source refs). If
        ``None``, defaults from :class:`AgentOptions` are used.

    Returns
    -------
    :class:`AgentOutput`
        Agent's markdown + citations + warnings + metadata.

    Raises
    ------
    ValueError
        Unknown agent slug, or LLM-required agent called with a zero
        budget, or the agent itself raised anything derivable from
        ValueError.
    Exception
        Whatever the agent itself raised — re-raised with context.
    """
    if options is None:
        options = AgentOptions()

    agent_cls = get_agent_by_name(agent_name)
    if agent_cls is None:
        raise ValueError(
            f"Unknown agent: {agent_name!r}. Registered agents: "
            f"{sorted(_registered_slugs())}"
        )

    # LLM budget guardrail. Agents that need LLM but got a zero budget
    # would fail deep inside execute() with a less-helpful message.
    if agent_cls.llm_dependency and options.max_llm_calls <= 0:
        raise ValueError(
            f"Agent {agent_name!r} requires LLM synthesis but "
            f"options.max_llm_calls={options.max_llm_calls}. Set "
            f"options.max_llm_calls >= 1 to run this agent."
        )

    inputs = AgentInput(raw=dict(raw_inputs or {}))
    # Copy pattern_filter through from raw_inputs if the caller sent it —
    # server-side default is 'active_only'. Frontend can override.
    pf = str(raw_inputs.get("pattern_filter", "active_only") or "active_only")
    if pf in {"active_only", "include_superseded", "all"}:
        inputs.pattern_filter = pf  # type: ignore[assignment]

    agent = agent_cls()

    t0 = time.time()
    try:
        output = agent.execute(cart_path, inputs, options)
    except Exception as exc:
        raise type(exc)(
            f"[agents.executor] {agent_name!r} against {cart_path!r} "
            f"failed: {exc}"
        ) from exc
    elapsed_ms = int((time.time() - t0) * 1000)

    if not isinstance(output, AgentOutput):
        raise TypeError(
            f"Agent {agent_name!r} returned {type(output).__name__} "
            f"but Agent.execute() must return AgentOutput"
        )

    meta = dict(output.metadata) if output.metadata else {}
    meta.setdefault("agent_name", agent_name)
    meta.setdefault("cart_path", cart_path)
    meta.setdefault("elapsed_ms", elapsed_ms)
    if options.verbose:
        meta.setdefault(
            "options",
            {
                "llm_provider": options.llm_provider,
                "llm_model_hint": options.llm_model_hint,
                "max_llm_calls": options.max_llm_calls,
                "max_tokens": options.max_tokens,
                "max_context_patterns": options.max_context_patterns,
                "include_source_refs": options.include_source_refs,
            },
        )
    output.metadata = meta

    return output


def _registered_slugs() -> list[str]:
    """Return the list of registered agent slugs (for error messages)."""
    from .registry import REGISTRY
    return list(REGISTRY.keys())


__all__ = ["run_agent"]
