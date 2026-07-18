"""Public surface for the VPS Agents engine.

Mirrors the ``api.reports`` package layout — Agent ABC + registry +
executor + prompt helper — but with LLM-first agents wrapping cart
substrate rather than deterministic pattern-based reports.

Usage — dispatching an agent from a FastAPI route::

    from api.agents import run_agent, AgentOptions

    output = run_agent(
        agent_name="qa",
        cart_path="/opt/vps/cartridges/gutenberg-poetry.cart.npz",
        raw_inputs={"question": "What is the tone of War Poems?"},
    )
    return output.to_json_response()

Standing conventions (mirroring reports engine):

- ``@register_agent`` decorators fire on module import — this package's
  ``__init__.py`` does NOT import individual agent modules. Side-effect
  imports happen in ``api/agents_routes.py`` at app boot. New agents
  MUST be added to the import block at the top of ``agents_routes.py``.

- Slug format — underscore for multi-word (``auto_briefing``,
  ``cart_curator``). Frontend ``agent-definitions.ts`` must match.

- All agents call ``prompt.wrap_llama3_instruct()`` before invoking the
  LLM adapter — see ``prompt.py`` for the rationale.
"""
from .base import Agent, AgentInput, AgentOptions, AgentOutput
from .executor import run_agent
from .prompt import wrap_llama3_instruct
from .registry import (
    REGISTRY,
    get_agent_by_name,
    list_agents,
    register_agent,
)

__all__ = [
    # Core types
    "Agent",
    "AgentInput",
    "AgentOptions",
    "AgentOutput",
    # Registry
    "REGISTRY",
    "register_agent",
    "get_agent_by_name",
    "list_agents",
    # Runner
    "run_agent",
    # Prompt helper (Llama 3 instruct-template wrapper)
    "wrap_llama3_instruct",
]
