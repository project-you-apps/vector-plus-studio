"""Agent registry — auto-registration decorator + lookup helpers.

Agent modules import ``@register_agent`` from this module and stack the
decorator on their :class:`~api.agents.base.Agent` subclass:

    from api.agents.base import Agent
    from api.agents.registry import register_agent

    @register_agent
    class QAAgent(Agent):
        name = "qa"
        display_name = "Q&A"
        ...

Naming discipline: ``Agent.name`` values MUST match the ``name`` field
in ``frontend/src/agents/agent-definitions.ts`` for frontend → backend
routing to work. Underscore convention for multi-word slugs (matches
Reports engine convention set in ):

    auto_briefing        # NOT "auto-briefing"
    qa
    professor
    cart_curator         # NOT "cart-curator"
"""
from __future__ import annotations

from typing import Any, Optional

from .base import Agent


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

# Public dict — Agent class keyed by its slug. Read-only for external
# callers; mutation goes through @register_agent.
REGISTRY: dict[str, type[Agent]] = {}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def register_agent(cls: type[Agent]) -> type[Agent]:
    """Register ``cls`` in the global :data:`REGISTRY`.

    Meant to be used as a decorator on an :class:`~api.agents.base.Agent`
    subclass. Raises :class:`ValueError` if the slug is already taken —
    duplicates are always a bug so failing loud at import time is right.

    Also validates that ``cls.name`` is a non-empty string, otherwise
    the registry would be silently keyed on ``""``.
    """
    if not isinstance(cls.name, str) or not cls.name:
        raise ValueError(
            f"Agent subclass {cls.__name__!r} has no ``name`` set; "
            f"add a slug matching the frontend agent-definitions.ts entry."
        )
    if cls.name in REGISTRY:
        existing = REGISTRY[cls.name].__name__
        raise ValueError(
            f"Agent {cls.name!r} already registered by {existing}; "
            f"cannot re-register {cls.__name__}."
        )
    REGISTRY[cls.name] = cls
    return cls


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_agent_by_name(name: str) -> Optional[type[Agent]]:
    """Return the :class:`Agent` subclass for ``name``, or ``None``.

    Returns ``None`` (instead of raising) so the executor + FastAPI
    route both get to distinguish "unknown agent" (user error → 400)
    from other exceptions (internal error → 500).
    """
    return REGISTRY.get(name)


def list_agents() -> list[dict[str, Any]]:
    """Return metadata for every registered agent.

    Powers ``GET /api/agents/list`` (frontend card grid). Same shape as
    :meth:`Agent.describe` so the endpoint is a one-liner.

    Order: insertion order (Python 3.7+ dict semantics), which matches
    ``import`` order in ``api/agents_routes.py``.
    """
    return [cls.describe() for cls in REGISTRY.values()]


__all__ = [
    "REGISTRY",
    "register_agent",
    "get_agent_by_name",
    "list_agents",
]
