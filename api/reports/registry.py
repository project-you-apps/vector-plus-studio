"""Report registry — auto-registration decorator + lookup helpers.

Report modules import ``@register_report`` from this module and stack
the decorator on their :class:`~api.reports.base.Report` subclass:

    from api.reports.base import Report
    from api.reports.registry import register_report

    @register_report
    class SummaryReport(Report):
        name = "summary"
        display_name = "Summary"
        ...

Wave-2 wires the FastAPI route to ``run_report`` (executor.py). Wave-2+
also adds ``GET /api/reports/`` which serializes :func:`list_reports`
for the frontend card grid.

Naming discipline: ``Report.name`` values MUST match the ``name`` field
in ``frontend/src/reports/report-definitions.ts`` for the frontend →
backend routing to work. Current expected slugs (Andy 2026-07-11):

    summary
    timeline
    trend
    comparison
    entity_rollup      # NOT "entity-rollup" — frontend uses underscore
    financial_rollup   # NOT "financial-rollup"
    change_log         # NOT "change-log"
    tldr               # NOT "executive-tldr"

The Wave-1a foundation dispatch brief called out hyphenated slugs; the
actual frontend definitions use underscores. Wave-1b agents: match the
frontend file, not the brief.
"""
from __future__ import annotations

from typing import Any, Optional

from .base import Report


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

# Public dict — Report class keyed by its slug. Read-only for external
# callers; mutation goes through @register_report.
REGISTRY: dict[str, type[Report]] = {}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def register_report(cls: type[Report]) -> type[Report]:
    """Register ``cls`` in the global :data:`REGISTRY`.

    Meant to be used as a decorator on a :class:`~api.reports.base.Report`
    subclass. Raises :class:`ValueError` if the slug is already taken —
    duplicates are always a bug (import-order accident, copy-paste
    mistake) so failing loud at import time is the right call.

    Also validates that ``cls.name`` is a non-empty string, otherwise
    the registry would be silently keyed on ``""`` and every subsequent
    duplicate check would collide.
    """
    if not isinstance(cls.name, str) or not cls.name:
        raise ValueError(
            f"Report subclass {cls.__name__!r} has no ``name`` set; "
            f"add a slug matching the frontend report-definitions.ts entry."
        )
    if cls.name in REGISTRY:
        existing = REGISTRY[cls.name].__name__
        raise ValueError(
            f"Report {cls.name!r} already registered by {existing}; "
            f"cannot re-register {cls.__name__}."
        )
    REGISTRY[cls.name] = cls
    return cls


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_report_by_name(name: str) -> Optional[type[Report]]:
    """Return the :class:`Report` subclass for ``name``, or ``None``.

    Returns ``None`` (instead of raising) because the executor + FastAPI
    route both want to distinguish "unknown report" (user error → 400)
    from other exceptions (internal error → 500). Callers get to shape
    the error response.
    """
    return REGISTRY.get(name)


def list_reports() -> list[dict[str, Any]]:
    """Return metadata for every registered report.

    Powers the Wave-2 ``GET /api/reports/`` list endpoint (frontend
    card grid). The shape here is deliberately the same as
    :meth:`Report.describe` so the endpoint is a one-liner.

    Order: insertion order (Python 3.7+ dict semantics), which matches
    ``import`` order of the report modules. Import order is set in
    ``api/reports/__init__.py`` to mirror the Report Types Design doc
    section ordering (§1 Summary first, §8 TL;DR last).
    """
    return [cls.describe() for cls in REGISTRY.values()]
