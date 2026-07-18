"""Report engine base types — the interface every report module implements.

This module is the foundation for the VPS Reports engine. Report
modules (Summary / Timeline / Trend / Comparison / Entity Rollup /
Financial Rollup / Change Log / Executive TL;DR) all subclass
:class:`Report` and re-export via the :mod:`api.reports.registry`
decorator ``@register_report``.

Common architecture: this module defines the ABC + IO shapes; the
registry lives in ``registry.py``; extractors are dispatched separately.

Type dependencies:

- LLM synthesis: reports opt in by setting ``llm_dependency = True`` and
  calling :func:`api.llm.get_llm_adapter` inside ``generate()``. The
  executor (``executor.py``) enforces the ``max_llm_calls`` guardrail
  before ``generate()`` runs so a misconfigured caller can't burn tokens
  by accident.
- Cart reads: use :class:`api.reports.cart_reader.CartHandle` rather
  than re-implementing NPZ parsing per report. That module owns the
  passage / source_paths / pattern0 / per_pattern_meta dance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# ReportInput — the untyped user-supplied form values
# ---------------------------------------------------------------------------

@dataclass
class ReportInput:
    """User-supplied inputs for a single :meth:`Report.generate` call.

    ``raw`` mirrors the frontend form payload shape defined in
    ``frontend/src/reports/report-definitions.ts`` (``FieldSchema``
    entries turn into ``{name: value}`` pairs on submit). Report modules
    read via the ``get_*`` helpers instead of touching ``raw`` directly
    so the accessors can grow validation / coercion / defaulting without
    every module having to re-implement it.

    Missing keys return ``default`` — reports opting in to strict
    validation should check ``default is None`` on required inputs and
    raise a helpful ``ValueError`` themselves. The base layer stays
    permissive by design; the executor wraps every ``generate()`` in a
    try/except that adds report name + input keys to the failure
    message, so a bad input surfaces with enough context to debug.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    # ---- typed accessors -------------------------------------------------
    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch ``key`` as a stripped string, or ``default`` if missing/empty."""
        val = self.raw.get(key, default)
        if val is None:
            return default
        s = str(val).strip()
        # Frontend forms tend to submit empty strings for un-filled optional
        # fields; treat those as "missing" for report-author convenience.
        return s if s else default

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Fetch ``key`` as an int, or ``default`` if missing / unparseable."""
        val = self.raw.get(key, default)
        if val is None or val == "":
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Fetch ``key`` as a bool. Accepts native bools + common string forms
        ("true"/"1"/"yes"/"on" → True; everything else → False)."""
        val = self.raw.get(key, default)
        if isinstance(val, bool):
            return val
        if val is None or val == "":
            return default
        return str(val).strip().lower() in {"true", "1", "yes", "on"}

    def get_list(
        self,
        key: str,
        default: Optional[list[Any]] = None,
    ) -> list[Any]:
        """Fetch ``key`` as a list.

        Native ``list`` values pass through. Comma-separated strings
        (common frontend shape for the entity-rollup ``aliases`` field)
        get split + stripped. Missing / empty returns ``default`` or an
        empty list.
        """
        if default is None:
            default = []
        val = self.raw.get(key)
        if val is None or val == "":
            return list(default)
        if isinstance(val, list):
            return list(val)
        if isinstance(val, str):
            parts = [p.strip() for p in val.split(",")]
            return [p for p in parts if p]
        # Anything else — wrap single value in a list.
        return [val]


# ---------------------------------------------------------------------------
# ReportOptions — caller-side execution config (NOT user inputs)
# ---------------------------------------------------------------------------

@dataclass
class ReportOptions:
    """Caller-supplied execution options for :meth:`Report.generate`.

    Separated from :class:`ReportInput` because these are not exposed to
    the end user — they're set by the FastAPI route (or the CLI runner,
    or a scheduled brief job) based on tier / policy / debug flags. The
    frontend form never touches these.

    ``max_llm_calls`` is a hard cap enforced by the executor **before**
    ``generate()`` runs — a non-LLM report gets ``max_llm_calls=0`` and
    stays cost-safe by construction. Executive TL;DR requires
    ``max_llm_calls >= 1`` or the executor short-circuits with a clear
    error pointing at this field.
    """

    # LLM routing overrides. Empty-string sentinel means "let the LLM
    # registry/env decide" — matches the ``get_llm_adapter()`` default path.
    llm_provider: str = "cloudflare"
    llm_model_hint: str = "default"

    # Hard budget cap. Reports check this before their FIRST synthesize()
    # call. 0 = no LLM allowed (the default). Executor blocks reports
    # marked ``llm_dependency=True`` when this is 0.
    max_llm_calls: int = 0

    # Extra logging surfaces (populated into ReportOutput.metadata for the
    # UI's "why did this report say X" audit view).
    verbose: bool = False

    # Whether ReportOutput should carry pattern-index references so the
    # UI can render "click through to source passage" affordances.
    # Turned off for lightweight briefs where the payload is bandwidth-
    # sensitive (email, Slack).
    include_source_refs: bool = True


# ---------------------------------------------------------------------------
# ReportOutput — what generate() returns
# ---------------------------------------------------------------------------

@dataclass
class ReportOutput:
    """Result payload returned by :meth:`Report.generate`.

    ``markdown`` is the primary + always-populated surface. Other fields
    are optional per report type:

    - ``csv_data``: populated by Trend + Financial Rollup — those reports
      are naturally consumed as data grids.
    - ``html_extra``: populated by anything that markdown can't express
      cleanly (SVG charts, nested tables). Future work.
    - ``metadata``: timing, LLM calls made, source-passage references,
      warnings surfaced by the report author. UI reads this for the
      audit / "show your work" footer.
    - ``warnings``: soft-fail messages surfaced to the user separately
      from the markdown body (e.g. "Metric found only 2 times — trend
      requires at least 3 datapoints"). Empty list is the happy path.
    """

    markdown: str
    csv_data: Optional[str] = None
    html_extra: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_json_response(self) -> dict[str, Any]:
        """Serialize for the FastAPI ``/api/reports/generate`` route.

        The route wraps this in the outer envelope; the report itself
        only owns what's inside. Keeping the shape flat so the frontend
        can render + export without a schema-migration dance later.
        """
        return {
            "markdown": self.markdown,
            "csv_data": self.csv_data,
            "html_extra": self.html_extra,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Report ABC — the interface every report module implements
# ---------------------------------------------------------------------------

class Report(ABC):
    """Base class for a single report type.

    Every concrete report is a class (not a function) so it can carry
    class-level metadata (``name``, ``display_name``, ``input_schema``,
    ``llm_dependency``, ``supports_scheduling``) that the registry
    surfaces to the frontend list endpoint without instantiating the
    report just to read its shape.

    ``name`` MUST match the corresponding entry in
    ``frontend/src/reports/report-definitions.ts`` — the frontend maps
    the card the user clicks onto ``POST /api/reports/generate`` with
    that slug. Currently expected slugs:
    ``summary``, ``timeline``, ``trend``, ``comparison``,
    ``entity_rollup``, ``financial_rollup``, ``change_log``, ``tldr``.
    (See ``registry.py`` for the naming-collision discipline.)

    ``input_schema`` mirrors the frontend ``FieldSchema[]`` array shape
    verbatim; keeping the source-of-truth on the backend lets us
    auto-generate the frontend list from ``list_reports()`` when the
    list endpoint is wired. Until then, the two sides are manually kept
    in sync — a smoke test asserts equality.
    """

    # -- class-level metadata (overridden by subclasses) ------------------
    # Slug used in URLs + the registry + the frontend definition file.
    name: str = ""
    # User-facing label rendered on the card + report header.
    display_name: str = ""
    # One-line description for the card grid.
    description: str = ""
    # List[dict] mirroring frontend ``FieldSchema``; each entry:
    # {name, label, type, required, default?, options?, placeholder?, helpText?}
    input_schema: list[dict[str, Any]] = []
    # True iff ``generate()`` calls ``llm.synthesize()``. Only Executive
    # TL;DR is True today.
    llm_dependency: bool = False
    # True iff this report makes sense as a scheduled brief (Hot Stack
    # composition surface). Summary + Entity Rollup + Financial + TL;DR
    # are candidates; Change Log requires two carts so skip.
    supports_scheduling: bool = False

    # -- interface --------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        cart_path: str,
        inputs: ReportInput,
        options: ReportOptions,
    ) -> ReportOutput:
        """Produce the report for ``cart_path``.

        Implementations should:

        1. Load the cart via
           :class:`api.reports.cart_reader.CartHandle` (do NOT open the
           NPZ directly — the helper handles both passage-format
           dialects and the pattern0 + per_pattern_meta decodes).
        2. If ``llm_dependency=True``: call
           :func:`api.llm.get_llm_adapter` and drive synthesize().
           Respect ``options.max_llm_calls`` — the executor guards the
           entry, but reports still count internally so metadata reports
           actual usage.
        3. Return a :class:`ReportOutput` with ``markdown`` populated.
           Soft failures go in ``warnings``; hard failures raise
           (the executor wraps the message with report context).
        """
        ...

    # -- convenience helpers subclasses can lean on -----------------------
    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return the class-level metadata dict used by the registry.

        Reports rarely override this; the registry calls it directly to
        build the ``/api/reports/`` list response.
        """
        return {
            "name": cls.name,
            "display_name": cls.display_name,
            "description": cls.description,
            "input_schema": list(cls.input_schema),
            "llm_dependency": cls.llm_dependency,
            "supports_scheduling": cls.supports_scheduling,
        }


# ---------------------------------------------------------------------------
# Convenience re-imports for report authors
# ---------------------------------------------------------------------------
# Re-declared as ``__all__`` for ``from api.reports.base import *`` users;
# the package's ``__init__.py`` re-exports the same set at a shorter path.
__all__ = [
    "Report",
    "ReportInput",
    "ReportOptions",
    "ReportOutput",
]
