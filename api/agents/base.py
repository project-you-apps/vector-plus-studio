"""Agent engine base types — the interface every agent module implements.

Foundation for the VPS Agents engine (2026-07-13, Andy dispatch). Modeled
on ``api/reports/base.py`` so callers who already know the Reports shape
can navigate this package by analogy. Key differences vs Reports:

- Agents are **LLM-first**. Where Reports opt in via ``llm_dependency``,
  Agents assume LLM synthesis is the point — the ``AgentOptions.max_llm_calls``
  budget defaults to 1 (not 0) and the executor validates against 0 for
  agents that need it.

- Agents surface **cited patterns**. ``AgentOutput.cited_patterns`` is a
  first-class field carrying the pattern indices the LLM's answer draws
  from; the frontend renders them as ``vps://source/{slug}`` links (same
  handler as Reports' source-file links).

- Agents are **saveable to cart**. ``AgentOutput`` carries enough
  provenance (the raw prompt + timestamp + citations) that the
  ``/api/agents/save_to_cart`` route can persist the run as a new
  pattern in the user's Membot cart — the "your own memory" loop.

Type dependencies:

- LLM synthesis: agents call :func:`api.llm.get_llm_adapter` inside
  ``execute()`` and drive ``synthesize()``. Prompts MUST be wrapped via
  :func:`api.agents.prompt.wrap_llama3_instruct` first to avoid the
  completion-mode failure logged 2026-07-13.
- Cart reads: use :class:`api.reports.cart_reader.CartHandle` rather
  than re-implementing NPZ parsing. The Reports package owns the cart
  reader — Agents just import it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# AgentInput — user-supplied form values
# ---------------------------------------------------------------------------

@dataclass
class AgentInput:
    """User-supplied inputs for a single :meth:`Agent.execute` call.

    Same shape as :class:`api.reports.base.ReportInput` so agents that
    reuse Reports form patterns feel familiar. ``raw`` mirrors the
    frontend form payload; typed accessors (``get_str`` / ``get_int`` /
    ``get_bool`` / ``get_list``) handle coercion and defaults.

    ``pattern_filter`` is agent-specific: this a
    Hot Stack integration story where SUPERSEDED / ARCHIVED patterns
    should be filtered out server-side. In v1 (before Hot Stack lands)
    the metadata bit is absent, filter is a no-op, everything passes
    through. The field is on the base type so agent authors don't have
    to redeclare it.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    # Hot Stack metadata filter.
    # 'active_only' is the safe default once Hot Stack lands; today it's
    # a no-op so v1 agents don't need to plumb the field through.
    pattern_filter: Literal["active_only", "include_superseded", "all"] = "active_only"

    # ---- typed accessors -------------------------------------------------
    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch ``key`` as a stripped string, or ``default`` if missing/empty."""
        val = self.raw.get(key, default)
        if val is None:
            return default
        s = str(val).strip()
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
        """Fetch ``key`` as a bool. Accepts native bools + common string forms."""
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
        """Fetch ``key`` as a list. Comma-separated strings get split."""
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
        return [val]


# ---------------------------------------------------------------------------
# AgentOptions — caller-side execution config (NOT user inputs)
# ---------------------------------------------------------------------------

@dataclass
class AgentOptions:
    """Caller-supplied execution options for :meth:`Agent.execute`.

    Separated from :class:`AgentInput` because these are not exposed to
    the end user — they're set by the FastAPI route based on tier /
    policy / neuron-cap enforcement.

    ``max_llm_calls`` defaults to 1 (agents are LLM-first, unlike
    Reports which default to 0). Executor validates against 0 for
    agents whose ``llm_dependency=True``.

    ``max_context_patterns`` bounds the retrieval size passed to the
    LLM as context — bigger = higher quality but higher neuron cost.
    Q&A and Cart Curator both retrieve top-N patterns and stuff them
    into the prompt as context blocks.
    """

    # LLM routing overrides — empty string means "let the LLM registry
    # decide" (matches ``get_llm_adapter()`` default path).
    llm_provider: str = "cloudflare"
    llm_model_hint: str = "default"

    # Hard budget cap. Agents check this before their FIRST synthesize()
    # call. 0 = no LLM allowed (executor blocks LLM-dep agents up front).
    max_llm_calls: int = 1

    # Max tokens per synthesis call — caps output size on any provider.
    # 2048 matches the LLM adapter default; agents can override per call.
    max_tokens: int = 2048

    # Max patterns pulled into the LLM prompt context (Q&A + Curator).
    max_context_patterns: int = 8

    # Extra logging surfaces (populated into AgentOutput.metadata).
    verbose: bool = False

    # Whether to include cited_patterns provenance in the output. Off
    # for lightweight briefs (email preview); on for the interactive UI.
    include_source_refs: bool = True


# ---------------------------------------------------------------------------
# AgentOutput — what execute() returns
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """Result payload returned by :meth:`Agent.execute`.

    ``markdown`` is the primary always-populated surface. Structure:

    - ``markdown``: rendered agent response — headings, lists, cited
      ``vps://source/{slug}`` links.
    - ``cited_patterns``: list of pattern indices the LLM's answer drew
      from. Populated when the agent retrieves top-N patterns as
      context (Q&A, Curator). Empty for agents that don't cite (Auto
      Briefing summarizes the whole cart, no pattern-level citations).
    - ``metadata``: timing, LLM calls made, source-pattern references,
      warnings surfaced by the agent author.
    - ``warnings``: soft-fail messages (e.g. empty cart, LLM returned
      empty text). Empty list is the happy path.
    - ``llm_usage``: aggregate token / neuron / cost totals across all
      LLM calls made during ``execute()``. Feeds the neuron-cap counter
      in ``agents_routes.py`` so consumption is tracked per-request.
    """

    markdown: str
    cited_patterns: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Aggregate LLM usage across all synthesize() calls in this run.
    # Keys: tokens_used, neurons_used, cost_usd, calls_made.
    llm_usage: dict[str, Any] = field(default_factory=dict)

    def to_json_response(self) -> dict[str, Any]:
        """Serialize for the ``/api/agents/run`` route.

        Flat shape so the frontend can render + save + download without
        a schema migration when the response grows.
        """
        return {
            "markdown": self.markdown,
            "cited_patterns": list(self.cited_patterns),
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "llm_usage": dict(self.llm_usage),
        }


# ---------------------------------------------------------------------------
# Agent ABC — the interface every agent module implements
# ---------------------------------------------------------------------------

class Agent(ABC):
    """Base class for a single agent recipe.

    Every concrete agent is a class (not a function) so it can carry
    class-level metadata (``name``, ``display_name``, ``input_schema``,
    ``llm_dependency``) that the registry surfaces to the frontend list
    endpoint without instantiating.

    ``name`` MUST match the corresponding entry in
    ``frontend/src/agents/agent-definitions.ts``. Currently expected
    slugs (v1):

        auto_briefing
        qa
        professor
        cart_curator

    ``input_schema`` mirrors the frontend ``FieldSchema[]`` array shape
    verbatim. Same field types as Reports; agents reuse them so the
    frontend form-renderer can be shared.
    """

    # -- class-level metadata (overridden by subclasses) ------------------
    # Slug used in URLs + the registry + the frontend definition file.
    name: str = ""
    # User-facing label rendered on the card + agent header.
    display_name: str = ""
    # One-line description for the card grid.
    description: str = ""
    # List[dict] mirroring frontend ``FieldSchema``.
    input_schema: list[dict[str, Any]] = []
    # True iff ``execute()`` calls ``llm.synthesize()``. All 4 v1 agents
    # are True; cart_curator is optionally False for the deterministic-
    # only variant but ships True in v1.
    llm_dependency: bool = True

    # -- interface --------------------------------------------------------
    @abstractmethod
    def execute(
        self,
        cart_path: str,
        inputs: AgentInput,
        options: AgentOptions,
    ) -> AgentOutput:
        """Produce the agent's response for ``cart_path``.

        Implementations should:

        1. Load the cart via
           :class:`api.reports.cart_reader.CartHandle` (do NOT open the
           NPZ directly).
        2. If ``llm_dependency=True``: call
           :func:`api.llm.get_llm_adapter` and drive synthesize().
           EVERY prompt MUST be wrapped in
           :func:`api.agents.prompt.wrap_llama3_instruct` first — raw
           prompts trigger the "book review about environmental
           economics" completion-mode failure logged 2026-07-13.
        3. Return an :class:`AgentOutput` with ``markdown`` populated.
           Soft failures go in ``warnings``; hard failures raise.
        """
        ...

    # -- convenience helpers subclasses can lean on -----------------------
    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return the class-level metadata dict used by the registry."""
        return {
            "name": cls.name,
            "display_name": cls.display_name,
            "description": cls.description,
            "input_schema": list(cls.input_schema),
            "llm_dependency": cls.llm_dependency,
        }


__all__ = [
    "Agent",
    "AgentInput",
    "AgentOptions",
    "AgentOutput",
]
