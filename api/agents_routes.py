"""FastAPI routes for the Agents engine (2026-07-13 MVP).

Wires ``POST /api/agents/run`` + ``GET /api/agents/list`` +
``POST /api/agents/save_to_cart`` to :func:`api.agents.run_agent` and
the in-memory neuron-cap counter.

Route mapping
-------------
- ``GET  /api/agents/list``         → registered agents + display metadata.
- ``POST /api/agents/run``          → run an agent against a server cart.
- ``POST /api/agents/save_to_cart`` → v1 stub: log the save + return
                                       success. Real Membot write is v1.5.

Cart resolution
---------------
Delegated to ``reports_routes._resolve_cart_ref`` so Agents get the SAME
cart-lookup story as Reports (canonical + sandbox, TTL-race handling,
legacy-format friendly errors). This keeps the two surfaces feeling
identical from a user perspective and means agents pick up new resolver
features automatically. Import is at module top — deliberate — because
``reports_routes`` is already loaded before us (main.py include_router
order), so no cycle.

Neuron cap
----------
In-memory per-session counter, keyed by ``session_id`` from the request
body (client-provided) or the client IP as a fallback. Resets at UTC
midnight. Hard cap = 100 requests/day OR 10,000 neurons/day, whichever
hits first. 80% used → response header warning. 100% used → HTTP 429 +
``{error, reset_at}`` body. Session identity is not persistent; a browser
refresh doesn't clear the counter, but a browser instance restart with
a new session id does. v1.5 upgrades to Supabase-anchored per-user
counters — the wire shape stays identical.

Error codes
-----------
- ``400`` — unknown agent slug OR bad ``pattern_filter`` value.
- ``404`` — ``cart_ref`` doesn't resolve. Same failure taxonomy as
  Reports (``cart_not_found`` / ``cart_legacy_format`` /
  ``local_cart_unsupported``).
- ``410`` — sandbox cart evicted mid-request.
- ``422`` — request-body validation (Pydantic-level).
- ``429`` — neuron cap exceeded.
- ``500`` — internal error; server logs full traceback.
- ``503`` — LLM adapter unavailable (config or upstream error).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .agents import (  # noqa: F401  (import side-effects only)
    auto_briefing,
    qa,
    professor,
    cart_curator,
    free_agent,
)
from .agents import (
    REGISTRY,
    AgentOptions,
    list_agents,
    run_agent,
)
from .llm.adapter import LLMError
from .reports_routes import _resolve_cart_ref  # cart resolution reused

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neuron-cap config + storage
# ---------------------------------------------------------------------------

# Free-tier caps. v1 is a session-scoped counter; v1.5 replaces this with
# a per-user counter anchored to Supabase session. Both caps operate as
# OR — whichever hits first ends the day for that session.
MAX_REQUESTS_PER_DAY = 100
MAX_NEURONS_PER_DAY = 10_000
WARN_THRESHOLD = 0.80  # 80% used → warning header


@dataclass
class _SessionUsage:
    """Per-session daily usage record. Resets at UTC midnight.

    Reset works lazily — every ``incr`` / ``check_headers`` call
    inspects ``epoch_utc_day`` first and zeros the counters if we've
    rolled to a new day. That avoids a background sweep thread and
    keeps state simple.
    """
    requests: int = 0
    neurons: int = 0
    epoch_utc_day: int = field(default_factory=lambda: _utc_day_epoch())


# The in-memory table. Keyed by ``session_id`` (client-provided) or
# ``ip:<addr>`` (fallback). Mutation guarded by ``_usage_lock`` — the
# route is async but the counter is a small critical section so a
# threading Lock is enough (no need for asyncio.Lock overhead).
_USAGE: dict[str, _SessionUsage] = {}
_usage_lock = threading.Lock()


def _utc_day_epoch() -> int:
    """Integer count of days since epoch in UTC. Rolls over at midnight UTC."""
    return int(datetime.now(timezone.utc).timestamp() // 86400)


def _reset_at_iso() -> str:
    """ISO timestamp of the next UTC midnight — for the 429 body's
    ``reset_at`` field so the UI can render a friendly countdown."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return tomorrow.isoformat()


def _session_key(session_id: Optional[str], request: Request) -> str:
    """Derive a stable-per-session identifier.

    Prefers a caller-supplied ``session_id`` (opaque string; frontend
    generates once per browser instance and reuses). Falls back to
    client IP so anonymous callers still hit the cap. Empty caller ids
    fall through to IP so a bogus empty string doesn't share one row.
    """
    if session_id and session_id.strip():
        return f"sid:{session_id.strip()}"
    # Trust X-Forwarded-For only if we're behind a known proxy — on the
    # droplet nginx is the proxy so this is fine. Otherwise use the
    # direct peer address. This isn't security-critical (the cap is a
    # UX guardrail, not an auth boundary).
    client_host = "unknown"
    if request.client:
        client_host = request.client.host or "unknown"
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if fwd:
        client_host = fwd
    return f"ip:{client_host}"


def _get_usage(key: str) -> _SessionUsage:
    """Return the usage record for ``key``, resetting if a new UTC day.

    Not thread-safe on its own; callers hold ``_usage_lock``.
    """
    today = _utc_day_epoch()
    rec = _USAGE.get(key)
    if rec is None:
        rec = _SessionUsage(epoch_utc_day=today)
        _USAGE[key] = rec
    elif rec.epoch_utc_day != today:
        rec.requests = 0
        rec.neurons = 0
        rec.epoch_utc_day = today
    return rec


def _cap_check(key: str) -> Optional[dict[str, Any]]:
    """Return None if under cap; a 429-shaped detail dict if over.

    Called BEFORE the LLM run — checks request count only, since neuron
    usage isn't known until after the call. Post-call incr happens
    unconditionally so subsequent requests see the running total.
    """
    with _usage_lock:
        rec = _get_usage(key)
        if rec.requests >= MAX_REQUESTS_PER_DAY:
            return {
                "error": "quota_exceeded",
                "message": (
                    f"Daily agent quota reached ({MAX_REQUESTS_PER_DAY} "
                    f"requests). Comes back tomorrow, or upgrade for a "
                    f"higher tier."
                ),
                "reset_at": _reset_at_iso(),
                "cap_hit": "requests",
            }
        if rec.neurons >= MAX_NEURONS_PER_DAY:
            return {
                "error": "quota_exceeded",
                "message": (
                    f"Daily neuron budget reached "
                    f"({MAX_NEURONS_PER_DAY} neurons). Comes back tomorrow, "
                    f"or upgrade for a higher tier."
                ),
                "reset_at": _reset_at_iso(),
                "cap_hit": "neurons",
            }
    return None


def _cap_incr(key: str, neurons_used: int) -> _SessionUsage:
    """Increment the counter after a successful run. Returns fresh snapshot."""
    with _usage_lock:
        rec = _get_usage(key)
        rec.requests += 1
        if neurons_used > 0:
            rec.neurons += neurons_used
        return _SessionUsage(
            requests=rec.requests,
            neurons=rec.neurons,
            epoch_utc_day=rec.epoch_utc_day,
        )


def _cap_headers(response: Response, rec: _SessionUsage) -> None:
    """Attach cap-status headers to ``response``. Warning at 80% used.

    Headers:
        X-Agent-Requests-Used, X-Agent-Requests-Max
        X-Agent-Neurons-Used, X-Agent-Neurons-Max
        X-Agent-Quota-Warning (present iff >= 80% of either cap)
    """
    response.headers["X-Agent-Requests-Used"] = str(rec.requests)
    response.headers["X-Agent-Requests-Max"] = str(MAX_REQUESTS_PER_DAY)
    response.headers["X-Agent-Neurons-Used"] = str(rec.neurons)
    response.headers["X-Agent-Neurons-Max"] = str(MAX_NEURONS_PER_DAY)
    req_frac = rec.requests / MAX_REQUESTS_PER_DAY
    neu_frac = rec.neurons / MAX_NEURONS_PER_DAY
    if req_frac >= WARN_THRESHOLD or neu_frac >= WARN_THRESHOLD:
        pct = int(max(req_frac, neu_frac) * 100)
        response.headers["X-Agent-Quota-Warning"] = (
            f"You've used {pct}% of today's agent budget."
        )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunAgentRequest(BaseModel):
    """POST body for ``/api/agents/run``."""

    agent_slug: str = Field(..., description="Registered agent slug.")
    cart_ref: str = Field(..., description="Server cart identifier.")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-agent form values keyed by field name.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque per-browser session id for neuron-cap keying.",
    )


class RunAgentResponse(BaseModel):
    """Success payload for ``/api/agents/run``.

    ``run_id`` is a fresh UUID assigned per run — used by save_to_cart
    to reference the specific execution the user wants persisted.
    """
    run_id: str
    markdown: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    cited_patterns: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    agent_slug: str
    generated_at: str
    elapsed_ms: int


class SaveToCartRequest(BaseModel):
    """POST body for ``/api/agents/save_to_cart``."""

    run_id: str = Field(..., description="Run id from a prior /run response.")
    cart_ref: Optional[str] = Field(
        default=None,
        description="Destination cart. v1: ignored (real save is v1.5).",
    )
    session_id: Optional[str] = Field(default=None)


class SaveToCartResponse(BaseModel):
    """Success payload for ``/api/agents/save_to_cart``."""

    success: bool
    saved_at: str
    message: str
    run_id: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/list")
async def get_agent_list() -> dict[str, Any]:
    """Return every registered agent's metadata + cap config."""
    return {
        "agents": list_agents(),
        "caps": {
            "max_requests_per_day": MAX_REQUESTS_PER_DAY,
            "max_neurons_per_day": MAX_NEURONS_PER_DAY,
            "warn_threshold": WARN_THRESHOLD,
        },
    }


@router.post("/run", response_model=RunAgentResponse)
async def run_agent_route(
    req: RunAgentRequest, request: Request, response: Response,
) -> RunAgentResponse:
    """Dispatch an agent against a server-side cart."""
    slug = (req.agent_slug or "").strip()

    # -- 1. Unknown slug → 400 (agents never "coming soon" — either
    # registered or not) ------------------------------------------------
    if slug not in REGISTRY:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_agent",
                "message": (
                    f"No agent registered under slug {slug!r}. "
                    f"Registered: {sorted(REGISTRY.keys())}."
                ),
                "agent_slug": slug,
            },
        )

    # -- 2. Neuron-cap check (before spending any tokens) ---------------
    session_key = _session_key(req.session_id, request)
    cap_fail = _cap_check(session_key)
    if cap_fail is not None:
        raise HTTPException(status_code=429, detail=cap_fail)

    # -- 3. Resolve cart_ref → filesystem path --------------------------
    resolution = _resolve_cart_ref(req.cart_ref)
    if resolution.path is None:
        failure = resolution.failure or "cart_not_found"
        if failure == "local_cart_unsupported":
            detail_msg = (
                "Agents require a server-side cart. The selected cart is a "
                "browser-only LocalCart — pick a server cart to run this "
                "agent."
            )
        elif failure == "cart_legacy_format":
            detail_msg = (
                f"'{req.cart_ref}' uses a legacy format that Agents can't "
                f"read yet. Rebuild as .cart.npz and try again."
            )
        else:
            detail_msg = (
                f"Cart '{req.cart_ref}' isn't available on the server "
                f"anymore. Pick another cart above to continue."
            )
        raise HTTPException(
            status_code=404,
            detail={
                "error": failure,
                "message": detail_msg,
                "cart_ref": req.cart_ref,
            },
        )
    cart_path = resolution.path
    cart_location = resolution.location

    # -- 4. Compute remaining neuron budget for the LLM call ------------
    # Convert remaining budget into a max_tokens ceiling so a long-response
    # agent doesn't accidentally blow past the daily cap on one call.
    with _usage_lock:
        rec = _get_usage(session_key)
        remaining_neurons = max(0, MAX_NEURONS_PER_DAY - rec.neurons)
    # CF adapter reports neurons_used per synthesis. A rough neurons-per-
    # token conversion doesn't exist upstream, so we cap max_tokens at
    # the adapter default when there's headroom, and squeeze it when
    # there isn't. Aggressive but safe: the request may fail loud rather
    # than silently truncate deep into the response.
    options = AgentOptions(
        max_llm_calls=1,
        max_tokens=2048 if remaining_neurons >= 1000 else 512,
        include_source_refs=True,
        verbose=False,
    )

    # -- 5. Dispatch -----------------------------------------------------
    t0 = time.time()
    try:
        output = run_agent(
            agent_name=slug,
            cart_path=cart_path,
            raw_inputs=dict(req.inputs or {}),
            options=options,
        )
    except ValueError as exc:
        # Agent modules raise ValueError for input-validation problems
        # (missing required field, empty question, bad enum value).
        logger.warning(
            "[agents] ValueError running %r on %r: %s",
            slug, cart_path, exc,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_input",
                "message": str(exc),
                "agent_slug": slug,
            },
        )
    except FileNotFoundError as exc:
        # Sandbox TTL race — same taxonomy as Reports.
        if cart_location == "sandbox":
            logger.warning(
                "[agents] Sandbox cart evicted mid-request for %r: %s",
                cart_path, exc,
            )
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "sandbox_cart_expired",
                    "message": (
                        "This sandbox cart was evicted before the agent "
                        "could complete. Sandbox carts have a limited "
                        "lifetime — re-upload if you'd like to try again."
                    ),
                    "cart_ref": req.cart_ref,
                },
            )
        logger.warning(
            "[agents] Cart disappeared mid-request for %r: %s",
            cart_path, exc,
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": "cart_not_found",
                "message": str(exc),
                "cart_ref": req.cart_ref,
            },
        )
    except LLMError as exc:
        # Adapter-config problem — bubble up as 503 so the UI knows this
        # isn't the user's fault.
        logger.error("[agents] LLM adapter unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_provider_unavailable",
                "message": str(exc),
                "agent_slug": slug,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[agents] Unexpected error running %r on %r:\n%s",
            slug, cart_path, traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": (
                    "Agent execution failed. This has been logged; try "
                    "again or contact support if it persists."
                ),
                "agent_slug": slug,
            },
        )

    elapsed_ms = int((time.time() - t0) * 1000)

    # -- 6. Post-call cap update ----------------------------------------
    neurons_used = 0
    llm_usage = output.llm_usage or {}
    if isinstance(llm_usage.get("neurons_used"), (int, float)):
        neurons_used = int(llm_usage["neurons_used"])
    fresh_rec = _cap_incr(session_key, neurons_used)
    _cap_headers(response, fresh_rec)

    # -- 7. Register the run for save_to_cart ---------------------------
    run_id = uuid.uuid4().hex
    _register_run(run_id, session_key, slug, req.cart_ref, output.markdown)

    # -- 8. Bundle route-level context into metadata --------------------
    meta = dict(output.metadata) if output.metadata else {}
    meta.setdefault("route_elapsed_ms", elapsed_ms)
    meta.setdefault("cart_ref", req.cart_ref)
    if cart_location:
        meta.setdefault("cart_location", cart_location)
    meta["quota_requests_used"] = fresh_rec.requests
    meta["quota_neurons_used"] = fresh_rec.neurons

    return RunAgentResponse(
        run_id=run_id,
        markdown=output.markdown,
        metadata=meta,
        cited_patterns=list(output.cited_patterns),
        warnings=list(output.warnings),
        llm_usage=llm_usage,
        agent_slug=slug,
        generated_at=datetime.now(timezone.utc).isoformat(),
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Save-to-cart stub
# ---------------------------------------------------------------------------
#
# v1 keeps a tiny in-memory record of every run so save_to_cart has a
# plausible object to reference. v1.5 replaces this with a real Membot
# write — the wire shape (run_id → success/saved_at) stays identical so
# the frontend doesn't need to change when the real path lands.

@dataclass
class _RunRecord:
    session_key: str
    agent_slug: str
    cart_ref: str
    markdown_snippet: str  # first 400 chars, for logging only
    created_at: str


_RUNS: dict[str, _RunRecord] = {}
_runs_lock = threading.Lock()
# Cap the run cache so a long-lived process doesn't grow unbounded. FIFO
# eviction; a user who saves >_MAX_RUN_CACHE hours after run time gets a
# friendly "run expired" error rather than a silent failure.
_MAX_RUN_CACHE = 500


def _register_run(
    run_id: str, session_key: str, agent_slug: str,
    cart_ref: str, markdown: str,
) -> None:
    """Register a run so save_to_cart can reference it later."""
    with _runs_lock:
        _RUNS[run_id] = _RunRecord(
            session_key=session_key,
            agent_slug=agent_slug,
            cart_ref=cart_ref,
            markdown_snippet=markdown[:400],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        # FIFO evict when over cap. Python 3.7+ dicts preserve insertion
        # order so we pop the oldest.
        while len(_RUNS) > _MAX_RUN_CACHE:
            _RUNS.pop(next(iter(_RUNS)))


@router.post("/save_to_cart", response_model=SaveToCartResponse)
async def save_to_cart_route(
    req: SaveToCartRequest, request: Request,
) -> SaveToCartResponse:
    """v1 STUB — logs the intended save + returns success.

    Real Membot cart write is v1.5 (when user-cart infrastructure is
    fully wired). The wire shape here is the shape the real path will
    return so the frontend doesn't need to change when the real save
    lands. See ``docs/vps-internal/Agents Tab Design 2026-07-13.md``
    "Save-to-cart provenance shape" for the eventual Membot pattern
    structure.
    """
    with _runs_lock:
        run = _RUNS.get(req.run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "run_not_found",
                "message": (
                    "Run id not found. Either the id is wrong or the "
                    "run has aged out of the server cache."
                ),
                "run_id": req.run_id,
            },
        )
    logger.info(
        "[agents.save_to_cart STUB] session=%s agent=%s source_cart=%s "
        "dest_cart=%s snippet=%r",
        run.session_key, run.agent_slug, run.cart_ref,
        req.cart_ref or "<default>", run.markdown_snippet[:120],
    )
    return SaveToCartResponse(
        success=True,
        saved_at=datetime.now(timezone.utc).isoformat(),
        message=(
            "Saved to cart (v1 stub — real Membot write ships in v1.5)."
        ),
        run_id=req.run_id,
    )


__all__ = ["router"]
