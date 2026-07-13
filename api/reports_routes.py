"""FastAPI routes for the Reports engine (Wave-2 dispatch).

Wires ``POST /api/reports/generate`` to :func:`api.reports.run_report`.
The engine itself (``api/reports/*.py``) stays untouched — this module is
the composition layer: import the 5 Wave-1 report modules to trigger
``@register_report``, resolve ``cart_ref`` to a filesystem path, dispatch,
serialize.

Route mapping
-------------
- ``POST /api/reports/generate``  → run a report against a server cart.
- ``GET  /api/reports/list``      → registered slugs + display metadata
                                    (thin passthrough of :func:`list_reports`).

Contract details for ``/generate`` — see the request/response models below.

Error codes:

- ``404`` — unknown report slug (not in the frontend definitions file)
  OR ``cart_ref`` doesn't resolve to a real ``.cart.npz`` on the server.
- ``422`` — schema validation error on the payload itself (Pydantic-level
  missing/wrong type). Also used when the report author's ``generate()``
  raised :class:`ValueError` for a missing required input.
- ``501`` — slug exists in the frontend Wave-2 whitelist (Timeline, Trend,
  Financial Rollup, Executive TL;DR) but hasn't been added to the backend
  registry yet. Body carries a "future release" hint the UI renders as
  a friendly note.
- ``500`` — any other runtime error; the traceback is logged server-side.

The route is intentionally synchronous inside a threadpool — the 5 Wave-1
reports are pure Python + numpy over already-loaded arrays and finish in
well under a second for the cart sizes we care about; async doesn't buy
us anything and complicates error propagation. If a Wave-2 report needs
to stream, add a second route rather than making this one async.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import the report modules so ``@register_report`` fires. Without these
# imports the REGISTRY dict is empty when the route boots (api/reports
# __init__.py deliberately doesn't import individual reports — the brief
# forbids modifying it, so we do the wiring here instead).
from .reports import (  # noqa: F401  (import side-effects only)
    summary,
    entity_rollup,
    change_log,
    comparison,
    coverage,
)
from .reports import (
    REGISTRY,
    ReportOptions,
    list_reports,
    run_report,
)
from .cartridge_io import DATA_DIR, find_companion_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wave 2 whitelist — slugs the frontend knows about but the backend doesn't
# ---------------------------------------------------------------------------

# These match ``frontend/src/reports/report-definitions.ts`` slugs that
# have no registered backend Report class. When a user clicks Generate on
# one of these, the route responds with 501 + a friendly "future release"
# hint. Kept in sync manually — when a Wave-2 module lands and registers
# itself, drop its slug from this set.
WAVE2_KNOWN_SLUGS: frozenset[str] = frozenset({
    "timeline",
    "trend",
    "financial_rollup",
    "tldr",
})


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateReportRequest(BaseModel):
    """POST body for ``/api/reports/generate``.

    Shape mirrors the brief:

    - ``report_slug`` — one of the registered report names (or a Wave-2
      known slug → 501).
    - ``cart_ref`` — server cart identifier. Resolved to a ``.cart.npz``
      via ``find_companion_file`` (see :func:`_resolve_cart_ref`).
    - ``cart_ref_b`` — optional secondary cart id. Reserved for a future
      two-cart comparison mode; currently ignored by the Wave-1
      ComparisonReport, which slices one cart via subset queries. Kept
      in the surface so the frontend contract doesn't need to churn when
      the two-cart mode lands. (Andy 2026-07-13 brief called this
      required for comparison; the actual ComparisonReport takes one cart
      + two subset queries via ``inputs``, so leaving as optional avoids
      forcing users to send a value the engine won't use.)
    - ``inputs`` — untyped dict passed straight into ``ReportInput.raw``.
    """

    report_slug: str = Field(..., description="Registered report slug.")
    cart_ref: str = Field(..., description="Server cart identifier — cart name or filename.")
    cart_ref_b: Optional[str] = Field(
        default=None,
        description="Optional secondary cart identifier (reserved).",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-report form values keyed by field name.",
    )


class GenerateReportResponse(BaseModel):
    """Success payload for ``/api/reports/generate``."""

    markdown: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    csv_data: Optional[str] = None
    html_extra: Optional[str] = None
    report_slug: str
    generated_at: str


# ---------------------------------------------------------------------------
# cart_ref resolution
# ---------------------------------------------------------------------------

def _resolve_cart_ref(cart_ref: str) -> Optional[str]:
    """Resolve ``cart_ref`` to an absolute path to a ``.cart.npz`` file.

    Accepts several friendly shapes so the frontend doesn't have to
    normalize:

    - Absolute path to an existing ``.cart.npz``.
    - Bare cart name (as it appears in ``/api/cartridges``).
    - Cart name with the ``.cart.npz`` suffix already attached.

    Returns ``None`` when nothing resolves. The route converts that to a
    404. We only accept ``.cart.npz`` — the Reports engine's
    :class:`CartHandle` opens NPZ files, and legacy ``.pkl`` carts would
    fail deeper in with a less-useful error. That's a conscious cut for
    Wave-1; a Wave-2 upgrade could add a lightweight ``.pkl`` → NPZ
    coercion path if any pitch cart needs it.

    Rejects local-only cart selectors from the frontend that carry the
    ``local:`` prefix — LocalCarts are browser-side arrays, not files on
    disk. The 404 body names the prefix explicitly so the UI can display
    a friendlier explanation than "cart not found."
    """
    if not cart_ref:
        return None

    ref = cart_ref.strip()

    # LocalCart guard — frontend uses "local:<name>" for browser-side carts
    # that never touched the server. Reports need a file on disk.
    if ref.startswith("local:"):
        return None

    # Strip an optional "server:" prefix from the frontend cart selector,
    # which annotates entries as "local:foo" or "server:bar" — reports
    # only run against server carts, so the prefix is redundant here.
    if ref.startswith("server:"):
        ref = ref[len("server:"):]

    # 1. Absolute path to an existing NPZ — trust and return.
    import os as _os
    if _os.path.isabs(ref) and _os.path.exists(ref) and ref.endswith(".cart.npz"):
        return ref

    # 2. Bare filename ending in .cart.npz — search cartridge dirs.
    if ref.endswith(".cart.npz"):
        stem = ref[: -len(".cart.npz")]
    else:
        stem = ref

    # find_companion_file walks the cartridge dirs (DATA_DIR + SAMPLE_DIR)
    # so this handles both the pitch carts and sample_data carts.
    resolved = find_companion_file(stem, ".cart.npz")
    if resolved:
        return resolved

    # 3. Last-ditch: a raw path relative to cwd.
    if _os.path.exists(ref) and ref.endswith(".cart.npz"):
        return _os.path.abspath(ref)

    return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/list")
async def get_report_list() -> dict[str, Any]:
    """Return every registered report's metadata.

    Powers a future "auto-generated card grid" in the UI; for now the
    frontend has its own ``report-definitions.ts`` shell and this is
    convenient for debugging. Cheap to compute — just describes classes.
    """
    return {
        "reports": list_reports(),
        "wave2_pending": sorted(WAVE2_KNOWN_SLUGS),
    }


@router.post("/generate", response_model=GenerateReportResponse)
async def generate_report(req: GenerateReportRequest) -> GenerateReportResponse:
    """Dispatch a report against a server-side cart.

    Error semantics documented at module top. On success returns the
    :class:`ReportOutput` fields flattened into the response envelope +
    an ``ISO-8601`` ``generated_at`` timestamp for the UI header.
    """
    slug = (req.report_slug or "").strip()

    # -- 1. Wave-2 known-slug shortcut (return 501 with a friendly hint) --
    # Order matters: check Wave-2 BEFORE checking the registry, otherwise
    # a Wave-2 slug that happens to also be in REGISTRY would be treated
    # as "not built" — which is only true if it isn't registered yet.
    if slug in WAVE2_KNOWN_SLUGS and slug not in REGISTRY:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "not_yet_available",
                "message": f"The '{slug}' report will be available in a future release.",
                "report_slug": slug,
            },
        )

    # -- 2. Unknown slug → 404 --
    if slug not in REGISTRY:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_report",
                "message": (
                    f"No report registered under slug {slug!r}. "
                    f"Registered: {sorted(REGISTRY.keys())}."
                ),
                "report_slug": slug,
            },
        )

    # -- 3. Resolve cart_ref → filesystem path --
    cart_path = _resolve_cart_ref(req.cart_ref)
    if cart_path is None:
        # Distinguish "local-cart" from generic not-found so the UI can
        # render a targeted message rather than a bare 404 dialog.
        is_local = req.cart_ref.startswith("local:")
        detail_msg = (
            "Reports require a server-side cart. The selected cart is a "
            "browser-only LocalCart — pick a server cart in the cart "
            "selector to generate a report."
            if is_local
            else f"Cart {req.cart_ref!r} not found on the server. "
                 f"Only .cart.npz files under the cartridges/ dir are supported."
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": "cart_not_found" if not is_local else "local_cart_unsupported",
                "message": detail_msg,
                "cart_ref": req.cart_ref,
            },
        )

    # -- 4. Dispatch --
    # Options: Wave-1 reports are LLM-free, so max_llm_calls stays at 0.
    # If ever we let this route dispatch TL;DR, the caller (or an
    # entitlement check here) will bump this — for now, leaving the guard
    # in place at 0 means an accidentally-registered LLM report would
    # short-circuit with a clear error rather than hitting Cloudflare.
    options = ReportOptions(
        max_llm_calls=0,
        include_source_refs=True,
        verbose=False,
    )

    t0 = time.time()
    try:
        output = run_report(
            report_name=slug,
            cart_path=cart_path,
            raw_inputs=dict(req.inputs or {}),
            options=options,
        )
    except ValueError as exc:
        # Report modules raise ValueError for input-validation problems
        # (missing required field, bad choice value). Surface as 422 so
        # the UI can render a form-level error rather than a scary 500.
        logger.warning(
            "[reports] ValueError running %r on %r: %s",
            slug, cart_path, exc,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_input",
                "message": str(exc),
                "report_slug": slug,
            },
        )
    except FileNotFoundError as exc:
        # Cart path passed our resolver but the file disappeared between
        # resolve and open. Treat as 404 for the UI's benefit.
        logger.warning(
            "[reports] Cart disappeared mid-request for %r: %s",
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
    except HTTPException:
        raise
    except Exception as exc:
        # Unexpected runtime error. Log the full traceback for post-mortem
        # then surface a short user-facing message. Sentry (initialized in
        # main.py) captures the exception via its FastAPI integration so
        # we don't need to explicitly report here.
        logger.error(
            "[reports] Unexpected error running %r on %r:\n%s",
            slug, cart_path, traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": (
                    "Report generation failed. This has been logged; "
                    "try again or contact support if it persists."
                ),
                "report_slug": slug,
            },
        )

    elapsed_ms = int((time.time() - t0) * 1000)

    # Include ambient context in metadata for the UI's audit footer. We
    # only ADD keys the report author didn't set, so a report that wants
    # to override elapsed_ms / cart_path stays in charge.
    meta = dict(output.metadata) if output.metadata else {}
    meta.setdefault("route_elapsed_ms", elapsed_ms)
    meta.setdefault("cart_ref", req.cart_ref)
    if req.cart_ref_b:
        meta.setdefault("cart_ref_b", req.cart_ref_b)

    return GenerateReportResponse(
        markdown=output.markdown,
        metadata=meta,
        warnings=list(output.warnings),
        csv_data=output.csv_data,
        html_extra=output.html_extra,
        report_slug=slug,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = ["router", "WAVE2_KNOWN_SLUGS"]
