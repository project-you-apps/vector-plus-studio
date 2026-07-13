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
import os
import re
import time
import traceback
from dataclasses import dataclass
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
from .cartridge_io import DATA_DIR, find_companion_file, get_cartridge_dirs
# Sandbox uploads land here as ``<12-hex-uuid>_<safe_name>.cart.npz``. The
# Reports-scope resolver walks this directory in addition to the canonical
# cartridge dirs so sandbox-uploaded carts can drive report generation
# without a further cart-mount step. See ``find_report_cart`` below.
from .uploads import SANDBOX_DIR

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
#
# 2026-07-13 (Wave-1 UX bug follow-up): the resolver now walks three
# directories rather than two (canonical DATA_DIR + SAMPLE_DIR + the
# sandbox-uploads subdir). Sandbox uploads land in
# ``cartridges/_session_uploads/<uuid>_<name>.cart.npz`` and previously
# 404'd here even though ``/api/reports/carts`` was starting to list them
# (via the frontend's cartridges list, which is populated by a different
# endpoint that DOES enumerate the sandbox). Splitting the resolver from
# ``find_companion_file`` keeps the mount/save flows untouched while
# giving Reports its own view of the world.

# Sandbox filenames carry a 12-hex-char UUID prefix (see uploads.py
# ``upload_cartridge()`` -> ``uuid.uuid4().hex[:12]``). This regex lets us
# strip the prefix for display and for suffix-match lookup when the
# frontend sends the bare human-readable stem.
_SANDBOX_PREFIX_RE = re.compile(r"^([0-9a-f]{12})_(.+)$")


def _strip_sandbox_prefix(filename_stem: str) -> Optional[str]:
    """Return the human-readable portion of a sandbox filename stem.

    Sandbox uploads are named ``<12-hex-uuid>_<safe_name>.cart.npz``.
    This returns just ``<safe_name>`` when the stem matches that pattern,
    otherwise ``None`` (i.e. the stem is not sandbox-shaped).
    """
    m = _SANDBOX_PREFIX_RE.match(filename_stem)
    return m.group(2) if m else None


def find_report_cart(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(path, location)`` for a ``.cart.npz`` matching ``stem``.

    Walks, in order:

      1. ``cartridges/`` + ``sample_data/`` top-level (canonical). This
         mirrors ``find_companion_file`` behavior — Reports gets the
         same set of "real" carts as mount/save.
      2. ``cartridges/_session_uploads/`` (sandbox). Files here have a
         ``<12-hex-uuid>_`` prefix; both the raw stem and the stripped
         form resolve.

    Returns ``(abs_path, "canonical" | "sandbox")`` on hit,
    ``(None, None)`` on miss. Canonical wins over sandbox when the same
    stem exists in both — the demo droplet ships a curated set that
    should always outrank a user upload of the same name.

    Sandbox resolution accepts both forms the frontend might send:
    the bare human stem (``Demo-Report-Cart-II``) matches sandbox files
    whose name ends in ``_<stem>.cart.npz``, and the already-uuid-prefixed
    stem (``e0fe920f325d_Demo-Report-Cart-II`` — what the dropdown shows)
    matches by direct filename. When multiple sandbox files match the
    bare form, the most recently modified one wins — that's what a user
    who re-uploaded a cart with the same name intends.
    """
    if not stem:
        return (None, None)

    # 1. Canonical dirs — reuse the existing helper so anything discoverable
    # by mount / list is discoverable here.
    canonical = find_companion_file(stem, ".cart.npz")
    if canonical:
        return (canonical, "canonical")

    # 2. Sandbox dir. Missing dir is fine (fresh droplet before first upload).
    sandbox_dir = str(SANDBOX_DIR)
    try:
        entries = os.listdir(sandbox_dir)
    except (FileNotFoundError, OSError):
        return (None, None)

    # 2a. Direct match — frontend sent the full <uuid>_<name> stem.
    direct = os.path.join(sandbox_dir, f"{stem}.cart.npz")
    if os.path.exists(direct):
        return (direct, "sandbox")

    # 2b. Suffix match — frontend sent the bare human stem. Look for any
    # sandbox file whose stripped-prefix name equals it.
    candidates: list[str] = []
    for name in entries:
        if not name.endswith(".cart.npz"):
            continue
        stripped = _strip_sandbox_prefix(name[: -len(".cart.npz")])
        if stripped == stem:
            candidates.append(name)

    if candidates:
        # Most recent mtime wins so a re-upload of the same name resolves
        # to the new file, not a stale one. TTL cleanup would eventually
        # evict the old one anyway.
        candidates.sort(
            key=lambda n: os.path.getmtime(os.path.join(sandbox_dir, n)),
            reverse=True,
        )
        return (os.path.join(sandbox_dir, candidates[0]), "sandbox")

    return (None, None)


def _find_legacy_pkl(stem: str) -> Optional[str]:
    """Return path to a legacy ``.pkl`` cart with this stem, or ``None``.

    Used only for error-message specificity — if a stem doesn't resolve
    as ``.cart.npz`` but does exist as ``.pkl``, the endpoint returns
    ``cart_legacy_format`` rather than the generic ``cart_not_found``.
    Reports engine only reads NPZ; a legacy cart is a real cart the user
    just needs to convert.
    """
    for d in get_cartridge_dirs():
        p = os.path.join(d, f"{stem}.pkl")
        if os.path.exists(p):
            return p
    return None


@dataclass
class ResolvedCart:
    """Discriminated result of ``_resolve_cart_ref``.

    Exactly one of ``path`` and ``failure`` will be set. ``location`` is
    only meaningful when ``path`` is set — it tells the endpoint whether
    a mid-request FileNotFoundError should be reported as
    ``sandbox_cart_expired`` (short-TTL race) or a generic
    ``cart_not_found`` (probably an admin-side deletion, rarer).
    """
    path: Optional[str] = None
    location: Optional[str] = None
    # 'local_cart_unsupported' | 'cart_legacy_format' | 'cart_not_found'
    failure: Optional[str] = None


def _resolve_cart_ref(cart_ref: str) -> ResolvedCart:
    """Resolve ``cart_ref`` to a ``.cart.npz`` path with failure taxonomy.

    Accepts several friendly shapes so the frontend doesn't have to
    normalize:

    - Absolute path to an existing ``.cart.npz``.
    - Bare cart name (as it appears in ``/api/cartridges``).
    - Cart name with the ``.cart.npz`` suffix already attached.
    - Sandbox filename with the ``<12-hex-uuid>_`` prefix present.
    - Bare human stem for a sandbox upload (matches by suffix).

    Returns a :class:`ResolvedCart`. On success ``path`` + ``location``
    are set. On failure ``failure`` is set to one of three tags the
    endpoint maps directly to error codes for the amber panel.

    Rejects local-only cart selectors from the frontend that carry the
    ``local:`` prefix — LocalCarts are browser-side arrays, not files on
    disk. The 404 body names the prefix explicitly so the UI can display
    a friendlier explanation than "cart not found."
    """
    if not cart_ref:
        return ResolvedCart(failure="cart_not_found")

    ref = cart_ref.strip()

    # LocalCart guard — frontend uses "local:<name>" for browser-side carts
    # that never touched the server. Reports need a file on disk.
    if ref.startswith("local:"):
        return ResolvedCart(failure="local_cart_unsupported")

    # Strip an optional "server:" prefix from the frontend cart selector,
    # which annotates entries as "local:foo" or "server:bar" — reports
    # only run against server carts, so the prefix is redundant here.
    if ref.startswith("server:"):
        ref = ref[len("server:"):]

    # 1. Absolute path to an existing NPZ — trust and return.
    if os.path.isabs(ref) and os.path.exists(ref) and ref.endswith(".cart.npz"):
        return ResolvedCart(path=ref, location="canonical")

    # 2. Normalize to a bare stem for dir-walking lookup.
    if ref.endswith(".cart.npz"):
        stem = ref[: -len(".cart.npz")]
    else:
        stem = ref

    # 3. Reports-scope resolver — canonical dirs first, then sandbox.
    path, location = find_report_cart(stem)
    if path:
        return ResolvedCart(path=path, location=location)

    # 4. Legacy-pkl hint — better error copy than generic "not found".
    if _find_legacy_pkl(stem):
        return ResolvedCart(failure="cart_legacy_format")

    # 5. Last-ditch: a raw path relative to cwd.
    if os.path.exists(ref) and ref.endswith(".cart.npz"):
        return ResolvedCart(path=os.path.abspath(ref), location="canonical")

    return ResolvedCart(failure="cart_not_found")


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


# ---------------------------------------------------------------------------
# Cart enumeration for the Reports tab selector (2026-07-13)
# ---------------------------------------------------------------------------
#
# The Reports tab needs to know which server carts the report engine can
# actually open — i.e. which stems have a companion ``.cart.npz`` on disk.
# Legacy ``.pkl`` carts (v7/v8) will 404 in ``/generate`` because
# ``CartHandle`` only reads NPZ; the frontend uses this endpoint to grey out
# the incompatible entries with a helpful tooltip rather than let the user
# discover the incompatibility via a scary 404 during Generate.
#
# The ``list_cartridges()`` helper in cartridge_io returns richer per-cart
# info (size, brain/sig companion presence, etc.) but it iterates ``.pkl``
# and ``.cart.npz`` separately and is used elsewhere. We do a targeted walk
# here to keep the payload lean and the compatibility rule co-located with
# the Reports engine.


@router.get("/carts")
async def list_report_carts() -> dict[str, Any]:
    """Enumerate server carts with per-cart report compatibility.

    Sources three directories:

    - Canonical ``cartridges/`` + ``sample_data/`` (top-level) — the
      curated pitch/demo carts.
    - Sandbox ``cartridges/_session_uploads/`` — short-TTL user uploads
      via ``POST /api/cartridges/upload``. These carry a
      ``<12-hex-uuid>_`` filename prefix; the ``id`` we return keeps the
      prefix (that's the full stem the resolver matches on) but
      ``display_name`` strips it plus tags "(sandbox)" so the selector
      renders a human-friendly label.

    Response shape (per entry)::

        {
          "id": "gutenberg-poetry",                     # canonical
          "display_name": "gutenberg-poetry",
          "report_compatible": true,
          "format": "npz",
          "location": "canonical"
        }
        {
          "id": "e0fe920f325d_Demo-Report-Cart-II",     # sandbox
          "display_name": "Demo-Report-Cart-II (sandbox)",
          "report_compatible": true,
          "format": "npz",
          "location": "sandbox"
        }

    Ordering:
      1. Canonical ``.cart.npz`` (alphabetical by id).
      2. Sandbox ``.cart.npz`` (alphabetical by display_name).
      3. Legacy ``.pkl`` (alphabetical by id).

    Deduped by (id, location) — a stem that exists in both canonical
    and sandbox surfaces twice under different ids (canonical id is the
    bare stem; sandbox id is uuid-prefixed) so users can pick either.
    A stem present as both ``.cart.npz`` and ``.pkl`` in the canonical
    dirs is listed once, as compatible.
    """
    # -- Canonical dirs -----------------------------------------------------
    # {stem: format} — first-seen wins for a given stem, but we upgrade a
    # "pkl" seen earlier if a matching ".cart.npz" is discovered later so
    # the same underlying cart isn't listed twice.
    canonical_stem_to_format: dict[str, str] = {}
    for d in get_cartridge_dirs():
        try:
            entries_in_dir = os.listdir(d)
        except (FileNotFoundError, OSError):
            continue
        for f in entries_in_dir:
            if f.endswith(".cart.npz"):
                stem = f[: -len(".cart.npz")]
                canonical_stem_to_format[stem] = "npz"  # npz always wins
            elif f.endswith(".pkl"):
                stem = f[: -len(".pkl")]
                # Only mark as pkl if we haven't already seen an npz for this
                # stem — otherwise leave the npz mark in place.
                canonical_stem_to_format.setdefault(stem, "pkl")

    canonical_npz: list[dict[str, Any]] = []
    legacy_pkl: list[dict[str, Any]] = []
    for stem, fmt in canonical_stem_to_format.items():
        compat = fmt == "npz" and find_companion_file(stem, ".cart.npz") is not None
        display = stem if compat else f"{stem} (legacy)"
        entry = {
            "id": stem,
            "display_name": display,
            "report_compatible": compat,
            "format": fmt,
            "location": "canonical",
        }
        if compat:
            canonical_npz.append(entry)
        else:
            legacy_pkl.append(entry)

    # -- Sandbox dir --------------------------------------------------------
    # Sandbox entries always report_compatible=true and format=npz — the
    # upload endpoint rejects anything that isn't a valid NPZ. If the file
    # goes away between enumeration and /generate open, the endpoint
    # emits ``sandbox_cart_expired`` (HTTP 410).
    sandbox_entries: list[dict[str, Any]] = []
    sandbox_dir = str(SANDBOX_DIR)
    try:
        sandbox_files = os.listdir(sandbox_dir)
    except (FileNotFoundError, OSError):
        sandbox_files = []
    for f in sandbox_files:
        if not f.endswith(".cart.npz"):
            continue
        stem = f[: -len(".cart.npz")]
        # Strip the uuid prefix for display. If somehow a sandbox file
        # has no prefix (bug / manual placement), keep the raw stem.
        human_name = _strip_sandbox_prefix(stem) or stem
        sandbox_entries.append({
            "id": stem,
            "display_name": f"{human_name} (sandbox)",
            "report_compatible": True,
            "format": "npz",
            "location": "sandbox",
        })

    # -- Order per contract -------------------------------------------------
    canonical_npz.sort(key=lambda e: e["id"].lower())
    sandbox_entries.sort(key=lambda e: e["display_name"].lower())
    legacy_pkl.sort(key=lambda e: e["id"].lower())

    return {"carts": canonical_npz + sandbox_entries + legacy_pkl}


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
    resolution = _resolve_cart_ref(req.cart_ref)
    if resolution.path is None:
        # Distinguish the three failure modes so the amber panel can render
        # accurate guidance instead of always saying "legacy format".
        failure = resolution.failure or "cart_not_found"
        if failure == "local_cart_unsupported":
            detail_msg = (
                "Reports require a server-side cart. The selected cart is a "
                "browser-only LocalCart — pick a server cart in the cart "
                "selector to generate a report."
            )
        elif failure == "cart_legacy_format":
            detail_msg = (
                f"'{req.cart_ref}' uses a legacy format that Reports can't "
                f"read yet. Rebuild it via Cart Builder → Save as .cart.npz, "
                f"then try again."
            )
        else:
            detail_msg = (
                f"Cart '{req.cart_ref}' isn't available on the server "
                f"anymore. It may have been removed or expired. Pick "
                f"another cart above to continue."
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
    cart_location = resolution.location  # 'canonical' | 'sandbox'

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
        # resolve and open. Sandbox carts hit this via the TTL cleanup
        # loop (uploads.py sweeps files older than UPLOAD_TTL_SEC every
        # UPLOAD_CLEANUP_INTERVAL_SEC); canonical carts hit it only when
        # an admin/deploy deletes the file mid-request, which is rare.
        # We report the two cases with different codes + copy so the amber
        # panel can tell users which one they're seeing.
        if cart_location == "sandbox":
            logger.warning(
                "[reports] Sandbox cart evicted mid-request for %r: %s",
                cart_path, exc,
            )
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "sandbox_cart_expired",
                    "message": (
                        "This sandbox cart was evicted before the report "
                        "could complete. Sandbox carts have a limited "
                        "lifetime — re-upload if you'd like to try again."
                    ),
                    "cart_ref": req.cart_ref,
                },
            )
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
    if cart_location:
        meta.setdefault("cart_location", cart_location)
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
