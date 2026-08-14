"""FastAPI dependencies that gate every route touching the mounted cart.

WHY A DEPENDENCY AND NOT A CHECK IN EACH HANDLER
=================================================
On 2026-08-05 Andy signed in as Betty -- who has no grant on redwood-finance -- and reached
every cart, opened the Edit screen, and could delete and tombstone passages. The mount
endpoint was gated. Nothing else was.

The mistake was putting the check AT AN ENDPOINT instead of AT A CHOKE POINT. Twelve
handlers each needing to remember a check is twelve places to keep in agreement, which is
the same sibling-call-site failure that produced five other bugs that week. A dependency is
declared once per route and `test_every_cart_route_is_guarded` fails the build if a new
route forgets it.

WHY IT RE-RESOLVES PER REQUEST INSTEAD OF TRUSTING THE MOUNT DECISION
=====================================================================
`engine` is a process-level singleton: ONE mounted cart shared by every caller. So the
decision recorded at mount time belongs to whoever mounted, not to whoever is asking now.
If Susie mounts finance and Betty then calls DELETE /api/patterns/3, reusing the mount-time
decision would hand Betty Susie's access.

That is not hypothetical -- it is the normal case the moment two browsers point at one
backend, which is exactly how Andy tests.

So each request resolves the CALLER's access to the CURRENTLY mounted cart. Correctness over
cleverness; the cache below is what keeps it affordable.

THE CACHE, AND ITS HONEST COST
===============================
Resolving means an RPC to Postgres. Doing that on every search would add a network
round-trip to every keystroke-adjacent operation, so results are cached per
(seat, cart) for `CACHE_TTL_SECONDS`.

**That means a revoked grant keeps working for up to that long.** A deliberate trade, stated
rather than hidden: grants change rarely, reads happen constantly. If revocation ever needs
to be immediate -- a firing, a breach -- this TTL is the knob, and `invalidate()` exists so
the grant endpoints can clear it on write.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request

from . import cart_access
from . import cartridge_io
from . import object_access
from .auth import get_current_user
from .engine import engine

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30.0

# (seat_or_None, cart_basename) -> (expires_at, MountDecision)
_cache: dict[tuple[Optional[str], str], tuple[float, cart_access.MountDecision]] = {}

# Same key and TTL as `_cache`, separate dict so `invalidate()` can clear either without
# the other -- a grant change and an exception change are different events.
_object_cache: dict = {}


def invalidate(seat: Optional[str] = None) -> None:
    """Drop cached decisions. Call after any grant change.

    With no argument, clears everything -- correct when a cart's grants change, since that
    affects other seats too.
    """
    if seat is None:
        _cache.clear()
        _object_cache.clear()
        return
    for key in [k for k in _cache if k[0] == seat]:
        _cache.pop(key, None)
    for key in [k for k in _object_cache if k[0] == seat]:
        _object_cache.pop(key, None)


def _seat_from_token(user: object) -> Optional[str]:
    """Identity for access decisions comes from the VERIFIED TOKEN, never elsewhere.

    Deliberately not `_seat_for()`, whose VPS_SEAT environment fallback exists for attention
    attribution. On 2026-08-05 that fallback made an anonymous caller look like a seat and
    turned a "please sign in" into "ask the owner for access". An env var must not shape an
    access decision even when it cannot grant one.
    """
    if isinstance(user, dict):
        sub = user.get("sub")
        if isinstance(sub, str) and sub.strip():
            return sub.strip()
    return None


def resolve(request: Request, user: object) -> Optional[cart_access.MountDecision]:
    """The caller's access to the currently mounted cart, or None if nothing is mounted."""
    mounted = getattr(engine, "mounted_path", None)
    if not mounted:
        return None

    if not cart_access.enforcement_available():
        return cart_access.decide(registered=False, owner_id=None, grant_level=None,
                                  seat=None, enforced=False)

    seat = _seat_from_token(user)
    cart = os.path.basename(str(mounted))
    key = (seat, cart)
    now = time.monotonic()

    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]

    try:
        from . import profile_routes
        client = profile_routes._supabase(profile_routes._token(request))
        decision = cart_access.lookup(client, cart, seat)
    except Exception as e:                                  # noqa: BLE001
        # Type as well as message: during the 08-04 migration a lookup failure was almost
        # always a configuration fact, not a bad request, and the class name is what tells
        # those apart at a glance.
        log.warning("cart access lookup failed: %s: %s", type(e).__name__, e)
        return cart_access.lookup_failed()

    _cache[key] = (now + CACHE_TTL_SECONDS, decision)
    return decision


def object_policy(request: Request, user: object):
    """This caller's DOCUMENT-level policy for the mounted cart. See db/006.

    Cached on the same (seat, cart) key and TTL as the mount decision, and for the same
    reason: a search returns passages from many documents and the policy is consulted once
    per result, so an uncached lookup would put a network round-trip inside the result loop.

    Returns an ObjectPolicy whose `available` is False when the lookup did not complete.
    Callers MUST refuse in that case rather than pick a default — see
    `object_access.policy_lookup_failed` for why both defaults are wrong.

    An unconfigured deployment gets an empty, permissive policy: with no auth there are no
    seats, so there is nothing document-level to enforce and every cart behaves as it always
    has. Same reasoning as `cart_access.enforcement_available`.
    """
    mounted = getattr(engine, "mounted_path", None)
    if not mounted or not cart_access.enforcement_available():
        return object_access.ObjectPolicy()

    seat = _seat_from_token(user)
    cart = os.path.basename(str(mounted))
    key = (seat, cart)
    now = time.monotonic()

    hit = _object_cache.get(key)
    if hit and hit[0] > now:
        return hit[1]

    try:
        from . import profile_routes
        client = profile_routes._supabase(profile_routes._token(request))
        policy = object_access.lookup(client, cart)
    except Exception as e:                                  # noqa: BLE001
        log.warning("object policy lookup failed: %s: %s", type(e).__name__, e)
        return object_access.policy_lookup_failed()

    _object_cache[key] = (now + CACHE_TTL_SECONDS, policy)
    return policy


def may_read_document(policy, decision, hippo_entry) -> bool:
    """Whether this caller may see one passage, given its document.

    The single place the search path and the agent path both ask, so the rule cannot drift
    between them — which is exactly what happened with PERM_R, where the bit was honoured
    for agents and ignored for people for weeks.

    PERM_R is deliberately NOT re-checked here. `pattern_permits_read` is the one authority
    for that bit and both paths already call it; doing it twice would mean two places to
    keep in agreement, which is the failure this function exists to avoid.
    """
    key = object_access.document_key(hippo_entry)
    if key is None:
        # No provenance in this cart -- most of ours predate it. Nothing document-level can
        # apply, so this is not a denial.
        return True

    level = getattr(decision, "level", None)
    is_owner = level == "owner"
    result = object_access.resolve(
        cart_level=level,
        inherit=policy.inherit,
        exception=policy.exception_for(key),
        is_owner=is_owner,
        perms_byte=None,
    )
    return result.may_read


def _refuse(decision: cart_access.MountDecision) -> HTTPException:
    detail = {
        cart_access.DECISION_NO_GRANT:
            "You do not have access to this cart. Ask its owner to grant you a level.",
        cart_access.DECISION_ANONYMOUS:
            "Sign in to use this cart.",
        cart_access.DECISION_LOOKUP_FAILED:
            "Cart access could not be verified right now, so the request was refused rather "
            "than allowed unchecked. This is a service problem, not a permissions one.",
    }.get(decision.reason, "Cart access denied.")
    code = 503 if decision.reason == cart_access.DECISION_LOOKUP_FAILED else 403
    return HTTPException(status_code=code, detail=detail)


def resolve_named(request: Request, user: object,
                  cart_name: str) -> cart_access.MountDecision:
    """The caller's access to a cart named in the PATH, which may not be the mounted one.

    `/api/cartridges/{cart_name}/embeddings` and friends read a cart by name without
    mounting it, so checking the mounted cart would guard the wrong object entirely -- and
    would pass whenever the caller happened to have access to something else.
    """
    if not cart_access.enforcement_available():
        return cart_access.decide(registered=False, owner_id=None, grant_level=None,
                                  seat=None, enforced=False)

    seat = _seat_from_token(user)
    # Callers pass a bare cart name; user_carts stores the on-disk filename. Shared with
    # main.load_cart_fields so the name that is CHECKED is the name that is LOADED.
    candidates = cartridge_io.name_candidates(cart_name)

    last = None
    for cand in candidates:
        key = (seat, cand)
        now = time.monotonic()
        hit = _cache.get(key)
        if hit and hit[0] > now:
            last = hit[1]
        else:
            try:
                from . import profile_routes
                client = profile_routes._supabase(profile_routes._token(request))
                last = cart_access.lookup(client, cand, seat)
            except Exception as e:                          # noqa: BLE001
                log.warning("cart access lookup failed: %s: %s", type(e).__name__, e)
                return cart_access.lookup_failed()
            _cache[key] = (now + CACHE_TTL_SECONDS, last)
        # A registered match is authoritative; keep looking only while "unregistered".
        if last.reason != cart_access.DECISION_UNREGISTERED:
            return last
    return last


def enforce_named_read(request: Request, user: object, cart_name: str):
    """Refuse unless this caller may read `cart_name`. CALL THIS; do not Depends() it.

    For routes that name their cart in the REQUEST BODY rather than the path --
    `/api/agents/run`, `/api/agents/save_to_cart`, `/api/reports/generate` all take a
    `cart_ref`. FastAPI resolves a `Depends` before the body is parsed, so
    `require_named_cart_read` cannot see that name and cannot guard those routes. Left
    unguarded on that technicality, all three answered strangers: measured 2026-08-12, they
    reached body validation with no caller identity at all.

    Same shape as `_gate_mount` in main.py, for the same reason, and like it these routes are
    EXEMPT from the dependency check in test_route_guards.py and asserted to call this
    instead. An exemption nobody verifies is just a hole with a comment on it.

    CALL IT AFTER RESOLVING cart_ref TO A REAL CART, not before -- guarding the string the
    caller sent rather than the cart it resolves to would check the wrong object, which is
    the bug `resolve_named` exists to avoid.
    """
    decision = resolve_named(request, user, cart_name)
    if not decision.allowed:
        raise _refuse(decision)
    return decision


def require_named_cart_read(request: Request, cart_name: str,
                            user: dict | None = Depends(get_current_user)):
    """Caller may read the cart named in the path. For by-name routes that never mount."""
    return enforce_named_read(request, user, cart_name)


def require_cart_read(request: Request,
                      user: dict | None = Depends(get_current_user)):
    """Caller may read the mounted cart. Declare on every route that returns its content."""
    decision = resolve(request, user)
    if decision is None:
        return None                      # nothing mounted; the route reports that itself
    if not decision.allowed:
        raise _refuse(decision)
    return decision


def require_cart_write(request: Request,
                       user: dict | None = Depends(get_current_user)):
    """Caller may modify the mounted cart. Declare on every destructive or additive route.

    Read access is necessary but nowhere near sufficient: viewer and commenter both pass
    `require_cart_read` and neither may write. `commenter` writes to the proposal sidecar,
    never to the cart, which is the distinction the whole access ladder turns on.
    """
    decision = resolve(request, user)
    if decision is None:
        return None
    if not decision.allowed:
        raise _refuse(decision)

    # NO GRANT GOVERNS THIS CART -> DEFER, do not refuse.
    #
    # `level is None` means either a legacy cart with no user_carts row, or enforcement not
    # configured at all. In both cases Andy's rule (2026-08-03) is that writability belongs
    # to the existing read-only flag, not to us: "if they are editable then anyone can write
    # them and if they are read-only then no one can write them."
    #
    # Refusing here instead would freeze every unregistered cart and the whole single-user
    # local studio -- which is precisely the failure `test_unregistered_cart_claims_no_write
    # _grant` was written to warn about, and which this function did on its first draft.
    # `may_write` answers "did a grant authorise this", never "is the cart writable".
    if decision.level is None:
        return decision

    if not decision.may_write:
        raise HTTPException(
            status_code=403,
            detail=(f"Your access to this cart is '{decision.level}', which cannot modify it. "
                    f"Ask its owner for editor access."))
    return decision
