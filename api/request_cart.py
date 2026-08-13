"""Which cart is this REQUEST for, and which view does it belong to.

Method B (decided 2026-08-13): the caller names its cart per request rather than the server
remembering one cart per seat. Andy asked what the difference is for users, and it is this:
a server-side binding is keyed by SEAT, and two browser tabs are the same seat -- so opening
Finance in one tab and Revenue in another would make the second silently switch the first.
That is the Susie/Betty bug inside one person's browser. Avoiding it needs a per-tab
identifier on each request, which is this.

It also buys a cart id in the URL (bookmarkable, shareable), survival across a server
restart, and removes the single-worker constraint that in-process binding would impose.

⚠ TWO KEYS, AND CONFLATING THEM WOULD BE A SECURITY BUG
================================================================================
`access_seat`  -- from the VERIFIED TOKEN only. None when anonymous. The ONLY thing an
                  access decision may look at. `cart_guard` derives its own; this module
                  never widens it.
`view_key`     -- who is LOOKING. The signed-in seat when there is one, otherwise the tab's
                  own id. Used for pool holders and idle accounting. **Grants nothing.**

On 2026-08-05 a `VPS_SEAT` env fallback made an anonymous caller look like a seat and turned
"please sign in" into "ask the owner for access." The lesson generalises: an identifier that
exists for bookkeeping must never reach an authorization path. A caller can forge
`X-VPS-Session` freely and gain exactly nothing -- it selects which cart they are looking at,
and every cart-touching route still resolves access from the token.

Design: docs/DESIGN-multi-mount-and-write-path.md §1b, §3
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from fastapi import Request

from . import cart_context
from .cart_pool import CartState, pool

__all__ = ["CART_HEADER", "SESSION_HEADER", "requested_cart", "view_key", "access_seat",
           "bound_cart"]

# The caller names its cart here, or as ?cart= for links we want to be pasteable.
CART_HEADER = "x-vps-cart"

# A per-TAB id the browser generates and keeps in sessionStorage. Per tab rather than per
# browser deliberately: two tabs wanting two carts is the workflow this exists to allow, and
# localStorage would make them share again.
SESSION_HEADER = "x-vps-session"

# A forged or absent session id must never widen access, so it is length-capped and
# character-restricted only to keep junk out of logs and cache keys -- not as a security
# control, because it is not one.
_MAX_SESSION_KEY = 64


def access_seat(user: object) -> Optional[str]:
    """The seat an ACCESS decision may use, or None. Verified token only.

    Deliberately duplicates `cart_guard._seat_from_token` rather than importing it, so that
    nothing in this module can be mistaken for a place where access identity is decided. If
    they ever disagree, cart_guard wins -- it is the authority.
    """
    if isinstance(user, dict):
        sub = user.get("sub")
        if isinstance(sub, str) and sub.strip():
            return sub.strip()
    return None


def view_key(request: Request, user: object) -> str:
    """Who is LOOKING at a cart. Never an access decision.

    Signed-in callers key on their seat, so their tabs share a view only if they share a tab
    id -- which they do not. Anonymous callers key on the tab id they generated. A caller
    that sends nothing gets a per-connection fallback, which is the honest answer for a
    client that has not adopted the header yet: it may be shared with other such callers, and
    sharing a VIEW is a UX wrinkle rather than a disclosure, because access is resolved
    independently on every request.
    """
    seat = access_seat(user)
    if seat:
        return f"seat:{seat}"

    raw = (request.headers.get(SESSION_HEADER) or "").strip()
    if raw:
        cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")[:_MAX_SESSION_KEY]
        if cleaned:
            return f"anon:{cleaned}"

    client = getattr(request, "client", None)
    host = getattr(client, "host", None) or "unknown"
    return f"anon-noheader:{host}"


def requested_cart(request: Request) -> Optional[str]:
    """The cart this request names, or None to mean "whatever the server has".

    None is not an error. Step 3 accepts BOTH shapes so the frontend can migrate screen by
    screen instead of on a flag day -- an un-migrated screen sends nothing and keeps today's
    behaviour, a migrated one names its cart and gets its own view.

    Header first, then `?cart=`, because the query form exists for links a person can paste
    and a header cannot travel in a URL.

    RETURNS THE NAME AS GIVEN. Resolving it to a file, and deciding whether this caller may
    read it, belong to the mount path and cart_guard respectively. A name arriving from a
    caller is a request, not a fact -- and `_refuse_path_shaped_filename` exists because we
    learned that the hard way on 2026-08-12.
    """
    raw = (request.headers.get(CART_HEADER) or "").strip()
    if not raw:
        raw = (request.query_params.get("cart") or "").strip()
    if not raw:
        return None

    # Basename only. A cart id is a NAME here; anything path-shaped is either a mistake or an
    # attempt, and neither should reach the loader with its separators intact.
    return os.path.basename(raw)[:256] or None


@contextmanager
def bound_cart(cart_id: Optional[str], viewer: str,
               loader: Callable[[str], object]) -> Iterator[Optional[CartState]]:
    """Bind `cart_id` for this request, loading it if nobody has it open.

    NAMING NO CART IS THE UN-MIGRATED PATH, not an error: yields None and binds nothing, so
    `engine.*` resolves to the process-wide cart exactly as it did before Method B. That is
    what lets the frontend migrate screen by screen.

    ⚠ ACCESS IS NOT CHECKED HERE, DELIBERATELY. This binds; `cart_guard` decides. Putting the
    check inside would create a second authority on the same question, which is the drift
    `_write_blocked` was written to end -- and it would be the WEAKER one, because a caller
    can name any cart they like. Callers must resolve access BEFORE binding.

    The seat is released on the way out but the cart STAYS POOLED, warm for the next request
    and for anyone else on it. Whether it leaves memory is eviction's business, not a user's.
    """
    if not cart_id:
        yield None
        return

    state = pool.acquire(cart_id, lambda: loader(cart_id), seat=viewer)
    try:
        fields = state.payload
        if not isinstance(fields, cart_context.CartFields):
            raise TypeError(
                f"pool loader for {cart_id!r} returned {type(fields).__name__}, "
                f"expected CartFields -- engine.* would silently read the wrong shape")
        with cart_context.use_cart(fields):
            yield state
    finally:
        # Release even if the request raised. A seat that keeps a cart pinned because its
        # request 500'd would make the cart unevictable forever, and the pool refuses to
        # evict pinned carts by design -- so a leak here becomes PoolFull later, far from
        # the cause.
        pool.release(cart_id, viewer)
