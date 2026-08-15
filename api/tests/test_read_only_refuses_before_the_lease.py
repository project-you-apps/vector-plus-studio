"""On a read-only server, a refused write must not leave a lease behind.

THE BUG (found reviewing the 08-15 droplet deploy, before it shipped). Andy's standing rule
for the public box: anonymous visitors may use the demo but must not be able to affect
anyone else. This broke it without changing a single byte of data.

`_enforce_writable()` is called in the route BODY. Dependencies run first, so
`require_write_lease` had already claimed a 90-second lease in the caller's name -- and the
lease is deliberately RENEWED on the way out, so it outlives the failed request. The access
guard defers on unregistered carts by design ("legacy carts readable by anyone"), which is
exactly what the curated demo carts are. So:

    anonymous POST /api/cartridges/save
      -> bind ok -> guard defers -> LEASE CLAIMED -> body 403s
      -> every other visitor gets 409 "Someone is editing this cart" for 90s
      -> repeat at will

Nothing could be changed, which is why it survived review. The demo could simply be switched
off for everybody else, by anybody, with no account.

The fix is declaration order -- the same control that fixes bind-before-guard. These tests
pin the order AND the behaviour, because the order alone is a shape and shapes drift.
"""

import inspect
import re

import pytest

from api import cart_lock, main, request_cart, uploads


def _lease_routes(module):
    """Full signature of every route declaring `require_write_lease`, one string per route.

    Signatures are joined across lines first. `uploads.eject` wraps its dependencies over
    four lines, and a line-at-a-time scan silently returned NOTHING for that module -- an
    exemption test asserting on an empty list passes for entirely the wrong reason.
    """
    src = inspect.getsource(module)
    signatures, current = [], None
    for line in src.split("\n"):
        stripped = line.strip()
        if stripped.startswith("async def "):
            current = stripped
        elif current is not None:
            current += " " + stripped
        if current is not None and re.search(r"\)\s*(->[^:]*)?:$", current):
            if "require_write_lease" in current:
                signatures.append(current)
            current = None
    return signatures


# -- the ordering, on every route that takes a lease ------------------------------------

def test_every_main_lease_route_refuses_read_only_first():
    """⚠ THE REGRESSION. `_ro` must be declared BEFORE `_lock` on every one of them.

    Declared after, the lease is claimed before the refusal and the 403 leaves a 90-second
    denial behind it.
    """
    routes = _lease_routes(main)
    assert routes, "no lease-taking routes found in main -- did the pattern change?"

    broken = []
    for line in routes:
        ro = line.find("refuse_when_read_only")
        lock = line.find("require_write_lease")
        name = re.sub(r"^async def (\w+).*", r"\1", line)
        if ro < 0:
            broken.append(f"{name}: takes a write lease but never refuses read-only")
        elif ro > lock:
            broken.append(f"{name}: refuses read-only AFTER claiming the lease")

    assert not broken, (
        "a refused write on the public demo will leave a lease behind:\n  "
        + "\n  ".join(broken))


def test_sandbox_eject_is_deliberately_exempt():
    """⚠ DO NOT 'FIX' THIS. Ejecting your own sandbox upload has to keep working on the
    read-only public box -- it is the one write the demo is built around, and it is confined
    to SANDBOX_DIR by `relative_to`. Adding the refusal here would break the upload flow."""
    routes = _lease_routes(uploads)
    assert routes, "eject no longer takes a write lease -- re-check this exemption"
    for line in routes:
        assert "refuse_when_read_only" not in line, (
            "sandbox eject must stay allowed under VPS_READ_ONLY")


# -- the behaviour, not just the shape ---------------------------------------------------

@pytest.fixture
def read_only(monkeypatch):
    monkeypatch.setattr(main, "READ_ONLY_MODE", True)


def test_the_refusal_actually_raises_403(read_only):
    import asyncio

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(main.refuse_when_read_only())
    assert caught.value.status_code == 403


def test_a_writable_server_does_not_refuse(monkeypatch):
    """The refusal must be inert off the public box, or every local studio write 403s."""
    import asyncio

    monkeypatch.setattr(main, "READ_ONLY_MODE", False)
    assert asyncio.run(main.refuse_when_read_only()) is None


def test_a_refused_write_leaves_no_lease_on_the_cart(read_only):
    """THE POINT OF ALL OF IT: after the refusal, the next visitor is not told the cart is busy.

    Drives the dependency chain in declaration order by hand rather than through the HTTP
    stack, because standing the app up would boot the embedder to prove a fact about
    ordering.
    """
    import asyncio

    from fastapi import HTTPException

    lock = cart_lock.locks.for_cart("redwood-company")
    assert lock.current_lease() is None, "test started with a lease already held"

    async def _anonymous_attempts_a_write():
        await main.refuse_when_read_only()          # `_ro` runs first...
        agen = request_cart.require_write_lease(request=None, user=None)   # ...`_lock` never
        await agen.__anext__()                                             # runs at all

    with pytest.raises(HTTPException):
        asyncio.run(_anonymous_attempts_a_write())

    assert lock.current_lease() is None, (
        "a refused write left a lease behind -- every other visitor now sees "
        "'Someone is editing this cart' for 90 seconds, refreshable by anyone")
