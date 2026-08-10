"""What /api/status is allowed to tell a stranger, and what eject accepts from one.

WHY THIS FILE EXISTS
====================
On 2026-08-06 the multi-seat gating shipped and `/api/status` was left unguarded, because
guarding it would have broken the app: App.tsx polls it every 2 seconds on mount, before
auth is consulted, and `engine_ready` is what boots the UI. A 401 there makes the whole
application look dead to every signed-out visitor — including the public demo.

So the endpoint has to stay public, which means the protection has to be per-FIELD. That is
a rule no route-level test can express, and the kind that quietly regresses the next time
someone adds a field to StatusResponse. Hence a test that names the two fields and says why.

The other half is `/api/cartridges/eject`. It used to take `cart_path`, which is the reason
`mounted_path` was published in the first place: the client could not ask for a delete
without being told the absolute server path. Both ends are fixed together here because
either one alone re-opens the other.
"""

import inspect

import pytest

from api import cart_access, cart_guard, main, uploads
from api.models import StatusResponse


# --------------------------------------------------------------------------
# The field that is simply gone
# --------------------------------------------------------------------------

def test_status_cannot_carry_a_filesystem_path_at_all():
    """Not "is nulled" — ABSENT from the schema, so no handler can put it back by accident.

    A stranger polling the public droplet was receiving the full server path of whatever
    cart happened to be mounted. Nulling it conditionally would leave the field available
    to the next person who finds it convenient; deleting it makes the leak unrepresentable.
    """
    assert "mounted_path" not in StatusResponse.model_fields


def test_sandboxed_flag_survives_because_it_names_nothing():
    """The Eject button still needs to know THAT the mount is an upload, not WHERE it is.

    This is the distinction the fix turns on: a boolean about the caller's own workflow is
    not a disclosure; an absolute path is.
    """
    assert "mounted_is_sandboxed" in StatusResponse.model_fields


# --------------------------------------------------------------------------
# The field that is conditional
# --------------------------------------------------------------------------

def _status_source() -> str:
    return inspect.getsource(main.get_status)


def test_status_resolves_the_callers_access_before_naming_the_cart():
    """Susie having finance open is not a fact a stranger is entitled to."""
    src = _status_source()
    assert "cart_guard.resolve(" in src, (
        "get_status must resolve the CALLER's access before reporting mounted_cartridge")
    assert "decision.allowed" in src, "get_status resolves access but ignores the answer"


def test_status_takes_the_caller_identity_it_needs_to_decide():
    """A gate that never learns who is asking is decoration.

    `user` must be OPTIONAL — the whole point is that anonymous callers still get a 200.
    """
    sig = inspect.signature(main.get_status)
    assert "request" in sig.parameters
    assert "user" in sig.parameters
    assert sig.parameters["user"].default is not inspect.Parameter.empty, (
        "user must have a Depends default; a required user would 401 the 2s poll")


@pytest.mark.parametrize("decision,expect_named", [
    # Enforcement not configured: the single-user local studio must be unchanged.
    (cart_access.decide(registered=False, owner_id=None, grant_level=None,
                        seat=None, enforced=False), True),
    # Unregistered cart: nobody has claimed it, so anyone may read — the public demo.
    (cart_access.decide(registered=False, owner_id=None, grant_level=None,
                        seat="betty"), True),
    # Granted: a seat that may read it may of course see its name.
    (cart_access.decide(registered=True, owner_id="susie", grant_level="viewer",
                        seat="betty"), True),
    # THE CASE THIS IS FOR: registered cart, no grant. Betty on redwood-finance.
    (cart_access.decide(registered=True, owner_id="susie", grant_level=None,
                        seat="betty"), False),
    # Anonymous against a registered cart — the stranger on the public droplet.
    (cart_access.decide(registered=True, owner_id="susie", grant_level=None,
                        seat=None), False),
])
def test_the_name_is_revealed_exactly_when_the_cart_is_readable(decision, expect_named):
    """One table, so the rule is legible: name it iff `allowed`.

    Guards the two directions at once. Over-hiding breaks the local studio and the public
    demo; under-hiding is the leak.

    NOTE this asserts the DECISION, not the handler. `test_status_actually_hides_the_name`
    below is the one that proves get_status honours it — the pattern that bit us six times
    in one week was a value computed correctly and then never consumed.
    """
    assert decision.allowed is expect_named


# --------------------------------------------------------------------------
# ...and the handler actually obeys it
# --------------------------------------------------------------------------

class _FakeRequest:
    headers: dict = {}


def _call_status(monkeypatch, decision, mounted="redwood-finance.cart.npz"):
    """Invoke the real handler with a stubbed access decision."""
    import asyncio

    monkeypatch.setattr(main.engine, "mounted_name", mounted, raising=False)
    monkeypatch.setattr(main.engine, "mounted_path",
                        f"/srv/carts/{mounted}" if mounted else None, raising=False)
    monkeypatch.setattr(cart_guard, "resolve", lambda request, user: decision)
    return asyncio.run(main.get_status(_FakeRequest(), None))


def test_status_actually_hides_the_name_from_an_ungranted_seat(monkeypatch):
    """Betty on redwood-finance, through the real handler. The 2026-08-05 case."""
    refused = cart_access.decide(registered=True, owner_id="susie",
                                 grant_level=None, seat="betty")
    res = _call_status(monkeypatch, refused)
    assert res.mounted_cartridge is None
    assert not hasattr(res, "mounted_path")


def test_status_still_names_the_cart_for_a_seat_that_may_read_it(monkeypatch):
    """The other direction, which is how we catch over-hiding breaking the local studio."""
    allowed = cart_access.decide(registered=True, owner_id="susie",
                                 grant_level="viewer", seat="betty")
    res = _call_status(monkeypatch, allowed)
    assert res.mounted_cartridge == "redwood-finance.cart.npz"


def test_status_survives_a_broken_access_lookup(monkeypatch):
    """The heartbeat must not stop. If the gate raises, hide the name and keep serving.

    A 500 here would make the entire UI look dead over an access-control hiccup, which is a
    worse outcome than a missing cart name.
    """
    def _boom(request, user):
        raise RuntimeError("supabase unreachable")

    monkeypatch.setattr(main.engine, "mounted_name", "x.cart.npz", raising=False)
    monkeypatch.setattr(main.engine, "mounted_path", "/srv/carts/x.cart.npz", raising=False)
    monkeypatch.setattr(cart_guard, "resolve", _boom)

    import asyncio
    res = asyncio.run(main.get_status(_FakeRequest(), None))
    assert res.mounted_cartridge is None
    assert res.engine_ready is not None      # the rest of the payload still built


# --------------------------------------------------------------------------
# Eject: the client no longer names the file
# --------------------------------------------------------------------------

def test_eject_accepts_no_path_from_the_client():
    """A sandbox check standing between a caller and unlink() is weaker than never
    accepting the path.

    The old signature was `eject_cartridge(cart_path: str)`. It was safe — the
    relative_to() check is sound — but it required publishing the path to make it usable,
    and that publication was the actual bug. Removing the parameter is what let the field go.
    """
    sig = inspect.signature(uploads.eject_cartridge)
    assert "cart_path" not in sig.parameters
    assert not sig.parameters, (
        f"eject must take nothing from the caller; got {list(sig.parameters)}")


def test_eject_still_refuses_anything_outside_the_sandbox():
    """Dropping the parameter must not drop the containment check.

    `engine.mounted_path` is server-controlled, so this is now belt-and-braces — but the
    check is the reason a mounted CATALOG cart cannot be deleted through this route, and
    that is a property worth keeping a detector on.
    """
    src = inspect.getsource(uploads.eject_cartridge)
    assert "relative_to(sandbox_resolved)" in src
    assert "403" in src


def test_eject_releases_the_file_before_deleting_it():
    """Windows will not unlink an open file, and a live engine handle to an erased cart is
    worse than either failure.

    Also the reason the client no longer calls unmount() first: doing so would leave eject
    with nothing mounted to find.
    """
    src = inspect.getsource(uploads.eject_cartridge)
    assert "engine.unmount()" in src


# --------------------------------------------------------------------------
# /health — the endpoint that exists so nobody probes /api/status instead
# --------------------------------------------------------------------------

def test_health_route_is_registered():
    paths = {getattr(r, "path", None) for r in main.app.routes}
    assert "/health" in paths, "no /health route; deploy scripts will go back to /api/status"


def test_health_says_nothing_about_the_corpus_or_the_machine():
    """Its whole purpose is to be safely public. memory_server's /status leaked
    `sessions_dir` — the Windows username and directory layout — which is why that service
    grew this exact route on 2026-07-30.
    """
    src = inspect.getsource(main.health)
    for forbidden in ("mounted", "cartridge", "passages", "path", "hostname"):
        assert forbidden not in src.split('"""')[-1], (
            f"/health response mentions {forbidden!r}; it must report liveness only")


def test_health_returns_503_while_loading_not_200():
    """The load-bearing detail, and the reason the route is worth having at all.

    `curl -sf .../health` is only a correct readiness gate if not-ready is a failure status.
    On 2026-07-30 a deploy loop polled a status endpoint for seven minutes against a healthy
    server because "still loading the model" and "refused" were indistinguishable from
    outside.
    """
    src = inspect.getsource(main.health)
    assert "status_code=503" in src
