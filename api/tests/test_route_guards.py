"""Every route that touches the mounted cart must declare an access dependency.

THIS TEST IS THE POINT OF THE WHOLE EXERCISE.

On 2026-08-05 the mount endpoint was gated and twelve other cart-touching routes were not.
Betty, with no grant on redwood-finance, reached every cart and could delete and tombstone
passages. Fixing those twelve by hand would leave thirteen places to keep in agreement, and
the fourteenth route someone adds next month would silently skip the check.

So the guarantee is enforced here rather than by remembering: enumerate the app's routes,
and fail if any cart-touching one lacks `require_cart_read` or `require_cart_write`. A new
route that forgets breaks the build with a message naming it.

Adding a route that genuinely needs no guard? Put it in EXEMPT with a reason. That makes the
exemption a deliberate, reviewable act instead of an omission nobody notices.
"""

import re

import pytest

from api.main import app
from api.cart_guard import (
    require_cart_read,
    require_cart_write,
    require_named_cart_read,
)

# Every dependency that constitutes a guard. `require_named_cart_read` covers the by-name
# routes that read a cart WITHOUT mounting it -- checking the mounted cart there would guard
# the wrong object entirely, and would pass whenever the caller happened to have access to
# whatever else was open.
GUARDS = {require_cart_read, require_cart_write, require_named_cart_read}

# Paths that touch cart CONTENT or cart STATE. Matched against the route path.
CART_TOUCHING = re.compile(
    r"^/api/("
    r"search"
    r"|patterns"
    r"|cart/"
    r"|cartridges/(mount|unmount|save|lock|unlock|mounted|\{)"
    r"|membox/(mount|unmount|imprint)"
    r")"
)

# Routes that match the pattern but legitimately need no guard, each with a reason.
EXEMPT = {
    # Lists carts on disk by filename and size. Exposes no passage content, and the cart
    # list is how a user discovers what to ask for access TO.
    ("GET", "/api/cartridges"),
    # Unmount only drops the process's own handle. Refusing it could strand a caller
    # holding a cart they may no longer read -- the opposite of useful.
    ("POST", "/api/cartridges/unmount"),
    ("POST", "/api/membox/unmount"),
    # The two MOUNT routes gate the cart being OPENED, not the one already open, so they
    # cannot use require_cart_read -- that would check the wrong object and would pass
    # whenever the caller happened to have access to whatever was previously mounted. Both
    # call `_gate_mount` directly instead. Verified by test_mount_routes_gate_themselves.
    ("POST", "/api/cartridges/mount"),
    ("POST", "/api/membox/mount"),
}


def _routes():
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None) or set()
        if not path or not methods:
            continue
        deps = set()
        for d in getattr(getattr(r, "dependant", None), "dependencies", []) or []:
            if d.call in GUARDS:
                deps.add(d.call)
            for sub in getattr(d, "dependencies", []) or []:
                if sub.call in GUARDS:
                    deps.add(sub.call)
        # The dependency may also sit directly in the signature.
        for d in getattr(getattr(r, "dependant", None), "dependencies", []) or []:
            if d.call in GUARDS:
                deps.add(d.call)
        for m in methods:
            if m in ("HEAD", "OPTIONS"):
                continue
            yield m, path, deps


def test_every_cart_route_is_guarded():
    unguarded = [
        f"{m} {p}"
        for m, p, deps in _routes()
        if CART_TOUCHING.match(p) and (m, p) not in EXEMPT and not deps
    ]
    assert not unguarded, (
        "These routes touch the mounted cart but declare no access dependency:\n  "
        + "\n  ".join(sorted(unguarded))
        + "\n\nAdd Depends(require_cart_read) or Depends(require_cart_write), or add the "
          "route to EXEMPT in this file WITH A REASON."
    )


def test_destructive_routes_require_write_not_merely_read():
    """viewer and commenter both pass require_cart_read. Neither may delete a passage.

    A destructive route guarded only by read access is the bug wearing a seatbelt.
    """
    destructive = [
        (m, p, deps) for m, p, deps in _routes()
        if p.startswith("/api/patterns") and m in ("DELETE", "PUT", "POST")
    ]
    assert destructive, "expected to find the pattern-mutation routes"
    wrong = [f"{m} {p}" for m, p, deps in destructive if require_cart_write not in deps]
    assert not wrong, (
        "These routes MODIFY the cart but do not require write access:\n  "
        + "\n  ".join(sorted(wrong))
    )


def test_the_guard_list_is_not_silently_empty():
    """If the matcher stops matching anything, the suite above passes vacuously."""
    matched = [p for _, p, _ in _routes() if CART_TOUCHING.match(p)]
    assert len(matched) >= 8, f"expected many cart-touching routes, matched {len(matched)}"


def test_mount_routes_gate_themselves():
    """The two mount routes are EXEMPT from the dependency, so prove they gate another way.

    An exemption nobody verifies is just a hole with a comment on it. Both must call
    `_gate_mount`, which resolves access to the cart being OPENED rather than the one
    already open.
    """
    import inspect
    from api import main

    for fn_name in ("mount_cartridge", "membox_mount_endpoint"):
        fn = getattr(main, fn_name)
        src = inspect.getsource(fn)
        assert "_gate_mount(" in src, f"{fn_name} is EXEMPT from the guard but never calls _gate_mount"
        assert "decision.allowed" in src, f"{fn_name} calls _gate_mount but ignores the answer"


# --------------------------------------------------------------- behaviour, not declaration

class _Req:
    headers: dict = {}


def _mounted(monkeypatch, name="redwood-finance.cart.npz"):
    from api import cart_guard
    monkeypatch.setattr(cart_guard.engine, "mounted_path", name, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    cart_guard._cache.clear()


def _decision(monkeypatch, **kw):
    from api import cart_access, cart_guard
    d = cart_access.decide(**kw)
    monkeypatch.setattr(cart_guard, "resolve", lambda request, user: d)
    return d


def test_viewer_may_read_but_is_refused_a_write(monkeypatch):
    """The distinction the whole ladder turns on: read access is not write access."""
    from fastapi import HTTPException
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=True, owner_id="owner", grant_level="viewer",
              seat="someone")

    assert cart_guard.require_cart_read(_Req(), {"sub": "someone"}) is not None

    with pytest.raises(HTTPException) as e:
        cart_guard.require_cart_write(_Req(), {"sub": "someone"})
    assert e.value.status_code == 403
    assert "viewer" in str(e.value.detail)


def test_ungranted_seat_is_refused_reads_entirely(monkeypatch):
    """Betty on redwood-finance, 2026-08-05: this is the case that was wide open."""
    from fastapi import HTTPException
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=True, owner_id="susie", grant_level=None, seat="betty")

    with pytest.raises(HTTPException) as e:
        cart_guard.require_cart_read(_Req(), {"sub": "betty"})
    assert e.value.status_code == 403


def test_editor_may_write(monkeypatch):
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=True, owner_id="susie", grant_level="editor",
              seat="andy")
    assert cart_guard.require_cart_write(_Req(), {"sub": "andy"}) is not None


def test_nothing_mounted_is_not_a_refusal(monkeypatch):
    """No cart open means nothing to protect; the route reports 'no cart' in its own words."""
    from api import cart_guard
    monkeypatch.setattr(cart_guard.engine, "mounted_path", None, raising=False)
    assert cart_guard.require_cart_read(_Req(), None) is None
