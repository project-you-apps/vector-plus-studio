"""A mount must leave a cart in the pool that still HAS something in it.

WHY THIS EXISTS. `_dispatch_mount` publishes the freshly-loaded cart into the pool so the
tab's next request finds it warm. The first version then called `default_fields().clear()`
to stop the next headerless caller inheriting it -- but `active()` returns the default
OBJECT itself when nothing is bound, so publish stored a reference and clear emptied the
very cart the pool had just taken. Mount reported success, the UI showed the cart name, and
every search returned nothing (Andy, 2026-08-14).

⚠ EVERY EXISTING MOUNT TEST PASSED THROUGH THIS. They asserted on the RESPONSE -- success,
name, passage count -- all of which are computed before the publish block runs. Nothing
asked the question that matters: after the request is over, is the cart still there. That is
the shape of assertion this file adds, and the reason it is worth its own file.
"""

import asyncio

import numpy as np
import pytest

from api import cart_context, main
from api.cart_context import CartFields
from api.cart_pool import pool


@pytest.fixture(autouse=True)
def preserve_ambient_event_loop():
    """`asyncio.run()` clears the current event loop; test_edit_succession.py needs one.

    Same restorer as test_cart_context.py, and for the same reason: that suite drives its
    coroutines through the deprecated `get_event_loop().run_until_complete(...)`, which
    works only while something has left a current loop installed. Twelve of its tests
    failed the last time a new file called `asyncio.run()` without this.
    """
    try:
        previous = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous = None
    try:
        yield
    finally:
        if previous is not None and not previous.is_closed():
            asyncio.set_event_loop(previous)


@pytest.fixture(autouse=True)
def clean_state():
    def _wipe():
        cart_context.reset_default()
        for state in list(pool):
            pool.drop(state.cart_id)

    _wipe()
    yield
    _wipe()


def _loaded(name: str, n: int = 3) -> CartFields:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    emb = rng.standard_normal((n, 768)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return CartFields(mounted_name=name, mounted_path=f"/carts/{name}.pkl",
                      passages=[f"{name} passage {i}" for i in range(n)], embeddings=emb)


@pytest.fixture
def dispatch(monkeypatch):
    """Call the real _dispatch_mount with the disk-touching post-steps stubbed out.

    The permission sidecar and the generation file are not what this test is about, and
    letting them run would make it a filesystem test that fails for unrelated reasons.
    """
    monkeypatch.setattr(main, "_apply_cart_permissions_after_mount", lambda path: None)
    monkeypatch.setattr(main, "_cart_generation_module", lambda: None)

    def _run(name: str):
        def _helper():
            fields = cart_context.active()
            source = _loaded(name)
            for field_name in CartFields.__dataclass_fields__:
                setattr(fields, field_name, getattr(source, field_name))
            return main.MountResponse(success=True, name=name,
                                      pattern_count=len(fields.passages))

        return asyncio.run(main._dispatch_mount(_helper))

    return _run


def test_pooled_cart_survives_the_request_that_mounted_it(dispatch):
    """THE REGRESSION. Publish handed the pool a reference; clear emptied it in place."""
    resp = dispatch("redwood-finance")
    assert resp.success

    state = pool.peek("redwood-finance")
    assert state is not None, "mount did not publish the cart at all"
    assert len(state.payload.passages) == 3, (
        "the pooled cart is EMPTY -- mount published a reference and then cleared it, so "
        "the tab that mounted finds a cart with nothing in it and every search fails")
    assert state.payload.embeddings is not None
    assert state.payload.mounted_name == "redwood-finance"


def test_the_default_does_not_keep_the_cart_after_an_unbound_mount(dispatch):
    """The leak the clear() was there to close, still closed. Both must hold at once."""
    dispatch("redwood-revenue")

    assert cart_context.active().mounted_name is None, (
        "a headerless caller would inherit the cart the last mount loaded")
    assert cart_context.active().passages == []


def test_a_bound_mount_publishes_without_disturbing_the_default(dispatch):
    """When the tab DID name a cart, the default was never involved and must be left alone."""
    cart_context.default_fields().mounted_name = "someone-elses-cart"
    caller = CartFields()

    with cart_context.use_cart(caller):
        dispatch("redwood-engineering")

    assert len(caller.passages) == 3, "the caller's own cart lost its contents"
    assert pool.peek("redwood-engineering").payload is caller
    assert cart_context.default_fields().mounted_name == "someone-elses-cart", (
        "a bound mount detached the process default, which belongs to someone else")


def test_two_mounts_in_a_row_do_not_alias_the_same_object(dispatch):
    """Detach must install a FRESH object, not hand the same one out twice.

    If the second mount wrote into the object the first one published, the pool would hold
    two ids pointing at one cart and the earlier one would silently show the later one's
    passages -- the wrong-cart failure this whole subsystem exists to prevent.
    """
    dispatch("cart-a")
    dispatch("cart-b")

    a, b = pool.peek("cart-a").payload, pool.peek("cart-b").payload
    assert a is not b
    assert a.mounted_name == "cart-a" and b.mounted_name == "cart-b"
    assert a.passages[0].startswith("cart-a")
