"""One writer at a time, through the real HTTP stack, with a name on the refusal.

The pool made concurrent writes reachable: two seats share one CartFields, so two requests
can mutate one cart's passages at once. `cart_lock` stops that; this proves it reaches the
routes, and that the refusal says WHO rather than spinning.

Andy, 2026-08-12: *"'Betty is editing this cart' is better."*
"""

import inspect

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import cart_context, cart_lock, main, request_cart
from api.cart_context import CartFields
from api.cart_pool import pool


@pytest.fixture(autouse=True)
def clean_state():
    cart_context.reset_default()
    cart_lock.locks.clear()
    for state in list(pool):
        pool.drop(state.cart_id)
    yield
    cart_context.reset_default()
    cart_lock.locks.clear()
    for state in list(pool):
        pool.drop(state.cart_id)


@pytest.fixture
def writable_cart():
    """A loader whose carts are UNLOCKED, so the write path is reached rather than refused.

    CartFields defaults `read_only=True`; a write against the default would come back
    "Cartridge is read-only" and never touch the lease at all -- which would make this whole
    suite pass while testing nothing.
    """
    original = request_cart._loader

    def _load(cart_id: str) -> CartFields:
        rng = np.random.default_rng(7)
        passages = [f"{cart_id} p0", f"{cart_id} p1", f"{cart_id} p2"]
        emb = rng.standard_normal((len(passages), 768)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return CartFields(mounted_name=cart_id, mounted_path=f"/carts/{cart_id}",
                          passages=passages, embeddings=emb, read_only=False)

    request_cart.set_loader(_load)
    yield
    request_cart.set_loader(original)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.cart_access, "enforcement_available", lambda: False)
    monkeypatch.setattr(main, "READ_ONLY_MODE", False, raising=False)
    return TestClient(main.app)


def _delete(client, tab: str, cart: str, idx: int = 0):
    return client.delete(f"/api/patterns/{idx}", headers={
        request_cart.CART_HEADER: cart,
        request_cart.SESSION_HEADER: tab,
    })


# -- structural: the lease must reach every write route, AFTER the guard -------

def test_every_write_route_takes_the_lease_after_the_guard():
    """Enumerated, not spot-checked -- the same discipline that found the eject straggler.

    AFTER the guard, deliberately. A caller who may not write must be refused for THAT
    reason, not told the cart is busy -- and taking a claim on a cart you cannot write would
    let a viewer block an editor by clicking around.
    """
    from api.cart_guard import require_cart_write

    offenders = []
    for route in main.app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        if require_cart_write not in [d.call for d in getattr(dependant, "dependencies", []) or []]:
            continue

        params = list(inspect.signature(route.endpoint).parameters)
        path = getattr(route, "path", "?")
        if "_lock" not in params:
            offenders.append(f"{path} -- writes without taking the write lease")
        elif "_guard" in params and params.index("_lock") < params.index("_guard"):
            offenders.append(f"{path} -- takes the lease BEFORE the access guard")

    assert not offenders, "\n  " + "\n  ".join(sorted(offenders))


# -- behaviour ----------------------------------------------------------------

def test_one_writer_succeeds(client, writable_cart):
    r = _delete(client, "tab-betty", "redwood-company")
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True


def test_a_second_writer_is_refused_with_409(client, writable_cart):
    """Betty writes; Susie's write is refused while Betty's claim is live."""
    assert _delete(client, "tab-betty", "redwood-company", 0).status_code == 200

    r = _delete(client, "tab-susie", "redwood-company", 1)
    assert r.status_code == 409, r.text
    body = r.json()["detail"]
    assert body["error"] == "cart_busy"
    assert "editing this cart" in body["message"]
    assert body["seconds_left"] > 0


def test_the_refusal_does_not_leak_a_seat_id(client, writable_cart):
    """A uuid in an error message is noise to a person and a disclosure to everyone else."""
    _delete(client, "tab-betty", "redwood-company", 0)
    r = _delete(client, "tab-susie", "redwood-company", 1)
    assert "tab-betty" not in r.text
    assert "anon:" not in r.text


def test_the_same_writer_may_keep_writing(client, writable_cart):
    """Re-entrant for the holder, or a person's second edit would refuse their own first."""
    assert _delete(client, "tab-betty", "redwood-company", 0).status_code == 200
    assert _delete(client, "tab-betty", "redwood-company", 1).status_code == 200


def test_a_different_cart_is_unaffected(client, writable_cart):
    """The claim is per-cart. Betty editing company must not block Susie on revenue."""
    assert _delete(client, "tab-betty", "redwood-company", 0).status_code == 200
    assert _delete(client, "tab-susie", "redwood-revenue", 0).status_code == 200


def test_an_expired_claim_frees_the_cart(client, writable_cart, monkeypatch):
    """Betty closes her laptop. Susie must not be locked out forever."""
    assert _delete(client, "tab-betty", "redwood-company", 0).status_code == 200

    lock = cart_lock.locks.for_cart("redwood-company")
    monkeypatch.setattr(lock, "_clock", lambda: lock._lease.expires_at + 1.0)

    assert _delete(client, "tab-susie", "redwood-company", 1).status_code == 200


def test_readers_are_never_blocked_by_a_writer(client, writable_cart):
    """Reads must not queue behind an edit -- the whole point of a per-cart WRITE lock."""
    assert _delete(client, "tab-betty", "redwood-company", 0).status_code == 200

    r = client.post("/api/search", json={"query": "p1", "top_k": 2}, headers={
        request_cart.CART_HEADER: "redwood-company",
        request_cart.SESSION_HEADER: "tab-susie",
    })
    assert r.status_code == 200, "a reader was blocked by someone else's write claim"
