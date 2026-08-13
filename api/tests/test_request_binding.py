"""Two callers, two carts, through the real HTTP stack.

Everything before this was mechanism provable in isolation. This is the test that says the
APP stops colliding -- a request naming Finance and a request naming Revenue, served by one
process, each seeing its own cart.

It also pins two things that would fail silently rather than loudly:

  1. `bind_caller_cart` must be `async def`. FastAPI runs SYNC dependencies in a threadpool,
     which gets a COPY of the context, so a ContextVar set there never reaches the endpoint.
     The binding would do nothing and every isolated test would still pass.
  2. The binding must be declared BEFORE the guard, because cart_guard.resolve() reads
     engine.mounted_path -- now context-backed. Declared after, access is checked against
     whatever was bound previously: guarding the wrong object.
"""

import inspect

import numpy as np

import pytest
from fastapi.testclient import TestClient

from api import cart_context, main, request_cart
from api.cart_context import CartFields
from api.cart_pool import pool


@pytest.fixture(autouse=True)
def clean_state():
    cart_context.reset_default()
    for state in list(pool):
        pool.drop(state.cart_id)
    yield
    cart_context.reset_default()
    for state in list(pool):
        pool.drop(state.cart_id)


@pytest.fixture
def fake_loader(monkeypatch):
    """Load a cart without touching disk -- this suite is about BINDING, not loading."""
    loaded = []

    def _load(cart_id: str) -> CartFields:
        loaded.append(cart_id)
        passages = [f"{cart_id} passage one", f"{cart_id} passage two"]
        # Real-SHAPED embeddings. The first attempt left these at None and search died in
        # einsum -- which was itself proof the binding had reached the endpoint, but a fake
        # that cannot be searched tests the plumbing and not the answer.
        rng = np.random.default_rng(abs(hash(cart_id)) % (2**32))
        emb = rng.standard_normal((len(passages), 768)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return CartFields(
            mounted_name=cart_id,
            mounted_path=f"/carts/{cart_id}",
            passages=passages,
            embeddings=emb,
        )

    monkeypatch.setattr(main, "load_cart_fields", _load)
    return loaded


@pytest.fixture
def client(monkeypatch):
    """No enforcement configured -- the single-user/unregistered case, so the guard defers."""
    monkeypatch.setattr(main.cart_access, "enforcement_available", lambda: False)
    return TestClient(main.app)


# -- the structural properties, asserted rather than assumed -------------------

def test_the_binding_dependency_is_async():
    """A sync dependency runs in a threadpool with a COPIED context; the bind would vanish."""
    assert inspect.isasyncgenfunction(main.bind_caller_cart), (
        "bind_caller_cart must be an async generator, or its ContextVar never reaches the "
        "endpoint and the binding silently does nothing")


def test_binding_is_declared_before_the_guard():
    """Order is the control, not a style choice."""
    params = list(inspect.signature(main.search_endpoint).parameters)
    assert "_bind" in params and "_guard" in params
    assert params.index("_bind") < params.index("_guard"), (
        "the guard would resolve access against the PREVIOUS caller's cart")


# -- behaviour through the real stack -----------------------------------------

def test_naming_no_cart_keeps_todays_behaviour(client, fake_loader):
    """The un-migrated screen: nothing bound, nothing loaded, no 500."""
    r = client.post("/api/search", json={"query": "anything", "top_k": 3})
    assert r.status_code == 200
    assert fake_loader == [], "a request naming no cart still loaded one"


def test_a_named_cart_is_loaded_and_bound(client, fake_loader):
    r = client.post("/api/search", json={"query": "passage", "top_k": 3},
                    headers={request_cart.CART_HEADER: "redwood-finance"})
    assert r.status_code == 200
    assert fake_loader == ["redwood-finance"]
    assert r.json()["query"] == "passage"


def test_two_callers_two_carts(client, fake_loader):
    """THE test. Susie names Finance, Betty names Revenue, one server, no collision."""
    susie = client.post("/api/search", json={"query": "passage", "top_k": 5},
                        headers={request_cart.CART_HEADER: "redwood-finance",
                                 request_cart.SESSION_HEADER: "tab-susie"})
    betty = client.post("/api/search", json={"query": "passage", "top_k": 5},
                        headers={request_cart.CART_HEADER: "redwood-revenue",
                                 request_cart.SESSION_HEADER: "tab-betty"})

    assert susie.status_code == 200 and betty.status_code == 200
    # `full_text`, not `passage` -- SearchResult's field names, checked rather than guessed.
    s_text = " ".join(h["full_text"] for h in susie.json()["results"])
    b_text = " ".join(h["full_text"] for h in betty.json()["results"])

    assert "redwood-finance" in s_text and "redwood-revenue" not in s_text
    assert "redwood-revenue" in b_text and "redwood-finance" not in b_text


def test_the_binding_does_not_survive_the_request(client, fake_loader):
    """A leak here would serve the last caller's cart to the next one -- the original bug."""
    client.post("/api/search", json={"query": "passage"},
                headers={request_cart.CART_HEADER: "redwood-finance"})
    assert not cart_context.is_bound()
    assert cart_context.default_fields().mounted_name is None


def test_a_cart_is_loaded_once_and_reused(client, fake_loader):
    for tab in ("tab-a", "tab-b", "tab-c"):
        client.post("/api/search", json={"query": "passage"},
                    headers={request_cart.CART_HEADER: "company",
                             request_cart.SESSION_HEADER: tab})
    assert fake_loader == ["company"], (
        f"loaded {len(fake_loader)} times for three viewers of one cart")


def test_an_unknown_cart_is_a_404_that_reveals_nothing(client, monkeypatch):
    def _missing(cart_id):
        raise FileNotFoundError(cart_id)

    monkeypatch.setattr(main, "load_cart_fields", _missing)
    r = client.post("/api/search", json={"query": "x"},
                    headers={request_cart.CART_HEADER: "someone-elses-cart"})

    assert r.status_code == 404
    body = r.text.lower()
    assert "someone-elses-cart" not in body, "the refusal echoed the cart name back"
    assert "traceback" not in body


def test_a_failed_request_does_not_leave_the_cart_pinned(client, monkeypatch, fake_loader):
    """An exception mid-request must still release the seat, or the cart becomes unevictable."""
    def _boom(*a, **kw):
        raise RuntimeError("search exploded")

    monkeypatch.setattr(main.engine, "embedder", None)
    monkeypatch.setattr(main, "_embed_query", _boom, raising=False)

    try:
        client.post("/api/search", json={"query": "x"},
                    headers={request_cart.CART_HEADER: "redwood-finance"})
    except Exception:
        pass

    state = pool.peek("redwood-finance")
    if state is not None:
        assert not state.pinned, "a failed request left the cart pinned"
