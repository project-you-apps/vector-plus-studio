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
    # Presence and the name cache are module-level like the pool. Forgetting them leaked one
    # test's viewers into the next and failed two occupancy tests for the wrong reason.
    def _wipe():
        cart_context.reset_default()
        request_cart._presence.clear()
        request_cart._display_names.clear()
        for state in list(pool):
            pool.drop(state.cart_id)

    _wipe()
    yield
    _wipe()


@pytest.fixture
def swap_loader():
    """Replace the registered loader, and put the real one back.

    ⚠ NOT `monkeypatch.setattr(main, "load_cart_fields", ...)`. `request_cart.set_loader()`
    captures the function ONCE at import, so patching the module attribute afterwards changes
    nothing the dependency ever reads -- three tests failed exactly that way when the registry
    landed. Anything that needs a different loader has to go through set_loader.
    """
    original = request_cart._loader

    def _swap(fn):
        request_cart.set_loader(fn)

    yield _swap
    request_cart.set_loader(original)


@pytest.fixture
def fake_loader(swap_loader):
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

    swap_loader(_load)
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


def test_every_guarded_route_binds_first():
    """EVERY route, not just search -- route eighteen is the one that bites.

    Same discipline as test_every_route_is_guarded_or_explicitly_exempt: checking the routes
    someone remembered is how `/api/agents`, `/api/reports` and `/api/llm` sat unguarded
    while a green test claimed otherwise.

    A guarded route that does NOT bind serves whatever cart the previous request left behind.
    A guarded route that binds AFTER the guard has its access checked against that same stale
    cart. Both are silent; both are the bug this phase exists to remove.
    """
    from api.cart_guard import require_cart_read, require_cart_write

    guards = {require_cart_read, require_cart_write}
    offenders = []

    for route in main.app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        deps = [d.call for d in getattr(dependant, "dependencies", []) or []]
        if not (guards & set(deps)):
            continue

        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        params = list(inspect.signature(endpoint).parameters)
        path = getattr(route, "path", "?")

        if "_bind" not in params:
            offenders.append(f"{path} -- guarded but never binds the caller's cart")
        elif "_guard" in params and params.index("_bind") > params.index("_guard"):
            offenders.append(f"{path} -- binds AFTER the guard; access checked on the "
                             f"previous caller's cart")

    assert not offenders, "\n  " + "\n  ".join(sorted(offenders))


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


def test_an_unknown_cart_is_a_404_that_reveals_nothing(client, swap_loader):
    def _missing(cart_id):
        raise FileNotFoundError(cart_id)

    swap_loader(_missing)
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


# -- occupancy: who else is in this cart --------------------------------------

def test_occupancy_names_the_other_viewer_not_their_key(client, fake_loader):
    """A seat uuid on screen is noise to a person and a disclosure to everyone else."""
    # Presence is recorded per TAB with the name attached, so seed it the way a request does.
    request_cart.touch_presence("company", "tab:betty-tab", "Betty Alvarez")

    r = client.get("/api/status", headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "susie-tab"})

    occupants = r.json()["cart_occupants"]
    assert occupants == ["Betty Alvarez"]
    assert not any("tab:" in o or "seat:" in o for o in occupants)


def test_the_same_account_in_two_tabs_sees_itself(client, fake_loader):
    """One account, two browsers = two TABS, so they are visible to each other.

    Deliberate: it is the multi-device signal Andy asked about, and it arrives free from
    keying presence by tab. Keyed by identity, both would have excluded each other as "self"
    and a person signed in twice would see nobody.
    """
    request_cart.touch_presence("company", "tab:susie-edge", "Susie Nakamura")

    r = client.get("/api/status", headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "susie-chrome"})
    assert r.json()["cart_occupants"] == ["Susie Nakamura"]


def test_changing_identity_replaces_a_tab_rather_than_haunting_the_cart(client, fake_loader):
    """Log out and the departed identity must not linger for the whole window.

    Found 2026-08-13: presence was keyed by view_key, which CHANGES on logout
    (seat:<uuid> -> anon:<tab>), so Betty stayed "here" after Andy signed out of her.
    """
    request_cart.touch_presence("company", "tab:andys-tab", "Betty Alvarez")
    request_cart.touch_presence("company", "tab:andys-tab", None)      # signed out

    r = client.get("/api/status", headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "other-tab"})
    occupants = r.json()["cart_occupants"]
    assert occupants == ["a guest"], f"Betty haunted the cart: {occupants}"


def test_you_are_not_your_own_occupant(client, fake_loader):
    r = client.get("/api/status", headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "tab-susie"})
    assert r.json()["cart_occupants"] == []


def test_an_unnamed_viewer_reads_as_a_guest(client, fake_loader):
    """Anonymous must not leak a tab id, and must not sound ominous.

    Andy on the first cut: "'Someone is also here' seems so ominous. lol"
    """
    client.post("/api/search", json={"query": "p"}, headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "ghost-tab"})

    r = client.get("/api/status", headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "susie-tab"})
    assert r.json()["cart_occupants"] == ["a guest"]


def test_occupancy_is_empty_for_a_different_cart(client, fake_loader):
    client.post("/api/search", json={"query": "p"}, headers={
        request_cart.CART_HEADER: "company", request_cart.SESSION_HEADER: "tab-betty"})
    r = client.get("/api/status", headers={
        request_cart.CART_HEADER: "revenue", request_cart.SESSION_HEADER: "tab-susie"})
    assert r.json()["cart_occupants"] == []


# -- mounting must not leak into the process default --------------------------

def test_a_first_mount_goes_to_the_pool_not_the_process_default(client, monkeypatch):
    """Andy, 2026-08-14: mounted a cart in one browser; a second user signing in fresh
    ARRIVED already holding it. A third user with a cart of her own was unaffected.

    A tab's FIRST mount carries no X-VPS-Cart -- the browser has nothing to name yet -- so
    the mount helpers wrote into the process-wide default, and the pool never saw the cart.
    Every headerless caller then inherited it. That is the single-global bug this whole
    feature removes, surviving in the one route that creates mounts.
    """
    import api.main as m

    def fake_dispatch_target():
        f = cart_context.active()
        f.mounted_name = "wiki_nomic_100k"
        f.mounted_path = "/carts/wiki_nomic_100k.pkl"
        f.passages = ["w0"]
        return m.MountResponse(success=True, message="ok", name="wiki_nomic_100k",
                               pattern_count=1)

    monkeypatch.setattr(m, "_mount_plan", lambda fn: (lambda *a: fake_dispatch_target(), ()))
    monkeypatch.setattr(m, "_refuse_path_shaped_filename", lambda fn: None)

    r = client.post("/api/cartridges/mount", json={"filename": "/carts/wiki_nomic_100k.pkl"},
                    headers={request_cart.SESSION_HEADER: "andys-firefox"})
    assert r.status_code == 200, r.text

    assert cart_context.default_fields().mounted_name is None, (
        "the mount landed in the process default; every headerless caller inherits it")
    assert "wiki_nomic_100k" in pool, "the mounted cart never reached the pool"

    # A different seat, no cart named, must see nothing.
    s = client.get("/api/status", headers={request_cart.SESSION_HEADER: "susies-chrome"})
    assert s.json()["mounted_cartridge"] is None, (
        "a fresh sign-in inherited someone else's mount")
