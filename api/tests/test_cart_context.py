"""Two requests, two carts, no leakage — and the default must stay exactly as it was.

The bug being fixed: `engine` is a process singleton, so Betty mounting Revenue dropped
Susie's Finance and Susie's next search answered from Revenue. If she held a grant on
Revenue she got Revenue's passages under Finance's label, which is worse than a refusal
because nothing looks broken.
"""

import asyncio
import inspect
import re

import pytest

from api import cart_context
from api.cart_context import CartFields, active, default_fields, is_bound, use_cart


@pytest.fixture(autouse=True)
def clean_default():
    """The process default is global; a test that dirties it must not leak into the next."""
    cart_context.reset_default()
    yield
    cart_context.reset_default()


@pytest.fixture(autouse=True)
def preserve_ambient_event_loop():
    """`asyncio.run()` clears the current event loop. Other suites still rely on one.

    test_edit_succession.py drives coroutines through the deprecated
    `asyncio.get_event_loop().run_until_complete(...)`, which works only while SOMETHING has
    left a current loop installed. Nothing in the suite used to set or clear one, so it
    quietly worked; the moment this file called `asyncio.run()` those twelve tests failed
    with "no current event loop" -- passing alone, failing together, which is the pollution
    shape we spent 2026-08-10 on.

    Restoring the ambient loop here fixes it at the source (this file introduced the change)
    rather than editing a suite that is not ours to churn. Their pattern is still fragile:
    the next `asyncio.run` anywhere in the test tree will break it again, so it wants
    migrating to `asyncio.run` / pytest-asyncio on its own terms.
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


# -- the default keeps single-user behaviour ---------------------------------

def test_nothing_bound_yields_the_process_default():
    assert not is_bound()
    assert active() is default_fields()


def test_active_never_returns_none():
    """`engine.passages` must not raise merely because a caller forgot to bind."""
    assert active() is not None


def test_the_default_is_writable_and_shared():
    """Startup, CLI and every existing test write here. That IS today's behaviour."""
    active().mounted_name = "gutenberg-poetry"
    assert default_fields().mounted_name == "gutenberg-poetry"


# -- binding ------------------------------------------------------------------

def test_use_cart_binds_and_restores():
    finance = CartFields(mounted_name="redwood-finance")
    assert not is_bound()
    with use_cart(finance):
        assert is_bound()
        assert active().mounted_name == "redwood-finance"
    assert not is_bound()
    assert active() is default_fields()


def test_binding_does_not_touch_the_default():
    with use_cart(CartFields(mounted_name="redwood-finance")):
        active().passages = ["secret"]
    assert default_fields().passages == [], "a bound cart leaked into the process default"


def test_nested_bind_restores_the_parent_not_the_default():
    outer = CartFields(mounted_name="outer")
    inner = CartFields(mounted_name="inner")
    with use_cart(outer):
        with use_cart(inner):
            assert active().mounted_name == "inner"
        assert active().mounted_name == "outer", "nested bind reset to the default"


# -- THE test -----------------------------------------------------------------

def test_two_concurrent_tasks_see_two_different_carts():
    """Susie and Betty, interleaved. The regression test for the whole exercise."""
    seen: dict[str, list[str]] = {"susie": [], "betty": []}

    async def seat(name: str, cart: str):
        with use_cart(CartFields(mounted_name=cart, passages=[f"{cart}-p0"])):
            for _ in range(5):
                # Yield control so the two tasks genuinely interleave; with a plain global
                # this is exactly where one would clobber the other.
                await asyncio.sleep(0)
                seen[name].append(active().mounted_name)

    async def both():
        await asyncio.gather(seat("susie", "redwood-finance"),
                             seat("betty", "redwood-revenue"))

    asyncio.run(both())

    assert set(seen["susie"]) == {"redwood-finance"}, seen["susie"]
    assert set(seen["betty"]) == {"redwood-revenue"}, seen["betty"]


def test_the_binding_survives_asyncio_to_thread():
    """main.py runs the sync numpy work in threads; the cart must travel with it.

    `asyncio.to_thread` copies the current context, so this holds -- but it holds by
    LIBRARY GUARANTEE rather than by anything we wrote, which is exactly the kind of
    assumption worth pinning with a test.
    """
    def read_it() -> str:
        return active().mounted_name

    async def go() -> str:
        with use_cart(CartFields(mounted_name="redwood-finance")):
            return await asyncio.to_thread(read_it)

    assert asyncio.run(go()) == "redwood-finance"


def test_a_thread_without_a_binding_gets_the_default():
    def read_it():
        return active() is default_fields()

    async def go():
        return await asyncio.to_thread(read_it)

    assert asyncio.run(go()) is True


# -- clear --------------------------------------------------------------------

def test_clear_resets_every_field():
    f = CartFields(mounted_name="x", passages=["a"], deleted_ids={1},
                   dirty=True, signatures_loaded=True)
    f.clear()
    assert f.mounted_name is None
    assert f.passages == []
    assert f.deleted_ids == set()
    assert f.dirty is False
    assert f.signatures_loaded is False


def test_clear_closes_a_sqlite_sidecar():
    closed = []

    class FakeConn:
        def close(self):
            closed.append(True)

    f = CartFields(sqlite_conn=FakeConn(), is_split_cart=True)
    f.clear()
    assert closed == [True], "the split-cart sidecar was leaked"
    assert f.sqlite_conn is None


def test_clear_survives_a_sidecar_that_raises_on_close():
    class BadConn:
        def close(self):
            raise RuntimeError("already closed")

    f = CartFields(sqlite_conn=BadConn())
    f.clear()
    assert f.sqlite_conn is None


# -- the drift guard ----------------------------------------------------------

def test_cart_fields_matches_what_engine_unmount_clears():
    """`EngineManager.unmount()` is the de-facto definition of per-cart state.

    A field cleared there but missing here would keep its value across a mount -- one cart's
    tombstones or permissions applied to the next, which is the worst class of bug this
    module could produce. A field here but not there is dead weight. Either way the two must
    agree, and remembering is not a mechanism.
    """
    from api.engine import EngineManager

    src = inspect.getsource(EngineManager.unmount)
    cleared = set(re.findall(r"self\.(\w+)\s*=", src))
    declared = set(CartFields.__dataclass_fields__)              # noqa: SLF001

    missing = cleared - declared
    assert not missing, (
        f"engine.unmount() clears these but CartFields does not declare them: "
        f"{sorted(missing)} -- they would survive a cart switch")
