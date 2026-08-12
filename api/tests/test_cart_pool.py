"""The pool must never drop a cart somebody is using.

That is the whole point. On 2026-08-12 we measured the failure it replaces: Betty mounting
Revenue dropped Susie's Finance mid-session, and if Susie happened to have a grant on Revenue
she read Revenue believing it was Finance. Every test here is ultimately about that.

The clock is injected, so none of these sleep.
"""

import pytest

from api.cart_pool import CartPool, CartState, PoolFull


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def loader_for(name):
    """A loader that records how many times it actually ran."""
    calls = []

    def load():
        calls.append(name)
        return {"cart": name}

    load.calls = calls
    return load


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def pool(clock):
    return CartPool(max_carts=3, clock=clock)


# -- the bug this module exists to fix ---------------------------------------

def test_two_seats_hold_two_different_carts_at_once(pool):
    """Susie on Finance and Betty on Revenue. THE regression test for 2026-08-12."""
    pool.acquire("finance", loader_for("finance"), seat="susie")
    pool.acquire("revenue", loader_for("revenue"), seat="betty")

    assert pool.get("finance").payload == {"cart": "finance"}
    assert pool.get("revenue").payload == {"cart": "revenue"}
    assert len(pool) == 2


def test_betty_mounting_cannot_disturb_susie(pool):
    susie_cart = pool.acquire("finance", loader_for("finance"), seat="susie")
    pool.acquire("revenue", loader_for("revenue"), seat="betty")

    still = pool.get("finance")
    assert still is susie_cart, "Susie's cart object was replaced out from under her"
    assert "susie" in still.holders


def test_two_seats_on_one_cart_share_a_single_copy(pool):
    """The 200-seat efficiency win: N readers, one copy in memory."""
    load = loader_for("company")
    a = pool.acquire("company", load, seat="susie")
    b = pool.acquire("company", load, seat="betty")

    assert a is b
    assert len(load.calls) == 1, "the cart was loaded twice for two readers"
    assert a.holders == {"susie", "betty"}


# -- pinning ------------------------------------------------------------------

def test_a_held_cart_is_never_evicted_for_capacity(pool):
    for name in ("a", "b", "c"):
        pool.acquire(name, loader_for(name), seat=f"seat_{name}")

    with pytest.raises(PoolFull):
        pool.acquire("d", loader_for("d"), seat="seat_d")

    for name in ("a", "b", "c"):
        assert name in pool, f"{name} was evicted while a seat held it"


def test_released_carts_are_evictable_but_stay_warm(pool, clock):
    pool.acquire("a", loader_for("a"), seat="susie")
    pool.release("a", "susie")

    assert "a" in pool, "release() unmounted the cart; it should only unpin it"
    assert not pool.peek("a").pinned


def test_capacity_evicts_the_least_recently_used_unpinned_cart(pool, clock):
    for name in ("a", "b", "c"):
        pool.acquire(name, loader_for(name), seat="s")
        pool.release(name, "s")
        clock.advance(10)

    pool.get("a")                      # 'a' becomes most-recent, 'b' the oldest
    pool.acquire("d", loader_for("d"), seat="s")

    assert "b" not in pool
    assert "a" in pool and "c" in pool and "d" in pool


def test_the_arriving_cart_is_never_its_own_eviction_victim(clock):
    p = CartPool(max_carts=1, clock=clock)
    p.acquire("a", loader_for("a"), seat="s")
    p.release("a", "s")
    p.acquire("b", loader_for("b"), seat="s")

    assert "b" in p, "the newcomer evicted itself to make room for itself"
    assert "a" not in p


# -- idle eviction ------------------------------------------------------------

def test_idle_eviction_skips_carts_in_use(pool, clock):
    pool.acquire("held", loader_for("held"), seat="susie")
    pool.acquire("idle", loader_for("idle"), seat="betty")
    pool.release("idle", "betty")

    clock.advance(3600)
    gone = pool.evict_idle(max_idle_seconds=1800)

    assert gone == ["idle"]
    assert "held" in pool, "a cart with a live holder was evicted for being idle"


def test_peek_does_not_reset_the_idle_clock(pool, clock):
    """An anonymous /api/status poll must not be able to pin a private cart forever."""
    pool.acquire("c", loader_for("c"), seat="s")
    pool.release("c", "s")

    clock.advance(3600)
    pool.peek("c")

    assert pool.evict_idle(max_idle_seconds=1800) == ["c"]


def test_get_does_reset_the_idle_clock(pool, clock):
    pool.acquire("c", loader_for("c"), seat="s")
    pool.release("c", "s")

    clock.advance(1700)
    pool.get("c")
    clock.advance(1000)

    assert pool.evict_idle(max_idle_seconds=1800) == []


# -- session lifecycle --------------------------------------------------------

def test_release_seat_unbinds_every_cart_that_seat_held(pool):
    pool.acquire("a", loader_for("a"), seat="susie")
    pool.acquire("b", loader_for("b"), seat="susie")
    pool.acquire("b", loader_for("b"), seat="betty")

    released = pool.release_seat("susie")

    assert sorted(released) == ["a", "b"]
    assert not pool.peek("a").pinned
    assert pool.peek("b").holders == {"betty"}, "signing out Susie unpinned Betty's cart too"


def test_drop_removes_a_cart_even_when_held(pool):
    pool.acquire("a", loader_for("a"), seat="susie")
    assert pool.drop("a") is True
    assert "a" not in pool


# -- byte budget --------------------------------------------------------------

def test_byte_budget_evicts_even_when_cart_count_is_fine(clock):
    p = CartPool(max_carts=100, max_bytes=250, clock=clock)
    p.acquire("a", loader_for("a"), seat="s", nbytes=100)
    p.release("a", "s")
    clock.advance(10)
    p.acquire("b", loader_for("b"), seat="s", nbytes=100)
    p.release("b", "s")
    clock.advance(10)
    p.acquire("c", loader_for("c"), seat="s", nbytes=100)

    assert p.total_bytes() <= 250
    assert "a" not in p, "the oldest cart should have gone first"
    assert "c" in p


# -- the load race ------------------------------------------------------------

def test_a_lost_load_race_yields_one_shared_state(pool):
    """Both seats must end up on ONE object, or they get two views of one cart's tombstones."""
    first = pool.acquire("c", loader_for("c"), seat="susie")

    def racing_loader():
        return {"cart": "c", "copy": 2}

    second = pool.acquire("c", racing_loader, seat="betty")

    assert second is first
    assert "copy" not in second.payload, "the racing loader's copy replaced the pooled one"
    assert second.holders == {"susie", "betty"}
