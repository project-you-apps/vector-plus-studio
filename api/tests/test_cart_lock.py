"""One writer per cart, a name on the refusal, and no way to wedge the cart.

The pool made concurrent writes reachable: two seats now share one CartFields, so two
requests can mutate one cart's passages at the same time. This is what stops them.

The clock is injected; nothing here sleeps.
"""

import threading
import time

import pytest

from api.cart_lock import CartBusy, CartWriteLock, _LockTable


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def lock(clock):
    return CartWriteLock("redwood-company", lease_seconds=90.0, clock=clock)


# -- the claim ----------------------------------------------------------------

def test_a_cart_starts_unclaimed(lock):
    assert lock.current_lease() is None


def test_writing_takes_the_claim(lock):
    with lock.write("seat:susie", "Susie Nakamura"):
        pass
    live = lock.current_lease()
    assert live is not None and live.holder == "seat:susie"


def test_a_second_writer_is_refused_BY_NAME(lock):
    """Andy, 2026-08-12: 'Betty is editing this cart' is better than a spinner."""
    with lock.write("seat:betty", "Betty Alvarez"):
        pass

    with pytest.raises(CartBusy) as exc:
        with lock.write("seat:susie", "Susie Nakamura"):
            pass

    assert "Betty Alvarez is editing this cart." in str(exc.value)
    assert exc.value.lease.holder == "seat:betty"


def test_a_holder_without_a_display_name_still_refuses_usefully(lock):
    """A raw uuid in an error message is useless to an office manager."""
    with lock.write("seat:betty", None):
        pass
    with pytest.raises(CartBusy) as exc:
        with lock.write("seat:susie", "Susie"):
            pass
    assert "Someone else is editing this cart." in str(exc.value)
    assert "seat:betty" not in str(exc.value)


def test_the_same_holder_is_re_entrant(lock):
    """A write path that calls another must not deadlock on itself."""
    with lock.write("seat:susie", "Susie"):
        pass
    with lock.write("seat:susie", "Susie"):
        pass                      # would hang or raise if the claim blocked its own holder


def test_releasing_frees_the_cart(lock):
    with lock.write("seat:betty", "Betty"):
        pass
    assert lock.release("seat:betty") is True
    with lock.write("seat:susie", "Susie"):
        pass


def test_only_the_holder_may_release_and_a_stranger_is_ignored(lock):
    """Release runs in cleanup paths; a cleanup that throws turns one failure into two."""
    with lock.write("seat:betty", "Betty"):
        pass
    assert lock.release("seat:susie") is False
    assert lock.current_lease().holder == "seat:betty"


# -- the lease is what stops a wedged cart ------------------------------------

def test_an_expired_lease_frees_the_cart(clock, lock):
    """Betty closes her laptop mid-edit. The cart must not be hers forever."""
    with lock.write("seat:betty", "Betty"):
        pass

    clock.advance(91.0)
    assert lock.current_lease() is None
    with lock.write("seat:susie", "Susie"):
        pass


def test_a_live_lease_still_refuses(clock, lock):
    with lock.write("seat:betty", "Betty"):
        pass
    clock.advance(89.0)
    with pytest.raises(CartBusy):
        with lock.write("seat:susie", "Susie"):
            pass


def test_writing_again_renews_the_lease(clock, lock):
    with lock.write("seat:betty", "Betty"):
        pass
    clock.advance(80.0)
    with lock.write("seat:betty", "Betty"):
        pass
    clock.advance(80.0)
    assert lock.current_lease() is not None, "an active writer lost the cart mid-session"


def test_the_lease_renews_from_the_END_of_a_long_write(clock, lock):
    """A save that takes most of the lease must not expire the moment it finishes."""
    with lock.write("seat:betty", "Betty"):
        clock.advance(85.0)
    clock.advance(10.0)
    assert lock.current_lease() is not None


# -- serialisation: the correctness half --------------------------------------

def test_writes_by_one_holder_do_not_interleave():
    """The reason the mutex exists. FastAPI runs sync endpoints in a threadpool."""
    lock = CartWriteLock("c", lease_seconds=90.0, clock=time.monotonic)
    log: list[str] = []
    barrier = threading.Barrier(2)

    def writer(tag: str):
        barrier.wait()
        for _ in range(50):
            with lock.write("seat:susie", "Susie"):
                log.append(f"{tag}-in")
                time.sleep(0)          # yield; an unguarded section would interleave here
                log.append(f"{tag}-out")

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every in must be followed by its own out.
    for i in range(0, len(log), 2):
        assert log[i].endswith("-in"), log[i:i + 2]
        assert log[i + 1] == log[i].replace("-in", "-out"), log[i:i + 2]


def test_the_mutex_is_released_when_a_write_raises(lock):
    """A failed save must not leave the cart locked for everyone including its holder."""
    with pytest.raises(ValueError):
        with lock.write("seat:susie", "Susie"):
            raise ValueError("save exploded")

    with lock.write("seat:susie", "Susie"):
        pass                       # would hang if the mutex leaked


# -- the table ----------------------------------------------------------------

def test_the_table_returns_one_lock_per_cart(clock):
    table = _LockTable(clock=clock)
    assert table.for_cart("a") is table.for_cart("a")
    assert table.for_cart("a") is not table.for_cart("b")


def test_holder_of_does_not_create_a_lock_for_an_unwritten_cart(clock):
    table = _LockTable(clock=clock)
    assert table.holder_of("never-touched") is None
    assert table._locks == {}, "asking who holds a cart created a lock for it"


def test_two_carts_do_not_block_each_other(clock):
    table = _LockTable(clock=clock)
    with table.for_cart("company").write("seat:betty", "Betty"):
        pass
    with table.for_cart("finance").write("seat:susie", "Susie"):
        pass                       # a different cart must be unaffected
