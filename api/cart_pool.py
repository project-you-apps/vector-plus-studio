"""Many carts mounted at once, keyed by cart — not one global mount for everybody.

THE BUG THIS EXISTS TO FIX (measured 2026-08-12). Susie mounts Finance, Betty mounts
Revenue, and `engine.unmount()` drops Finance out from under Susie. Her next search resolves
against Revenue. If she has no grant she gets a 403 on a cart she opened thirty seconds ago;
if she DOES have a grant she reads Revenue while her UI still says Finance -- a
misattribution failure, which is the worse one because nothing looks broken.

Susie and Betty were never competing for memory. They were competing for one variable.

THE REFRAME: MOUNTS ARE PER-CART, NOT PER-SEAT. A 200-seat office does not need 200 mounts.
It needs one copy of each DISTINCT cart in use -- realistically 5-20 -- shared by everyone
entitled to read it. What is per-seat is a single string: which cart_id am I looking at.

WHAT DOES NOT LIVE HERE. The CUDA lattice, the SentenceTransformer embedder and the four
encoders are MACHINE state: loaded once, shared by every cart, never duplicated. An earlier
estimate (2026-08-05) put multi-mount at "three CUDA engines and three copies of the
embedder" -- true of three separate PROCESSES, and only of those. In-process, only the cart
payload multiplies, which is roughly an order of magnitude cheaper.

DELIBERATELY IGNORANT OF numpy, cartridges and the engine. The pool takes a `loader`
callable and stores whatever it returns. That keeps the concurrency logic testable without
building a cart, and it is why the tests here run in milliseconds.

Prior art: `membot/multi_cart.py` has done exactly this for months across Dennis's six
machines. The studio is the outlier, not the pioneer.

Design: docs/DESIGN-multi-mount-and-write-path.md
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

__all__ = ["CartState", "CartPool", "PoolFull"]


class PoolFull(RuntimeError):
    """Every cart in the pool is pinned, and the newcomer does not fit.

    Distinct from an eviction: eviction is routine and silent, this is a refusal we owe the
    caller an explanation for. Raised rather than evicting a pinned cart, because dropping a
    cart somebody is actively reading is the exact failure this module was built to end.
    """


@dataclass
class CartState:
    """One mounted cart. The unit the pool hands out and evicts.

    `payload` is whatever the loader returned -- in practice the arrays that
    `engine.unmount()` already enumerates (embeddings, passages, hippocampus, signatures,
    sqlite_conn, permissions, flags). That list is the struct definition; it is deliberately
    NOT restated here, because restating it would create a second place to keep in agreement
    and the drift would be silent.
    """

    cart_id: str
    payload: Any
    loaded_at: float
    last_access: float
    nbytes: int = 0

    # Seats currently bound to this cart. A cart with holders is PINNED and cannot be
    # evicted -- idle time is about the cart being unused, not about the clock.
    holders: set[str] = field(default_factory=set)

    @property
    def pinned(self) -> bool:
        return bool(self.holders)

    def idle_seconds(self, now: float) -> float:
        return max(0.0, now - self.last_access)


class CartPool:
    """cart_id -> CartState, with reference counting and LRU/idle eviction.

    Thread-safe by one coarse lock. The lock protects the DICT, never a load: a slow cart
    load must not block every other seat's search, so loads happen outside it. The cost is
    that two seats can race to load the same cart, which is handled by letting both load and
    keeping the first to arrive -- wasteful once, correct always. The alternative
    (per-cart-id load locks) is more machinery than a rare double-load is worth.
    """

    def __init__(self, *, max_carts: int = 8, max_bytes: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._carts: dict[str, CartState] = {}
        self._lock = threading.Lock()
        self._clock = clock          # injectable so tests do not sleep
        self.max_carts = max_carts
        self.max_bytes = max_bytes

    # -- inspection ---------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._carts)

    def __contains__(self, cart_id: str) -> bool:
        with self._lock:
            return cart_id in self._carts

    def __iter__(self) -> Iterator[CartState]:
        with self._lock:
            return iter(list(self._carts.values()))

    def total_bytes(self) -> int:
        with self._lock:
            return sum(c.nbytes for c in self._carts.values())

    def peek(self, cart_id: str) -> Optional[CartState]:
        """The cart WITHOUT touching its clock. For status and admin views.

        Separate from `get` on purpose: an anonymous caller polling /api/status must not be
        able to hold a private cart open forever by keeping its idle timer reset.
        """
        with self._lock:
            return self._carts.get(cart_id)

    # -- the hot path -------------------------------------------------------

    def get(self, cart_id: str, seat: Optional[str] = None) -> Optional[CartState]:
        """The cart, marking it used. Returns None if not mounted.

        `seat` exists so that only a seat entitled to the cart resets its clock. The caller
        resolves entitlement (cart_guard does that); the pool just records who asked.
        """
        with self._lock:
            state = self._carts.get(cart_id)
            if state is not None:
                state.last_access = self._clock()
            return state

    def acquire(self, cart_id: str, loader: Callable[[], Any], seat: str,
                *, nbytes: int = 0) -> CartState:
        """Bind `seat` to `cart_id`, loading the cart if nobody has it open yet.

        The load runs OUTSIDE the lock. Two seats asking for the same cold cart will both
        load it; the first to finish wins and the loser's copy is dropped. That is a wasted
        load, not a correctness problem, and it keeps one slow load from stalling the pool.
        """
        now = self._clock()

        with self._lock:
            state = self._carts.get(cart_id)
            if state is not None:
                state.holders.add(seat)
                state.last_access = now
                return state

        payload = loader()                       # slow; deliberately unlocked

        with self._lock:
            existing = self._carts.get(cart_id)
            if existing is not None:
                # Lost the race. Keep the copy already in the pool so every seat shares one
                # object -- two CartStates for one cart would mean two views of one cart's
                # tombstones, which is the drift this whole module exists to prevent.
                existing.holders.add(seat)
                existing.last_access = self._clock()
                return existing

            state = CartState(cart_id=cart_id, payload=payload, loaded_at=now,
                              last_access=now, nbytes=nbytes)
            state.holders.add(seat)
            self._carts[cart_id] = state
            self._evict_locked(protect=cart_id)
            return state

    def release(self, cart_id: str, seat: str) -> None:
        """Unbind a seat. The cart STAYS mounted -- unpinned, so now evictable.

        Releasing is not unmounting. The next seat to want this cart should find it warm;
        eviction decides when it actually goes, based on pressure and idleness rather than on
        one user closing a tab.
        """
        with self._lock:
            state = self._carts.get(cart_id)
            if state is not None:
                state.holders.discard(seat)

    def release_seat(self, seat: str) -> list[str]:
        """Unbind this seat from every cart. For sign-out and session expiry.

        Returns the carts it was holding, so the caller can log what a departing session was
        actually using.
        """
        with self._lock:
            held = [cid for cid, s in self._carts.items() if seat in s.holders]
            for cid in held:
                self._carts[cid].holders.discard(seat)
            return held

    def drop(self, cart_id: str) -> bool:
        """Remove a cart outright, pinned or not. For an explicit unmount or an admin action.

        The blunt instrument. Eviction is the polite one.
        """
        with self._lock:
            return self._carts.pop(cart_id, None) is not None

    # -- eviction -----------------------------------------------------------

    def evict_idle(self, max_idle_seconds: float) -> list[str]:
        """Drop unpinned carts idle longer than the limit. Returns what went.

        Called lazily from requests that are already happening -- no background thread, no
        scheduler. A pool nobody is using never needs sweeping.
        """
        now = self._clock()
        with self._lock:
            doomed = [cid for cid, s in self._carts.items()
                      if not s.pinned and s.idle_seconds(now) > max_idle_seconds]
            for cid in doomed:
                del self._carts[cid]
            return doomed

    def _evict_locked(self, *, protect: Optional[str] = None) -> list[str]:
        """Enforce max_carts / max_bytes by dropping least-recently-used UNPINNED carts.

        `protect` is the cart that just arrived: evicting it to make room for itself would be
        both useless and very confusing to debug.

        Raises PoolFull rather than evicting a pinned cart. Refusing a newcomer is a bad
        afternoon; dropping the cart somebody is mid-search on is the bug this module was
        written to eliminate, and doing it under memory pressure would make it intermittent.
        """
        evicted: list[str] = []

        def over_budget() -> bool:
            if len(self._carts) > self.max_carts:
                return True
            if self.max_bytes is not None:
                return sum(c.nbytes for c in self._carts.values()) > self.max_bytes
            return False

        while over_budget():
            candidates = [s for s in self._carts.values()
                          if not s.pinned and s.cart_id != protect]
            if not candidates:
                raise PoolFull(
                    f"pool holds {len(self._carts)} carts and every one is in use; "
                    f"cannot make room for {protect!r}")
            victim = min(candidates, key=lambda s: s.last_access)
            del self._carts[victim.cart_id]
            evicted.append(victim.cart_id)

        return evicted


# One pool per process, mirroring how `engine` is a process singleton today. Kept module-level
# rather than on the app so a background task or a CLI can reach it without a Request.
#
# IN-PROCESS ONLY, and that is load-bearing: this dict is not shared across uvicorn workers.
# Running with --workers N would give each worker its own pool AND its own lock table, with
# nothing warning you. See DROPLET-DIVERGENCE-MAP.md §4.
pool = CartPool()
