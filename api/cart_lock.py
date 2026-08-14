"""One writer per cart, and a name to put on the refusal.

WHY NOW. Before the cart pool, two seats could not write the same cart concurrently because
they could not have the same cart open -- whoever mounted last owned the process. The pool
fixed that for reads and, in doing so, made concurrent writes REACHABLE: two seats now share
one `CartFields`, so two requests can mutate `passages`, `deleted_ids` and `embeddings` at the
same time. FastAPI runs sync endpoints in a threadpool, so that concurrency is genuine.

TWO LAYERS, DELIBERATELY NOT ONE
================================================================================
`_mutex`  -- held only across the actual mutation. Microseconds. Guarantees no interleaved
             writes to one cart's in-memory state. THIS is the correctness half.
`_lease`  -- holder + expiry, outliving the request. THIS is the human half: it is what lets
             a second writer be told *"Betty is editing this cart"* instead of silently
             queueing behind her.

They cannot be one thing. A `threading.Lock` held across requests cannot be released by a
different thread and would wedge the process the first time a browser tab closed mid-edit.
A lease alone would not stop two threads interleaving inside one request window.

PORTED FROM `membot/membox.py:CartLock`, WITH ITS RACE FIXED. That version checks lease
expiry against `self._holder` outside any lock and, on expiry, REPLACES `self._mutex`. Two
threads can therefore both decide the lease expired and both swap the mutex -- including out
from under a holder who is legitimately mid-write. Unreachable there today (one writer);
reachable here on day one. Every field below is read and written under `_meta`.

⚠ IN-PROCESS ONLY, and that is load-bearing. This is a module-global dict, so it does not
survive `--workers N`; a second worker gets its own lock table and no mutual exclusion, with
nothing logging a warning. Recorded in DROPLET-DIVERGENCE-MAP.md §4.

Design: docs/DESIGN-multi-mount-and-write-path.md §5.1
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

__all__ = ["CartBusy", "LeaseInfo", "CartWriteLock", "locks", "DEFAULT_LEASE_SECONDS"]

# How long a writer keeps the claim without writing again. Long enough to think and type,
# short enough that a closed tab frees the cart before anyone files a bug.
DEFAULT_LEASE_SECONDS = 90.0

# How long a second writer waits for the mutation itself. Writes are short; a caller waiting
# longer than this is queued behind something pathological and deserves an answer, not a hang.
DEFAULT_WAIT_SECONDS = 5.0


@dataclass(frozen=True)
class LeaseInfo:
    """Who holds a cart's write claim, and until when."""

    holder: str
    display_name: Optional[str]
    expires_at: float

    def seconds_left(self, now: float) -> float:
        return max(0.0, self.expires_at - now)


class CartBusy(RuntimeError):
    """Someone else holds the write claim. Carries WHO, so the caller can say so.

    Raised rather than returned because every write path must handle it, and a return value
    can be ignored -- which is exactly how a refused write got reported as success on
    2026-08-10 and again on 2026-08-13.
    """

    def __init__(self, cart_id: str, lease: LeaseInfo, now: float):
        self.cart_id = cart_id
        self.lease = lease
        self.seconds_left = lease.seconds_left(now)
        who = lease.display_name or "Someone else"
        super().__init__(f"{who} is editing this cart.")


class CartWriteLock:
    """The write claim for ONE cart."""

    def __init__(self, cart_id: str, *, lease_seconds: float = DEFAULT_LEASE_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self.cart_id = cart_id
        self._lease_seconds = lease_seconds
        self._clock = clock
        self._mutex = threading.Lock()          # held ONLY across a mutation
        self._meta = threading.Lock()           # guards every field below
        self._lease: Optional[LeaseInfo] = None
        self.acquire_count = 0
        self.refusal_count = 0

    # -- lease ---------------------------------------------------------------

    def current_lease(self) -> Optional[LeaseInfo]:
        """The live lease, or None. Expired leases read as None rather than as held."""
        with self._meta:
            return self._live_lease_locked()

    def _live_lease_locked(self) -> Optional[LeaseInfo]:
        if self._lease is None:
            return None
        if self._clock() >= self._lease.expires_at:
            # Expiry is lazy on purpose: no background sweeper, and a lease nobody asks
            # about costs nothing. A crashed or closed-tab writer frees the cart the moment
            # the next person tries.
            self._lease = None
        return self._lease

    def release(self, holder: str) -> bool:
        """Give up the claim. Only the holder may; anyone else is ignored, not an error.

        Ignoring rather than raising because release runs in cleanup paths, and a cleanup
        that can throw turns one failure into two.
        """
        with self._meta:
            if self._lease is not None and self._lease.holder == holder:
                self._lease = None
                return True
            return False

    # -- the write path ------------------------------------------------------

    @contextmanager
    def write(self, holder: str, display_name: Optional[str] = None, *,
              wait_seconds: float = DEFAULT_WAIT_SECONDS) -> Iterator[LeaseInfo]:
        """Hold the write claim and the mutation mutex for this block.

        Raises `CartBusy` if someone else holds a live lease -- BEFORE waiting on the mutex,
        so a refusal is immediate and names a person rather than timing out anonymously.

        Re-entrant for the same holder: a writer who already has the lease renews it instead
        of blocking on themselves. Without that, any write path calling another would
        deadlock, and the deadlock would only appear under a code path nobody tested.
        """
        now = self._clock()
        with self._meta:
            live = self._live_lease_locked()
            if live is not None and live.holder != holder:
                self.refusal_count += 1
                raise CartBusy(self.cart_id, live, now)
            # Ours, or nobody's. Claim it and start the lease at the WRITE, not at the
            # request -- a slow request should not spend its own lease waiting.
            self._lease = LeaseInfo(holder=holder, display_name=display_name,
                                    expires_at=now + self._lease_seconds)
            self.acquire_count += 1

        got = self._mutex.acquire(timeout=wait_seconds)
        if not got:
            # The lease is ours but the mutex is not, which means another request by the
            # SAME holder is mid-write -- two tabs signed in as one person, most likely.
            # Refuse rather than hang; the data is fine either way.
            with self._meta:
                live = self._lease or LeaseInfo(holder, display_name, self._clock())
            raise CartBusy(self.cart_id, live, self._clock())

        try:
            yield self._lease                                    # type: ignore[misc]
        finally:
            self._mutex.release()
            # Renew from the END of the write. A long save should not leave a lease that
            # expires the instant it finishes.
            with self._meta:
                if self._lease is not None and self._lease.holder == holder:
                    self._lease = LeaseInfo(holder, display_name,
                                            self._clock() + self._lease_seconds)


class _LockTable:
    """cart_id -> CartWriteLock, created on demand."""

    def __init__(self, *, lease_seconds: float = DEFAULT_LEASE_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self._locks: dict[str, CartWriteLock] = {}
        self._guard = threading.Lock()
        self._lease_seconds = lease_seconds
        self._clock = clock

    def for_cart(self, cart_id: str) -> CartWriteLock:
        with self._guard:
            lock = self._locks.get(cart_id)
            if lock is None:
                lock = CartWriteLock(cart_id, lease_seconds=self._lease_seconds,
                                     clock=self._clock)
                self._locks[cart_id] = lock
            return lock

    def holder_of(self, cart_id: str) -> Optional[LeaseInfo]:
        """Who is editing, without creating a lock for a cart nobody has written."""
        with self._guard:
            lock = self._locks.get(cart_id)
        return lock.current_lease() if lock is not None else None

    def clear(self) -> None:
        """FOR TESTS. Production never drops locks -- an unheld one costs nothing."""
        with self._guard:
            self._locks.clear()


locks = _LockTable()
