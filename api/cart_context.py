"""Which cart is THIS request looking at.

THE PROBLEM (measured 2026-08-12). `engine` is a process-wide singleton and its cart fields
are plain attributes, so there is exactly one mounted cart for everybody. Susie mounts
Finance, Betty mounts Revenue, and `engine.unmount()` drops Finance out from under Susie. Her
next search resolves against Revenue -- a 403 on a cart she just opened if she has no grant,
and if she DOES have one, Revenue's passages under Finance's label. The second is worse:
nothing looks broken.

WHY A CONTEXTVAR AND NOT A REFACTOR. `engine.` appears 305 times in main.py, ~212 of them
touching cart state. Threading a CartState parameter through all of them is a diff where one
missed call site silently serves the wrong cart -- a worse failure than the bug being fixed.
Binding the arrays into `engine` under a global lock works but serialises every cart-touching
request. A ContextVar is per-async-task, so each request sees its own cart with no lock and
no call-site churn, and `contextvars` propagate across `asyncio.to_thread` (10 sites in
main.py rely on that).

⚠ THE COST, STATED PLAINLY. This makes `engine.passages` **implicitly request-scoped**. Two
callers reading the same attribute get different data, and nothing at the call site says so.
That is a real readability tax, accepted deliberately because the alternatives are a 212-site
diff or a global lock. If you are debugging "wrong passages," look here first.

THE DEFAULT IS WHAT KEEPS SINGLE-USER BEHAVIOUR IDENTICAL. Startup, CLI tools, background
tasks and every existing test run with nothing bound, so they get one process-wide CartFields
-- exactly today's semantics, one shared mounted cart. Multi-tenancy is what the binding adds
on top, not a new requirement everything must satisfy.

Design: docs/DESIGN-multi-mount-and-write-path.md §4
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

__all__ = ["CartFields", "active", "use_cart", "default_fields", "is_bound",
           "reset_default", "detach_default"]


@dataclass
class CartFields:
    """The per-cart half of EngineManager.

    THE FIELD LIST IS NOT INVENTED HERE. It is exactly what `EngineManager.unmount()` clears,
    because that method has always been the de-facto definition of "what belongs to a cart
    rather than to the machine." Keeping the two in agreement matters: a field that lives here
    but is not cleared there, or vice versa, is a cart-bleed bug waiting to happen.

    Everything NOT in this list -- the CUDA lattice, the SentenceTransformer, the four
    encoders, gpu_available, engine_ready -- is MACHINE state. It is loaded once and shared by
    every cart and every seat, and duplicating it was the mistake in the 2026-08-05 estimate
    that made multi-mount look ten times more expensive than it is.
    """

    mounted_name: Optional[str] = None
    mounted_path: Optional[str] = None
    cart_generation: int = 0
    embeddings: Any = None
    passages: list = field(default_factory=list)
    compressed_lens: list = field(default_factory=list)
    compressed_texts: list = field(default_factory=list)
    signatures: Any = None
    signatures_loaded: bool = False
    multimodal_mode: bool = False
    brain_only_mode: bool = False
    physics_trained: bool = False
    deleted_ids: set = field(default_factory=set)
    dirty: bool = False
    read_only: bool = True

    # ⚠ Training progress is per-cart HERE because `unmount()` clears it, and this class
    # deliberately mirrors that method rather than re-deciding it. But the GPU underneath is
    # machine state and can only train one cart at a time (serialised by `engine.lock`), so
    # per-cart `training_active` cannot by itself stop two seats from starting a run. Today
    # that is unreachable -- one cart, one trainer -- and with a pool it becomes reachable.
    # Flagged in UNWIRED-AND-UNBUILT rather than fixed here: changing the semantics is a
    # separate decision from moving the field.
    training_active: bool = False
    training_progress: int = 0
    training_total: int = 0
    cart_permissions: Any = None
    hippocampus: Any = None
    sqlite_conn: Any = None
    sqlite_db_path: Optional[str] = None
    is_split_cart: bool = False

    def clear(self) -> None:
        """Return to the nothing-mounted state, closing the sidecar if we opened one."""
        if self.sqlite_conn is not None:
            try:
                self.sqlite_conn.close()
            except Exception:                                   # noqa: BLE001
                pass
        fresh = CartFields()
        for name in self.__dataclass_fields__:                  # noqa: SLF001
            setattr(self, name, getattr(fresh, name))


# The process-wide cart, used whenever nothing is bound. This IS today's behaviour: one
# mounted cart shared by the whole process. Every existing caller keeps it.
_default = CartFields()

_active: contextvars.ContextVar[Optional[CartFields]] = contextvars.ContextVar(
    "vps_active_cart", default=None)


def default_fields() -> CartFields:
    """The process-wide cart used when nothing is bound. For startup and CLI callers."""
    return _default


def detach_default() -> CartFields:
    """Hand the current process-wide cart to a new owner and install a fresh one.

    ⚠ EXISTS BECAUSE CLEARING IT IN PLACE DESTROYED IT. The mount route publishes the loaded
    cart into the pool and then wants the default empty again. `active()` returns THIS OBJECT
    when nothing is bound, so publishing a reference and then calling `.clear()` emptied the
    very object the pool had just taken -- mount reported success and every search found
    nothing (2026-08-14).

    Swapping the object is also cheaper than copying: no array is touched, the new owner keeps
    the loaded embeddings, and the default starts genuinely empty.
    """
    global _default
    previous = _default
    _default = CartFields()
    return previous


def reset_default() -> None:
    """Clear the process-wide cart. FOR TESTS -- production code should bind instead."""
    _default.clear()


def is_bound() -> bool:
    """True if this task has its own cart. Useful for asserting a request actually bound."""
    return _active.get() is not None


def active() -> CartFields:
    """The cart THIS task is looking at, or the process default.

    Never returns None, so `engine.passages` cannot raise merely because a caller forgot to
    bind -- it degrades to the single-user behaviour that has always been correct for
    startup, tests and the local studio, rather than to an exception.
    """
    current = _active.get()
    return current if current is not None else _default


@contextmanager
def use_cart(fields: CartFields) -> Iterator[CartFields]:
    """Bind `fields` for the duration of this block, then restore whatever was there.

    Token-based reset rather than set-to-None, because a nested bind must restore its parent
    and not the default. Nesting is not expected today; getting it wrong silently would be
    the kind of bug this module exists to prevent.
    """
    token = _active.set(fields)
    try:
        yield fields
    finally:
        _active.reset(token)
