"""May this seat mount this cart? The check the mount path never made.

Three ends existed and did not meet at mount time:

    auth.py       -- WHO is asking            (Supabase JWT -> seat UUID)
    profiles.py   -- what a LEVEL permits     (can_here(effective, capability))
    db/004        -- WHICH carts, at what level (user_carts + cart_grants)

`_dispatch_mount` consulted none of them. `cart_grants` was computed for DISPLAY in
ProfileScreen and enforced by Postgres RLS on the *rows*, but the mount endpoint opens a
file on disk, and a file on disk has no RLS. Net effect before this module: a cart a seat
was never granted did not appear in their list, and mounted fine if the name was reached
directly. Scope without secrecy.

WHAT THIS MODULE CAN AND CANNOT BE
-----------------------------------
Read `docs/GUIDE-MULTI-SEAT.md` §"scope is not secrecy" first, then this, because the
distinction lands differently here than it does on sub-cartridges.

On a HOSTED deployment (the droplet), this is a real boundary: the user reaches carts only
through the API, so refusing the mount refuses the data.

On a LOCAL desktop studio it is NOT a boundary and cannot be made into one. The person
running the studio owns the disk the cart sits on; anything the API declines to open, they
can open with a text editor. Enforcement here is scope -- keeping other people's carts out
of your way and out of your hot stack -- and calling it secrecy on a local install would be
a claim the filesystem contradicts.

That is why `enforcement_available()` exists and why its absence is not a failure: with no
auth configured there is no multi-user story to enforce, and pretending otherwise would
add a lock to a door standing in an open field.

THE LEGACY RULE (Andy, 2026-08-03)
-----------------------------------
    "If they are editable then anyone can write them and if they are read-only then no one
     can write them. It's just easier than messing with updating them all to the new rules."

So a cart with no `user_carts` row is READABLE by anyone and its writability is whatever
the existing read-only flag already said. No owner backfill, no migration, no per-cart
decision. It degrades correctly under any agent rule chosen later, because the flag keeps
meaning the same thing.

FAIL CLOSED, BUT ONLY WHERE CLOSING MEANS ANYTHING
---------------------------------------------------
Absence of a grant on a REGISTERED cart is a denial, not a default -- same rule as
profiles.py, which found two fail-open paths in the sidecar reader on 2026-08-01. But a
lookup that never ran (no auth configured) is a different fact from a lookup that returned
nothing, and the two must not collapse into one branch. `DECISION_UNENFORCED` keeps them
apart so a reader of the audit log can tell "allowed because nobody is restricted" from
"allowed because this seat was granted."
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .profiles import OWNER, can_here

log = logging.getLogger(__name__)

# Why the mount was allowed or refused. Distinct values rather than a bool because the
# audit line for each is a different sentence, and because "unenforced" must never read as
# "granted" when we are deciding what a demo may honestly claim.
DECISION_OWNER = "owner"
DECISION_GRANTED = "granted"
DECISION_UNREGISTERED = "unregistered"      # legacy cart, no user_carts row
DECISION_UNENFORCED = "unenforced"          # no auth configured; local single-user mode
DECISION_NO_GRANT = "no-grant"              # registered, seat has no grant  -> refuse
DECISION_ANONYMOUS = "anonymous"            # registered, no seat at all     -> refuse
DECISION_LOOKUP_FAILED = "lookup-failed"    # auth configured, DB unreachable -> refuse


@dataclass(frozen=True)
class MountDecision:
    """Outcome of the access check, with enough detail to log and to explain."""

    allowed: bool
    reason: str
    level: str | None = None          # 'owner' | 'viewer' | 'commenter' | 'editor' | None
    enforced: bool = True             # False when the deployment cannot enforce at all

    @property
    def may_write(self) -> bool:
        """Whether the resolved level permits cart writes.

        Unregistered and unenforced carts return False HERE and are handled by the caller
        via the existing read-only flag -- this property answers "did a grant authorise a
        write", not "is the cart writable", and conflating those is how a legacy cart would
        silently become read-only against Andy's stated rule.
        """
        return bool(self.level) and can_here(self.level, "write")

    def audit(self, cart: str, seat: str | None) -> str:
        verb = "allowed" if self.allowed else "REFUSED"
        who = seat or "anonymous"
        lvl = f" as {self.level}" if self.level else ""
        return f"mount {verb}{lvl}: cart={cart!r} seat={who} reason={self.reason}"


def enforcement_available() -> bool:
    """True when the deployment has an auth system whose answers could bind.

    Deliberately reads the environment at call time rather than at import: the tests and
    the desktop build both toggle it, and a module-level constant would freeze whichever
    state happened to exist when the first import ran.
    """
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY"))


def decide(*, registered: bool, owner_id: str | None, grant_level: str | None,
           seat: str | None, enforced: bool = True) -> MountDecision:
    """Pure decision. No I/O, no framework -- the lookup is the caller's problem.

    Split out so the rules can be tested exhaustively without a database, because every
    permission bug we have shipped was in the *rules*, not in the query.
    """
    if not enforced:
        return MountDecision(True, DECISION_UNENFORCED, None, enforced=False)

    # Legacy carts. Andy's rule: readable by anyone, writability per the read-only flag.
    if not registered:
        return MountDecision(True, DECISION_UNREGISTERED, None)

    if seat and owner_id and seat == owner_id:
        return MountDecision(True, DECISION_OWNER, OWNER)

    if not seat:
        return MountDecision(False, DECISION_ANONYMOUS, None)

    if grant_level:
        return MountDecision(True, DECISION_GRANTED, grant_level)

    return MountDecision(False, DECISION_NO_GRANT, None)


def lookup_failed() -> MountDecision:
    """A deployment with auth whose grant lookup did not complete.

    Refuses. On a hosted deployment a database outage must not become an amnesty on every
    cart at once, which is what allowing here would mean -- the one moment enforcement is
    unavailable is exactly the moment it would matter. Distinct from DECISION_NO_GRANT so
    an operator reading the log sees "could not check" rather than "checked and said no",
    since only one of those is fixed by restarting Postgres.
    """
    return MountDecision(False, DECISION_LOOKUP_FAILED, None)


UNREGISTERED = "unregistered"


def lookup(client, cart_filename: str, seat: str | None) -> MountDecision:
    """Ask Postgres for the caller's effective access. See db/005_cart_access_for.sql.

    ONE RPC, NOT TWO TABLE READS, AND THE REASON IS A FAIL-OPEN
    ------------------------------------------------------------
    The obvious client-side version -- select the cart from `user_carts`, then its row from
    `cart_grants` -- cannot work through a caller-scoped client. RLS returns zero rows both
    when the cart is unregistered AND when it is registered but this seat has no grant. The
    first must be ALLOWED (Andy's legacy rule); the second must be REFUSED. Same empty list,
    opposite answers, and the permissive branch is the one a test against a live database
    would silently take.

    `cart_access_for()` is SECURITY DEFINER so it can tell those apart, and returns a single
    string so it cannot be used to read anything else -- the shape `owns_cart()` already
    established in 004.

    Raises nothing of its own; a transport failure propagates to the caller, which owns the
    policy for what an unanswerable lookup means.
    """
    level = client.rpc("cart_access_for", {"p_filename": cart_filename}).execute().data

    if isinstance(level, list):          # some client versions wrap scalar returns
        level = level[0] if level else None
    if isinstance(level, dict):
        level = level.get("cart_access_for")

    if level == UNREGISTERED:
        return decide(registered=False, owner_id=None, grant_level=None, seat=seat)
    if level == OWNER:
        return decide(registered=True, owner_id=seat, grant_level=None, seat=seat)
    return decide(registered=True, owner_id=None, grant_level=level, seat=seat)
