"""May this seat read THIS DOCUMENT inside a cart it already has access to?

The layer between `cart_access` (may you open the cart at all) and `cartridge_io`'s PERM_R
(is this passage readable by anyone). Neither of those can say "Betty may read the company
cart, but not the three documents about her own compensation," which is the thing Andy
actually asked for on 2026-08-05 and the thing Dennis asked about for the fleet on 08-08.

WHY THIS IS NOT AN EXTENSION OF `seat_hide`
============================================
The tempting shortcut, and the one I proposed on 08-09, is to reuse the per-seat overlay:
it is already sparse, already content-keyed, already per-seat. It is the wrong object.

`seat_hide`'s own docstring is "Hide passages from YOUR view of the mounted cart. Affects
nobody else." The seat owns that file. Writing owner-imposed restrictions into it would
mean the restricted seat can unhide what the owner hid, and would collapse "this is noise
to me" and "this is forbidden to me" into the same byte. `overlay.py` also warns that an
overlay OUTLIVES ACCESS, which is a sane property for attention and a disqualifying one for
permission.

So: two objects, two authorities, two lifetimes. Attention stays with the seat. Restriction
lives server-side, written by the owner, and is what this module resolves.

WHAT AN "OBJECT" IS, AND WHY IT IS NOT A PASSAGE
================================================
Permissions attach to the largest thing that contains the passage — Andy, 08-09: "The only
time it would be hung on a passage is if the passage itself was isolated and didn't have a
file parent that contained it."

That is not just ergonomics. The session cart is ~39,000 passages; a control nobody can
maintain is a control that does not exist. Documents are what a person can reason about.

The identity was already on disk and we nearly missed it: h-block offset 18 is a uint32
`source_hash`, a deterministic md5 of the source filename, written by
`cartridge_builder.build_metadata` since long before any of this. Verified 2026-08-10 —
each Redwood office cart carries exactly 300 distinct values over ~1,400 passages, and all
300 resolve back to real filenames. So `object_ref = (cart, source_hash)` needs no format
change, no rebuild, and no migration.

THE THREE LAYERS, AND WHY "MOST SPECIFIC WINS" IS NOT ARBITRARY
===============================================================
    cart grant        Betty is a viewer on `company`     cart_access / Supabase
    object exposure   this doc requires >= editor        this module + inheritance mode
    seat exception    ...except Betty, who may not       this module + a sparse table

Most specific wins because every other order makes the specific rule unwritable: if the
cart grant took precedence there would be no way to carve anything out of it, and carving
out is the entire feature.

`access.py` already stated the other half of the composition — "A passage is readable iff
access_level grants read AND that passage's PERM_R permits it. Person-capability ∩
object-sensitivity." This module is the same intersection with a third term, so PERM_R is
applied here too rather than being left for a caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass

from .access import ACCESS_LEVELS
from .profiles import OWNER, can_here

# --------------------------------------------------------------------------- vocabulary

# An exception may GRANT a level or REVOKE entirely. Revocation needs its own value: an
# absent row means "no exception recorded", and if that were spelled the same way as "this
# seat is denied" the table could never distinguish "not yet decided" from "decided no" —
# the same fail-open collapse `cart_access.lookup` exists to prevent one layer up.
DENY = "deny"

# How this seat's access to this object was arrived at. Recorded, not derived, because the
# inheritance toggle can be flipped later and a decision made under the old policy must
# still be readable as such. Andy, 08-09: grandfathered documents get flagged in the UI, so
# this is a value with a declared consumer rather than another lifecycle flag nobody reads.
ORIGIN_INHERITED = "inherited"      # took the cart's level because the cart says inherit
ORIGIN_EXPLICIT = "explicit"        # an exposure level was set on the object itself
ORIGIN_EXCEPTION = "exception"      # a per-seat row governs
ORIGIN_OWNER = "owner"              # the cart owner, who is never locked out by exposure
ORIGIN_WITHHELD = "withheld"        # explicit-mode cart, nobody said this seat may see it


@dataclass(frozen=True)
class ObjectDecision:
    """Effective access to one document, with the provenance of the answer."""

    may_read: bool
    may_write: bool
    level: str | None
    origin: str
    #: True when the answer came from cart-level inheritance under a policy that has since
    #: been superseded — the flag the UI renders as "legacy access". Never inferred from
    #: current cart state; see `resolve` and the toggle-flip note there.
    grandfathered: bool = False

    def audit(self, cart: str, source_hash: int, seat: str | None) -> str:
        verb = "read" if self.may_read else "REFUSED"
        who = seat or "anonymous"
        lvl = f" as {self.level}" if self.level else ""
        legacy = " (grandfathered)" if self.grandfathered else ""
        return (f"object {verb}{lvl}{legacy}: cart={cart!r} doc={source_hash} "
                f"seat={who} origin={self.origin}")


def _rank(level: str | None) -> int:
    """Order levels so 'at least editor' is expressible. Owner outranks everything."""
    if level == OWNER:
        return len(ACCESS_LEVELS) + 1
    order = ["viewer", "commenter", "editor"]
    return order.index(level) + 1 if level in order else 0


# --------------------------------------------------------------------------- resolution

def resolve(*, cart_level: str | None, inherit: bool,
            exposure: str | None = None,
            exception: str | None = None,
            perms_byte: int | None = None,
            is_owner: bool = False) -> ObjectDecision:
    """Pure decision for one (seat, document). No I/O — the lookup is the caller's problem.

    Split from the query for the same reason `cart_access.decide` is: every permission bug
    this project has shipped was in the rules, not in the SQL, and rules are only cheap to
    test when they have no database attached.

    Args:
        cart_level: the seat's effective level on the CART, from `cart_access`. `None`
            means no grant governs — a legacy/unregistered cart or unconfigured
            enforcement — and is passed through rather than treated as a denial, per
            Andy's 08-03 legacy rule.
        inherit: the cart's inheritance toggle. True = new documents take the cart's
            level; False = a document is owner-only until someone grants it. Andy,
            08-09: the admin picks this per cart, because a company handbook and a
            patient-records cart want opposite defaults.
        exposure: a level required by the DOCUMENT, applying to every seat. This is the
            field membot already stores as `world_perms` and has never checked.
        exception: an owner-authored per-seat override — a level, or `DENY`.
        perms_byte: the passage's PERM bits. Object sensitivity, and it binds EVERYONE
            including the owner; see the note below.
        is_owner: whether this seat owns the cart.
    """
    # 1. A per-seat exception is the most specific statement anyone has made, so nothing
    #    below it can override it -- including ownership. An owner who writes "not Betty"
    #    and is then overruled by Betty's own grant has not written a rule, only a wish.
    if exception is not None:
        if exception == DENY:
            return _apply_perms(ObjectDecision(False, False, None, ORIGIN_EXCEPTION),
                                perms_byte)
        return _apply_perms(
            ObjectDecision(True, can_here(exception, "write"), exception, ORIGIN_EXCEPTION),
            perms_byte)

    # 2. The owner is not locked out of their own cart by an exposure level. Exposure
    #    exists to restrict OTHER seats; an owner who could hide a document from
    #    themselves by raising it would have built a trap, not a control. (PERM_R still
    #    binds them -- that bit is about the passage, not about the person.)
    if is_owner:
        return _apply_perms(ObjectDecision(True, True, OWNER, ORIGIN_OWNER), perms_byte)

    # 3. An exposure level set ON the document governs every remaining seat.
    if exposure is not None:
        if _rank(cart_level) >= _rank(exposure) and _rank(exposure) > 0:
            return _apply_perms(
                ObjectDecision(True, can_here(cart_level, "write"), cart_level,
                               ORIGIN_EXPLICIT),
                perms_byte)
        return _apply_perms(ObjectDecision(False, False, None, ORIGIN_EXPLICIT), perms_byte)

    # 4. No statement about this document at all. The cart's toggle decides what silence
    #    means -- which is the whole point of the toggle.
    if inherit:
        # Readable if a grant governs (all three levels include `read`), and ALSO readable
        # when none does -- `None` is the legacy/unenforced cart, which Andy's rule makes
        # readable by anyone. Only an empty or unrecognised level falls through to refusal.
        inherited_read = cart_level is None or _rank(cart_level) > 0 or cart_level == OWNER
        return _apply_perms(
            ObjectDecision(inherited_read, _writable(cart_level), cart_level,
                           ORIGIN_INHERITED),
            perms_byte)

    # Explicit mode: silence is refusal. Note this is NOT the same as DENY -- origin
    # distinguishes "nobody has said yes" from "somebody said no", and only one of those is
    # fixed by asking the owner.
    return _apply_perms(ObjectDecision(False, False, None, ORIGIN_WITHHELD), perms_byte)


def _writable(cart_level: str | None) -> bool:
    """Inherited write follows the cart grant, and defers when no grant governs.

    `cart_level is None` means legacy/unenforced, where Andy's rule puts writability on the
    cart's read-only flag rather than here: "if they are editable then anyone can write
    them and if they are read-only then no one can write them." Returning True would
    override that flag; returning False would freeze every legacy cart. Deferring is what
    `cart_guard.require_cart_write` already does one layer up, so this matches it.
    """
    if cart_level is None:
        return True
    return can_here(cart_level, "write")


def _apply_perms(decision: ObjectDecision, perms_byte: int | None) -> ObjectDecision:
    """Intersect the person's access with the passage's own bits.

    PERM_R BINDS EVERYONE, INCLUDING THE OWNER, and that is deliberate rather than an
    oversight: the bit says "this passage is not for reading", not "this passage is not for
    you". An append-only record nobody may re-read is a real thing to want, and making the
    owner an exception would mean the bit could never express it. Per-person-per-passage is
    what `exception` is for.

    `None` and legacy carts read as permissive -- anything stricter hides every cart we
    have ever built, all of which carry a uniform perms_byte of 3.
    """
    if perms_byte is None:
        return decision
    from .cartridge_io import PERM_R, PERM_W
    may_read = decision.may_read and bool(perms_byte & PERM_R)
    may_write = decision.may_write and bool(perms_byte & PERM_W)
    if may_read == decision.may_read and may_write == decision.may_write:
        return decision
    return ObjectDecision(may_read, may_write, decision.level, decision.origin,
                          decision.grandfathered)


def mark_grandfathered(decision: ObjectDecision, cart_now_explicit: bool) -> ObjectDecision:
    """Flag an inherited answer on a cart whose toggle has since been flipped to explicit.

    Andy, 08-09: "You just have to indicate on the grandfathered files in big bold red
    letters somehow that there's an access issue because it's a legacy file."

    This is what makes flipping the toggle survivable. Retroactively withdrawing inherited
    access would black out a cart in one click; silently keeping it would leave nobody able
    to tell which documents were ever deliberately shared. Grandfather, and SAY SO — the
    exposure becomes a punch list the admin can work down instead of a big-bang migration.

    The caller passes `cart_now_explicit` rather than this module reading current cart
    state, because the fact being reported is about the PAST: how this access was arrived
    at. Deriving it from today's toggle is the event-time-vs-ingest-time bug in another
    costume.
    """
    if not (cart_now_explicit and decision.origin == ORIGIN_INHERITED
            and decision.may_read):
        return decision
    return ObjectDecision(decision.may_read, decision.may_write, decision.level,
                          decision.origin, grandfathered=True)
