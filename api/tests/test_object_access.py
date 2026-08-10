"""The rules for per-document, per-seat access. No database, on purpose.

`cart_access` has a twin of this file for the same reason: every permission bug this
project has shipped lived in the rules, not in the query. Rules are cheap to test
exhaustively only while they have nothing attached to them.

Each test names the case in Redwood terms, because "Betty may read `company` but not the
three documents about her own compensation" is the requirement, and a test that only says
`assert not d.may_read` will be reread in six months by someone who cannot tell whether it
encodes a decision or an accident.
"""

import pytest

from api import object_access as OA
from api.object_access import DENY
from api.cartridge_io import PERM_R, PERM_W

RW = PERM_R | PERM_W


# --------------------------------------------------------------------------
# Inheritance ON — the company-handbook cart
# --------------------------------------------------------------------------

def test_inherited_access_follows_the_cart_grant():
    """The default that has to stay cheap: a viewer on the cart reads its documents."""
    d = OA.resolve(cart_level="viewer", inherit=True, perms_byte=RW)
    assert d.may_read and not d.may_write
    assert d.origin == OA.ORIGIN_INHERITED


def test_inherited_editor_may_write():
    d = OA.resolve(cart_level="editor", inherit=True, perms_byte=RW)
    assert d.may_read and d.may_write


def test_commenter_reads_but_never_writes_the_cart():
    """The distinction the whole ladder turns on — commenter writes the sidecar, not the
    cart. Canon §7.1.1."""
    d = OA.resolve(cart_level="commenter", inherit=True, perms_byte=RW)
    assert d.may_read and not d.may_write


def test_legacy_cart_with_no_grant_still_works(monkeypatch):
    """`cart_level=None` is a legacy/unenforced cart, NOT a denial.

    Andy's 08-03 rule puts writability on the cart's read-only flag in that case. Refusing
    here would freeze every cart we have ever built and the entire single-user studio —
    the exact regression `test_legacy_cart_writes_are_deferred_not_refused` guards one
    layer up.
    """
    d = OA.resolve(cart_level=None, inherit=True, perms_byte=RW)
    assert d.may_read and d.may_write


# --------------------------------------------------------------------------
# Inheritance OFF — the patient-records cart
# --------------------------------------------------------------------------

def test_explicit_mode_withholds_a_document_nobody_has_granted():
    """Andy's PII case: a new patient file must not be readable because it landed in a
    cart people already have access to."""
    d = OA.resolve(cart_level="viewer", inherit=False, perms_byte=RW)
    assert not d.may_read
    assert d.origin == OA.ORIGIN_WITHHELD


def test_withheld_is_distinguishable_from_denied():
    """"Nobody has said yes" and "somebody said no" are different facts, and only one of
    them is fixed by asking the owner. Collapsing them is the fail-open shape
    `cart_access.lookup` exists to prevent one layer up."""
    withheld = OA.resolve(cart_level="viewer", inherit=False, perms_byte=RW)
    denied = OA.resolve(cart_level="viewer", inherit=True, exception=DENY, perms_byte=RW)
    assert not withheld.may_read and not denied.may_read
    assert withheld.origin != denied.origin


def test_explicit_mode_still_serves_a_document_that_was_granted():
    """Fail-closed must not mean unusable — the grant is what turns it back on."""
    d = OA.resolve(cart_level="viewer", inherit=False, exception="viewer", perms_byte=RW)
    assert d.may_read
    assert d.origin == OA.ORIGIN_EXCEPTION


def test_the_owner_is_not_locked_out_by_explicit_mode():
    """Otherwise flipping the toggle would lock the admin out of their own cart, and the
    first thing they would do is flip it back."""
    d = OA.resolve(cart_level=None, inherit=False, is_owner=True, perms_byte=RW)
    assert d.may_read and d.may_write
    assert d.origin == OA.ORIGIN_OWNER


# --------------------------------------------------------------------------
# The case this module exists for
# --------------------------------------------------------------------------

def test_betty_reads_the_cart_but_not_the_document_about_her_pay():
    """THE REQUIREMENT, in one test.

    Betty holds viewer on `company`. `pay-equalization-roadmap` carries an exception
    denying her. Everything else in the cart is unaffected — which is the property that
    makes this different from moving the document to another cart.
    """
    ordinary = OA.resolve(cart_level="viewer", inherit=True, perms_byte=RW)
    her_pay = OA.resolve(cart_level="viewer", inherit=True, exception=DENY, perms_byte=RW)
    assert ordinary.may_read
    assert not her_pay.may_read
    assert her_pay.origin == OA.ORIGIN_EXCEPTION


def test_an_exception_outranks_ownership():
    """An owner who writes "not Betty" and is then overruled by Betty's own grant has
    written a wish, not a rule."""
    d = OA.resolve(cart_level="editor", inherit=True, exception=DENY,
                   is_owner=True, perms_byte=RW)
    assert not d.may_read


def test_an_exception_can_raise_as_well_as_lower():
    """Carve-in, not just carve-out: one contractor gets edit on one document."""
    d = OA.resolve(cart_level="viewer", inherit=True, exception="editor", perms_byte=RW)
    assert d.may_read and d.may_write
    assert d.level == "editor"


# --------------------------------------------------------------------------
# Object exposure — the field membot stores and has never checked
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cart_level,exposure,expect", [
    ("viewer",    "viewer", True),
    ("commenter", "viewer", True),
    ("editor",    "viewer", True),
    ("viewer",    "editor", False),   # doc requires editor; a viewer is below it
    ("commenter", "editor", False),
    ("editor",    "editor", True),
])
def test_exposure_requires_at_least_that_level(cart_level, exposure, expect):
    """"This document is for editors" has to mean editors AND ABOVE, or every raise in the
    ladder would silently revoke access from the people most entitled to it."""
    d = OA.resolve(cart_level=cart_level, inherit=True, exposure=exposure, perms_byte=RW)
    assert d.may_read is expect


def test_exposure_does_not_lock_out_the_owner():
    d = OA.resolve(cart_level=None, inherit=True, exposure="editor",
                   is_owner=True, perms_byte=RW)
    assert d.may_read


# --------------------------------------------------------------------------
# PERM_R — object sensitivity, intersected not substituted
# --------------------------------------------------------------------------

def test_perm_r_hides_a_passage_from_a_seat_that_may_otherwise_read_it():
    """`access.py`: "Person-capability ∩ object-sensitivity. Checking only this module
    leaks non-readable passages; checking only the bit ignores the caller.\""""
    d = OA.resolve(cart_level="editor", inherit=True, perms_byte=PERM_W)   # no PERM_R
    assert not d.may_read


def test_perm_r_binds_the_owner_too():
    """Deliberate. The bit says "this passage is not for reading", not "not for you" — an
    append-only record nobody may re-read is a real thing to want, and an owner exemption
    would make it inexpressible. Per-person carve-outs are what `exception` is for."""
    d = OA.resolve(cart_level=None, inherit=True, is_owner=True, perms_byte=PERM_W)
    assert not d.may_read


def test_perm_w_off_blocks_writes_but_keeps_reads():
    d = OA.resolve(cart_level="editor", inherit=True, perms_byte=PERM_R)
    assert d.may_read and not d.may_write


def test_absent_perms_is_permissive():
    """Every cart we have ever built carries a uniform perms_byte; anything stricter here
    hides all of them at once."""
    d = OA.resolve(cart_level="viewer", inherit=True, perms_byte=None)
    assert d.may_read


# --------------------------------------------------------------------------
# Grandfathering — Andy's ruling, 2026-08-09
# --------------------------------------------------------------------------

def test_inherited_access_survives_a_toggle_flip_but_is_flagged():
    """Retroactive withdrawal would black out a 4,000-document cart in one click.

    Grandfather, and say so — that turns explicit-release from a big-bang migration into a
    punch list the admin works down at their own pace.
    """
    d = OA.resolve(cart_level="viewer", inherit=True, perms_byte=RW)
    flagged = OA.mark_grandfathered(d, cart_now_explicit=True)
    assert flagged.may_read
    assert flagged.grandfathered is True


def test_a_deliberate_grant_is_never_flagged_as_legacy():
    """The badge has to mean something. If explicit grants also carried it, the admin
    could not tell which documents still need a decision — which is the entire use."""
    d = OA.resolve(cart_level="viewer", inherit=True, exception="viewer", perms_byte=RW)
    assert OA.mark_grandfathered(d, cart_now_explicit=True).grandfathered is False


def test_nothing_is_flagged_while_the_cart_still_inherits():
    d = OA.resolve(cart_level="viewer", inherit=True, perms_byte=RW)
    assert OA.mark_grandfathered(d, cart_now_explicit=False).grandfathered is False


def test_a_refusal_is_never_flagged_grandfathered():
    """A red badge on something the seat cannot see would be a UI element describing
    nothing, and would leak the document's existence."""
    d = OA.resolve(cart_level="viewer", inherit=False, perms_byte=RW)
    assert OA.mark_grandfathered(d, cart_now_explicit=True).grandfathered is False


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def test_audit_line_names_the_document_and_the_origin():
    """An audit line that cannot distinguish inherited from granted cannot answer the only
    question anyone asks after an incident: was this deliberate?"""
    d = OA.resolve(cart_level="viewer", inherit=True, perms_byte=RW)
    line = OA.mark_grandfathered(d, cart_now_explicit=True).audit(
        "redwood-company.cart.npz", 10857319, "betty")
    assert "betty" in line and "10857319" in line
    assert OA.ORIGIN_INHERITED in line and "grandfathered" in line
