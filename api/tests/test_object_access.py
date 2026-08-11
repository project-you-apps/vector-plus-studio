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


# --------------------------------------------------------------------------
# Lookup — the boundary between Postgres and the rules
# --------------------------------------------------------------------------

class _RPC:
    """Minimal stand-in for the supabase client's rpc().execute() chain."""

    def __init__(self, data):
        self._data = data

    def rpc(self, name, params):
        self.called = (name, params)
        return self

    def execute(self):
        class _R:
            data = self._data
        return _R()


def test_lookup_converts_jsonb_string_keys_to_ints():
    """`source_hash` is an integer everywhere else in the system; jsonb keys are strings.

    Converting at the boundary means no caller has to remember which side it is on — and a
    dict keyed by "10857319" would silently miss every lookup by 10857319.
    """
    c = _RPC({"inherit": True, "exceptions": {"10857319": "deny"}})
    p = OA.lookup(c, "redwood-finance.cart.npz")
    assert p.exception_for(10857319) == DENY
    assert p.exceptions == {10857319: DENY}


def test_lookup_survives_a_hash_above_int4():
    """Half of all real source hashes exceed 2^31 — verified 150/300 in redwood-company.

    A signed-int assumption anywhere on this path silently drops half the documents.
    """
    big = 4284229723
    c = _RPC({"inherit": False, "exceptions": {str(big): "viewer"}})
    p = OA.lookup(c, "x.cart.npz")
    assert p.exception_for(big) == "viewer"


def test_lookup_of_an_unregistered_cart_governs_nothing():
    c = _RPC({"inherit": True, "exceptions": {}})
    p = OA.lookup(c, "legacy.pkl")
    assert p.inherit and not p.exceptions
    assert p.exception_for(123) is None


def test_a_failed_lookup_is_a_third_state_not_a_permissive_default():
    """Assuming "no exceptions" would show restricted documents to the person they were
    hidden from. Assuming "all restricted" reads as data loss. Neither is acceptable, so
    the caller refuses instead — same 503 `cart_guard` already returns."""
    p = OA.policy_lookup_failed()
    assert p.available is False


def test_a_normal_lookup_is_available():
    assert OA.lookup(_RPC({"inherit": True, "exceptions": {}}), "x").available is True


def test_cart_now_explicit_drives_the_grandfathered_badge():
    """The policy carries the toggle, so the caller never has to re-read cart state to
    decide whether an inherited answer is legacy."""
    inheriting = OA.lookup(_RPC({"inherit": True, "exceptions": {}}), "x")
    flipped = OA.lookup(_RPC({"inherit": False, "exceptions": {}}), "x")
    assert inheriting.cart_now_explicit is False
    assert flipped.cart_now_explicit is True


def test_document_key_reads_v1_v2_source_hash():
    """v1/v2 carts carry the uint32 directly. Every Redwood cart is this shape."""
    assert OA.document_key({"source_hash": 4284229723}) == 4284229723


def test_document_key_derives_the_same_number_from_a_v3_filename():
    """THE BUG THIS FUNCTION EXISTS TO PREVENT.

    v3 replaced the uint32 at offset 18 with a uint16 source_idx plus a reserved uint16, so
    reading those four bytes as a uint32 -- which the first draft did -- produces a garbage
    key for every provenance-v3 cart, silently, with no error anywhere.

    Both formats must land on the SAME number for the same file, so an exception written
    against a v1/v2 cart survives that cart being rebuilt as v3.
    """
    import hashlib
    fn = "dsid_097aa97db50a406fbd83d29e97120b9e__staged-payload-evolution.txt"
    expected = int(hashlib.md5(fn.encode()).hexdigest()[:8], 16)

    v1 = OA.document_key({"source_hash": expected})
    v3 = OA.document_key({"source_idx": 7, "source_path": fn})
    assert v1 == v3 == expected


def test_document_key_matches_the_builder():
    """Pins our copy of the hash to membot's `cartridge_builder._source_hash`.

    Duplicated rather than imported so the studio takes no hard dependency on membot; this
    test is the thing that stops the two drifting.
    """
    import hashlib
    fn = "pay-equalization-roadmap.txt"
    assert OA.document_key({"source_path": fn}) == int(
        hashlib.md5(fn.encode()).hexdigest()[:8], 16)


def test_document_key_is_none_when_the_cart_has_no_provenance():
    """Most carts predate provenance. That must read as "no policy can apply here", never
    as a denial -- the latter would empty every legacy cart the moment this shipped."""
    assert OA.document_key({}) is None
    assert OA.document_key({"perms": {"r": True}}) is None
    assert OA.document_key(None) is None


# --------------------------------------------------------------------------
# Wiring — the part that made PERM_R a bug for six weeks
# --------------------------------------------------------------------------

def test_search_consults_the_document_gate():
    """`/api/search` must ask, not just import.

    PERM_R was stored, returned in every result, and never filtered on. The rule existed;
    nothing called it. This asserts the call site rather than the rule.
    """
    import inspect
    from api import main
    src = inspect.getsource(main.search_endpoint)
    assert "cart_guard.object_policy(" in src
    assert "may_read_document(" in src


def test_search_refuses_when_the_policy_lookup_fails():
    """Not "return everything" and not "return nothing" -- refuse, and say it is a service
    problem. Both silent degradations are wrong in opposite directions."""
    import inspect
    from api import main
    src = inspect.getsource(main.search_endpoint)
    assert "_obj_policy.available" in src
    assert "503" in src


def test_a_cart_without_provenance_is_not_silently_emptied():
    """Most of our carts predate provenance entirely. `document_key` returns None for them
    and the gate must read that as "nothing document-level applies", never as a denial --
    the alternative empties every legacy cart the moment this ships."""
    from api import cart_guard
    policy = OA.ObjectPolicy(inherit=False, exceptions={})     # strictest possible policy
    decision = type("D", (), {"level": "viewer"})()
    assert cart_guard.may_read_document(policy, decision, {"perms": {"r": True}}) is True


def test_the_gate_denies_a_document_the_seat_is_excepted_from():
    """End to end through the wiring function, not just the rule."""
    from api import cart_guard
    import hashlib
    fn = "pay-equalization-roadmap.txt"
    key = int(hashlib.md5(fn.encode()).hexdigest()[:8], 16)
    policy = OA.ObjectPolicy(inherit=True, exceptions={key: DENY})
    decision = type("D", (), {"level": "viewer"})()

    assert cart_guard.may_read_document(policy, decision, {"source_path": fn}) is False
    assert cart_guard.may_read_document(
        policy, decision, {"source_path": "some-other-doc.txt"}) is True
