"""Mount-time access rules.

`decide()` is pure, so every branch is reachable without a database. That is the point of
the split: the permission bugs we have actually shipped were in the rules (an empty
sidecar reading as writable, 2026-08-01), never in the query.
"""

import pytest

from api.cart_access import (
    DECISION_ANONYMOUS,
    DECISION_GRANTED,
    DECISION_NO_GRANT,
    DECISION_OWNER,
    DECISION_UNENFORCED,
    DECISION_UNREGISTERED,
    decide,
    enforcement_available,
)

OWNER_SEAT = "11111111-1111-1111-1111-111111111111"
OTHER_SEAT = "22222222-2222-2222-2222-222222222222"


# ----------------------------------------------------------------- legacy carts

def test_unregistered_cart_is_readable_by_anyone():
    """Andy's legacy rule, 2026-08-03: no owner row means readable, no migration."""
    d = decide(registered=False, owner_id=None, grant_level=None, seat=OTHER_SEAT)
    assert d.allowed
    assert d.reason == DECISION_UNREGISTERED


def test_unregistered_cart_is_readable_anonymously():
    d = decide(registered=False, owner_id=None, grant_level=None, seat=None)
    assert d.allowed


def test_unregistered_cart_claims_no_write_grant():
    """`may_write` answers 'did a grant authorise this', not 'is the cart writable'.

    A legacy cart's writability is the existing read-only flag's business. If this
    returned True the flag would be bypassed; if the CALLER treated False as read-only,
    every legacy cart would silently freeze -- which is the failure Andy's rule exists to
    avoid. So it reports False and the mount path leaves the flag alone.
    """
    d = decide(registered=False, owner_id=None, grant_level=None, seat=OWNER_SEAT)
    assert d.may_write is False
    assert d.level is None


# ----------------------------------------------------------------- owned carts

def test_owner_may_mount_and_write():
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=OWNER_SEAT)
    assert d.allowed
    assert d.reason == DECISION_OWNER
    assert d.level == "owner"
    assert d.may_write


def test_ungranted_seat_is_refused():
    """The hole this module closes: registered cart, no grant, previously mounted fine."""
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=OTHER_SEAT)
    assert not d.allowed
    assert d.reason == DECISION_NO_GRANT


def test_anonymous_is_refused_on_a_registered_cart():
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=None)
    assert not d.allowed
    assert d.reason == DECISION_ANONYMOUS


@pytest.mark.parametrize("level,writable", [
    ("viewer", False),
    ("commenter", False),
    ("editor", True),
])
def test_grant_levels_carry_their_capabilities(level, writable):
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=level, seat=OTHER_SEAT)
    assert d.allowed
    assert d.reason == DECISION_GRANTED
    assert d.level == level
    assert d.may_write is writable


def test_commenter_may_mount_but_not_write():
    """Commenter is the level proposals ride on; mounting is exactly what it needs."""
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level="commenter",
               seat=OTHER_SEAT)
    assert d.allowed and not d.may_write


# ----------------------------------------------------------------- unenforced

def test_unenforced_allows_but_does_not_claim_a_grant():
    """No auth configured: allowed, and the reason must NOT read as 'granted'.

    A local desktop user owns the disk the cart sits on, so refusing the mount would
    protect nothing while breaking the single-user studio. What matters is that the audit
    line says so, because 'allowed because nobody is restricted' and 'allowed because this
    seat was granted' are the difference between a demo claim that is true and one that is
    not.
    """
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=None,
               enforced=False)
    assert d.allowed
    assert d.enforced is False
    assert d.reason == DECISION_UNENFORCED
    assert d.level is None
    assert d.may_write is False


def test_enforcement_available_follows_the_environment(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert enforcement_available() is False

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    assert enforcement_available() is True


def test_enforcement_needs_both_halves(monkeypatch):
    """A URL with no key cannot authenticate anyone, so it must not enable enforcement."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert enforcement_available() is False


# ----------------------------------------------------------------- audit lines

def test_audit_line_names_seat_cart_and_reason():
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=OTHER_SEAT)
    line = d.audit("payroll.pkl", OTHER_SEAT)
    assert "REFUSED" in line
    assert "payroll.pkl" in line
    assert OTHER_SEAT in line
    assert DECISION_NO_GRANT in line


def test_audit_line_marks_anonymous_without_inventing_a_seat():
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=None)
    assert "anonymous" in d.audit("payroll.pkl", None)
