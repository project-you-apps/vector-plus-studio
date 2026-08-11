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

# Every Supabase variable enforcement_available() consults. Cleared before each test so no
# case depends on ambient environment.
_ENFORCEMENT_VARS = ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_PUBLISHABLE_KEY")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Start every test from a known-empty environment.

    `api/main.py` calls `load_dotenv()` AT IMPORT, so any test in the suite that imports it
    loads the developer's real .env into os.environ for the whole session. On 2026-08-05,
    once .env gained a live SUPABASE_PUBLISHABLE_KEY, `test_enforcement_needs_both_halves`
    began failing in the suite while passing alone: it cleared SUPABASE_ANON_KEY but the
    publishable key was still there from the dotenv load.

    Clearing per test rather than fixing the import, because loading .env at import is the
    right behaviour for the app -- it is only wrong to let a test inherit it.
    """
    for var in _ENFORCEMENT_VARS:
        monkeypatch.delenv(var, raising=False)


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


# ------------------------------------------------------- the client library is config too

def test_enforcement_off_when_supabase_library_missing(monkeypatch):
    """Credentials present, library absent -> unavailable, NOT fail-closed.

    Regression for 2026-08-04: the studio had SUPABASE_URL and a key but no `supabase`
    package (it was never in requirements.txt). Every lookup raised, the gate fell back to
    fail-closed, and EVERY MOUNT returned 503 -- a missing dependency presenting as a
    permissions outage.

    A missing library is a PERMANENT fact: enforcement can never work, so refusing forever
    protects nothing. A reachable library with an unreachable database is TRANSIENT and is
    exactly when fail-closed matters. Different facts, opposite answers.
    """
    import builtins
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    real_import = builtins.__import__

    def no_supabase(name, *a, **kw):
        if name == "supabase":
            raise ImportError("No module named 'supabase'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_supabase)
    assert enforcement_available() is False


def test_publishable_key_also_enables_enforcement(monkeypatch):
    """The migration renames anon -> publishable. Gating on the old name only would switch
    enforcement OFF at the moment of a rename -- a security control disabled by a rename."""
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "pub-key")
    assert enforcement_available() is True


def test_anonymous_reason_is_anonymous_not_no_grant():
    """An unauthenticated caller must be told to SIGN IN, not to ask the owner.

    Regression for 2026-08-05: VPS_SEAT was set in the shell, `_seat_for()` fell back to it
    when no token was present, and the gate therefore saw a seat where there was no
    authenticated user -- refusing a registered cart as 'no-grant'. The user was told to
    request access from the owner when all they needed to do was log in.

    Access was never bypassed (cart_access_for() reads auth.uid() and correctly denied), but
    an environment variable must not shape an access decision even when it cannot grant one.
    The gate now takes identity from the verified token only.
    """
    d = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=None)
    assert d.reason == DECISION_ANONYMOUS
    assert not d.allowed

    # And with a seat present it is a genuine no-grant, which is a different message.
    d2 = decide(registered=True, owner_id=OWNER_SEAT, grant_level=None, seat=OTHER_SEAT)
    assert d2.reason == DECISION_NO_GRANT


# ------------------------------------------------- the read-only / public-host flag split

@pytest.mark.parametrize("env,writes_blocked,fs_blocked", [
    # Existing droplet config: only VPS_READ_ONLY. Both must stay blocked -- the split
    # must not weaken a deployment that has not been reconfigured.
    ({"VPS_READ_ONLY": "1"},                        True,  True),
    # The target droplet config: filesystem stays shut, writes governed per-cart.
    ({"VPS_PUBLIC_HOST": "1"},                      False, True),
    # Local single-user: neither.
    ({},                                            False, False),
    # Explicitly opting a private host OUT of the filesystem block while keeping writes off.
    ({"VPS_READ_ONLY": "1", "VPS_PUBLIC_HOST": "0"}, True, False),
])
def test_read_only_and_public_host_are_independent(monkeypatch, env, writes_blocked, fs_blocked):
    """VPS_READ_ONLY governs WRITES. VPS_PUBLIC_HOST governs FILESYSTEM EXPOSURE.

    They were one flag until 2026-08-06, which meant turning writes back on -- the whole
    point of per-cart access control -- would silently restore a stranger's ability to walk
    /opt and /etc via /api/cartbuilder/carts?path=. A control disabled as a side effect of
    fixing an unrelated feature is the worst kind.
    """
    import importlib
    for var in ("VPS_READ_ONLY", "VPS_PUBLIC_HOST"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from api import main as _main
    try:
        importlib.reload(_main)
        assert _main.READ_ONLY_MODE is writes_blocked
        assert _main.PUBLIC_HOST is fs_blocked
    finally:
        # RELOAD BACK, or this test poisons every test that runs after it.
        #
        # monkeypatch restores the ENVIRONMENT on teardown but has no idea a module was
        # reloaded from it, so `main.READ_ONLY_MODE` kept whatever the last parametrised
        # case set -- True. Found 2026-08-10 when new per-caller lock tests passed alone and
        # failed in the suite; the flag they read had been frozen True by this test hours
        # earlier. Same shape as the `.env` leak fixed on 08-05: module state derived from
        # the environment at import time is not restored by restoring the environment.
        for var in ("VPS_READ_ONLY", "VPS_PUBLIC_HOST"):
            monkeypatch.delenv(var, raising=False)
        importlib.reload(_main)
