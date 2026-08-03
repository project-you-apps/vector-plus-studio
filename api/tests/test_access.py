"""Conformance tests for api/access.py against CARTRIDGE-FORMAT.md §7.1 / §7.1.1.

Canon is normative. Every assertion here cites the clause it enforces, so that a canon
change that this file does not reflect shows up as a failing test rather than as drift —
which is the failure mode canon §0 was written about ("documentation with no detector").

Run: pytest api/tests/test_access.py -v
"""

import pytest

from api import access as A


# --------------------------------------------------------------------------
# §7.1.1 — the capability table is normative, verbatim
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level,expected", [
    ("viewer",    {"read"}),
    ("commenter", {"read", "annotate"}),
    ("editor",    {"read", "annotate", "write"}),
])
def test_capability_table_matches_canon(level, expected):
    assert set(A.ACCESS_LEVELS[level]) == expected


def test_exactly_three_levels():
    """Canon: 'Hence three values, not four.' Owner is not a sidecar value."""
    assert set(A.ACCESS_LEVELS) == {"viewer", "commenter", "editor"}
    assert "owner" not in A.ACCESS_LEVELS


@pytest.mark.parametrize("capability", ["share", "sign"])
def test_no_access_level_grants_owner_capabilities(capability):
    """§7.1: ownership is proven by holding the signing key, not by a string in a file.

    This is the test that makes ownership unforgeable. If it ever fails, a sidecar edit
    can grant `share` or `sign` and the whole ownership model is decorative.
    """
    assert not any(A.can(level, capability) for level in A.ACCESS_LEVELS)
    assert A.can_as_owner(capability)


# --------------------------------------------------------------------------
# §7.1 — dual-read migration
# --------------------------------------------------------------------------

def test_access_level_preferred_over_legacy_default():
    """'Readers prefer access_level; fall back to legacy default when it is absent.'"""
    assert A.resolve_access_level({"access_level": "commenter", "default": "rwx"}) == "commenter"


@pytest.mark.parametrize("legacy,expected", [
    ("r", "viewer"),
    ("rw", "editor"),
    ("rwx", "editor"),
])
def test_legacy_mapping(legacy, expected):
    """Canon: r→viewer, rw→editor, rwx→editor.

    rwx must NOT promote beyond editor — 'x was never semantically distinct for carts,
    so it must not silently promote.'
    """
    assert A.resolve_access_level({"default": legacy}) == expected


def test_absent_sidecar_keeps_legacy_behaviour():
    """No sidecar at all means a pre-Step-2a cart. Those are writable today.

    Deliberately NOT fail-closed: canon's fail-closed rule covers a sidecar that EXISTS
    and cannot be interpreted. Failing absent-sidecar closed would break every existing
    local cart to defend against a threat that is not present on a single-user box.
    """
    assert A.resolve_access_level(None) == "editor"


# --------------------------------------------------------------------------
# §7.1 — 'fail closed, never open'
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sidecar", [
    {"access_level": "admin"},          # unrecognized level
    {"access_level": "OWNER"},          # owner is not grantable via sidecar, any casing
    {"default": "rwxq"},                # uninterpretable legacy value
    {"description": "no perms here"},   # sidecar exists, declares nothing
    {},                                 # empty sidecar
    "editor",                           # not even a dict
    ["editor"],
    42,
])
def test_unrecognized_input_fails_closed(sidecar):
    assert A.resolve_access_level(sidecar) == A.FAIL_CLOSED_LEVEL == "viewer"


def test_role_key_is_not_an_access_level():
    """`role` means CART TYPE here (identity/episodic/semantic/federated) and is exposed
    publicly as multi_search(role_filter=…). Canon §7.1 (2026-08-01) renamed the sidecar
    key to `access_level` precisely so these two never merge. Reading `role` as an access
    level would let a cart-type string grant write access.
    """
    assert A.resolve_access_level({"role": "editor"}) == "viewer"
    assert A.resolve_access_level({"role": "federated"}) == "viewer"


# --------------------------------------------------------------------------
# can() — the enforcement primitive
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level,capability,expected", [
    ("viewer",    "read",     True),
    ("viewer",    "annotate", False),
    ("viewer",    "write",    False),
    ("commenter", "annotate", True),
    ("commenter", "write",    False),
    ("editor",    "write",    True),
])
def test_can(level, capability, expected):
    assert A.can(level, capability) is expected


@pytest.mark.parametrize("level", [None, "", "wizard", "Editor ", 7])
def test_can_denies_unknown_levels(level):
    """Including 'Editor ' — normalization must not become a way to smuggle a level in."""
    result = A.can(level, "read")
    assert result is (str(level).strip().lower() in A.ACCESS_LEVELS)


def test_typo_in_capability_denies_rather_than_grants():
    """A misspelled capability must fail closed. `can(x, "wrtie")` granting would be the
    worst possible direction for a typo to fail in."""
    assert A.can("editor", "wrtie") is False
    assert A.can("editor", "") is False


# --------------------------------------------------------------------------
# Denials must be explicable
# --------------------------------------------------------------------------

def test_denial_message_names_the_rule():
    """'You cannot write' without 'here is which rule stopped you' is undebuggable from
    the far side of an API call. Same lesson as membot_server._write_blocked."""
    msg = A.describe_denial("viewer", "write")
    assert "editor" in msg and "viewer" in msg

    owner_msg = A.describe_denial("editor", "sign")
    assert "owner" in owner_msg.lower() and "key" in owner_msg.lower()

    unknown_msg = A.describe_denial("wizard", "read")
    assert "not recognized" in unknown_msg


# --------------------------------------------------------------------------
# Integration: the live enforcement path must go through can()
# --------------------------------------------------------------------------

def test_cart_permits_write_delegates_to_access():
    """`cartridge_io.cart_permits_write` is the shipped enforcement point. It must resolve
    through this module rather than re-implementing the rules, or the two drift."""
    from api.cartridge_io import cart_permits_write

    # legacy behaviour preserved
    assert cart_permits_write(None) is True            # no sidecar = pre-Step-2a cart
    assert cart_permits_write({"default": "rw"}) is True
    assert cart_permits_write({"default": "rwx"}) is True
    assert cart_permits_write({"default": "r"}) is False

    # new vocabulary understood
    assert cart_permits_write({"access_level": "editor"}) is True
    assert cart_permits_write({"access_level": "commenter"}) is False
    assert cart_permits_write({"access_level": "viewer"}) is False


@pytest.mark.parametrize("sidecar", [{}, {"description": "x"}, {"version": "1.0"}])
def test_sidecar_without_readable_permission_fails_closed(sidecar):
    """REGRESSION GUARD for two fail-OPEN cases closed on 2026-08-01.

    Before: `{}` short-circuited on `not permissions` and a sidecar lacking `default` fell
    through to `.get("default", "rw")` — both returned WRITABLE. A sidecar that exists and
    declares nothing we can read must fail closed, per canon §7.1.
    """
    from api.cartridge_io import cart_permits_write
    assert cart_permits_write(sidecar) is False
