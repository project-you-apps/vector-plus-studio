"""Tests for user profiles, seat identity, and effective cart access.

    PYTHONIOENCODING=utf-8 python -m pytest vector-plus-studio-repo/api/tests/test_profiles.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from api import profiles as p

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_004 = REPO_ROOT / "db" / "004_cart_grants.sql"


UUID = "8f2c1a7b-4d10-4c3e-9a55-0b6d2e7f1a33"


# ------------------------------------------------------------------ seat identity

def test_seat_id_is_the_supabase_sub():
    assert p.seat_id({"sub": UUID, "email": "andy@example.com"}) == UUID


@pytest.mark.parametrize("user", [None, {}, {"sub": ""}, {"sub": "   "},
                                  {"sub": 7}, {"email": "a@b.c"}, "not a dict"])
def test_anonymous_has_no_seat(user):
    """None, not a placeholder -- a shared fallback seat would merge strangers'
    attention into one overlay."""
    assert p.seat_id(user) is None


def test_seat_id_is_not_the_email():
    """Emails change and are re-assignable; using one as identity silently re-points a
    person's whole attention history."""
    assert p.seat_id({"sub": UUID, "email": "andy@example.com"}) != "andy@example.com"


def test_overlay_name_is_path_safe():
    assert p.overlay_name(UUID) == UUID.replace("-", "-")
    assert "/" not in p.overlay_name("../../etc/passwd")
    assert ".." not in p.overlay_name("../../etc/passwd").replace("etcpasswd", "")
    assert re.fullmatch(r"[A-Za-z0-9\-_]+", p.overlay_name(UUID))


@pytest.mark.parametrize("bad", ["", "   ", "///", None, 7])
def test_overlay_name_refuses_unusable_seats(bad):
    with pytest.raises(ValueError):
        p.overlay_name(bad)


# ------------------------------------------------------------------ display names

def test_display_name_prefers_display_then_full_then_username():
    assert p.display_name_for({"display_name": "Andy", "full_name": "A G",
                               "username": "ag"}) == "Andy"
    assert p.display_name_for({"full_name": "A G", "username": "ag"}) == "A G"
    assert p.display_name_for({"username": "ag"}) == "ag"


def test_display_name_falls_back_to_email_local_part():
    assert p.display_name_for({}, {"email": "andy@example.com"}) == "andy"


def test_display_name_never_raises_on_a_missing_profile():
    """A brand-new user has no profile row yet; that is normal, not an error."""
    assert p.display_name_for(None) == "Unknown"
    assert p.display_name_for("nonsense") == "Unknown"


def test_blank_fields_are_skipped_not_returned():
    assert p.display_name_for({"display_name": "   ", "full_name": "A G"}) == "A G"


# ------------------------------------------------------------------ access levels

def test_sql_and_python_vocabularies_agree():
    """Two spellings of one vocabulary is how a cart grants write in one layer and
    denies it in another."""
    sql = SQL_004.read_text(encoding="utf-8")
    constraint = re.search(r"access_level in \(([^)]*)\)", sql).group(1)
    in_sql = tuple(re.findall(r"'([a-z]+)'", constraint))
    assert in_sql == p.GRANTABLE_LEVELS


def test_owner_is_not_a_grantable_level():
    """If 'owner' were grantable, a compromised grant row would be an ownership transfer."""
    assert "owner" not in p.GRANTABLE_LEVELS
    sql = SQL_004.read_text(encoding="utf-8")
    constraint = re.search(r"access_level in \(([^)]*)\)", sql).group(1)
    assert "owner" not in constraint


@pytest.mark.parametrize("value,expected", [
    ("owner", "owner"), ("editor", "editor"), ("VIEWER", "viewer"),
    ("  commenter  ", "commenter"),
    (None, None), ("", None), ("admin", None), ("rwx", None), (7, None), ([], None),
])
def test_normalize_effective(value, expected):
    assert p.normalize_effective(value) == expected


def test_no_grant_means_denied_not_defaulted():
    """Absence of a grant is a denial. Two fail-OPEN paths were found on 2026-08-01."""
    assert p.can_here(None, "read") is False
    assert p.capabilities_here(None) == frozenset()
    assert p.describe_cart_access(None)["access"] is None


# ------------------------------------------------------------------ capabilities

def test_viewer_reads_but_cannot_write():
    assert p.can_here("viewer", "read") is True
    assert p.can_here("viewer", "write") is False
    assert p.can_here("viewer", "annotate") is False


def test_commenter_annotates_but_cannot_write():
    assert p.can_here("commenter", "annotate") is True
    assert p.can_here("commenter", "write") is False


def test_editor_writes_but_cannot_share_or_sign():
    """The whole ownership model rests on this staying true."""
    assert p.can_here("editor", "write") is True
    assert p.can_here("editor", "share") is False
    assert p.can_here("editor", "sign") is False


def test_owner_can_share_and_sign():
    assert p.can_here("owner", "share") is True
    assert p.can_here("owner", "sign") is True
    assert p.is_owner("owner") is True


def test_share_and_sign_are_unreachable_from_every_grantable_level():
    for level in p.GRANTABLE_LEVELS:
        assert p.can_here(level, "share") is False, level
        assert p.can_here(level, "sign") is False, level


def test_capabilities_here_matches_can_here():
    for level in ("owner",) + p.GRANTABLE_LEVELS:
        for capability in ("read", "annotate", "write", "share", "sign"):
            assert (capability in p.capabilities_here(level)) == p.can_here(level, capability)


def test_unknown_capability_is_denied_at_every_level():
    for level in ("owner",) + p.GRANTABLE_LEVELS:
        assert p.can_here(level, "sudo") is False


# ------------------------------------------------------------------ response shapes

def test_describe_marks_read_only_correctly():
    assert p.describe_cart_access("viewer")["read_only"] is True
    assert p.describe_cart_access("commenter")["read_only"] is True
    assert p.describe_cart_access("editor")["read_only"] is False
    assert p.describe_cart_access("owner")["read_only"] is False


def test_describe_of_no_access_is_not_read_only_it_is_nothing():
    """`read_only` must not be True for a cart you cannot open at all -- that would
    render as 'you can view this' in any UI that checks the flag."""
    d = p.describe_cart_access(None)
    assert d["read_only"] is False and d["capabilities"] == []


# ------------------------------------------------------------------ cart lists

def _row(**kw):
    base = {"id": "cart-1", "cart_filename": "brain.pkl", "display_name": "Brain",
            "size_bytes": 1024, "pattern_count": 12, "effective_access": "viewer"}
    base.update(kw)
    return base


def test_visible_carts_shapes_rows_for_the_ui():
    out = p.visible_carts([_row()])
    assert out[0]["cart_id"] == "cart-1" and out[0]["access"] == "viewer"
    assert out[0]["is_owner"] is False and out[0]["read_only"] is True


def test_inaccessible_rows_are_dropped_not_greyed_out():
    out = p.visible_carts([_row(effective_access=None), _row(effective_access="bogus"),
                           _row(effective_access="editor")])
    assert [r["access"] for r in out] == ["editor"]


def test_visible_carts_survives_garbage():
    assert p.visible_carts(None) == []
    assert p.visible_carts([None, "row", 7]) == []


def test_display_name_falls_back_to_filename():
    out = p.visible_carts([_row(display_name=None)])
    assert out[0]["display_name"] == "brain.pkl"


def test_owner_row_reports_share():
    out = p.visible_carts([_row(effective_access="owner")])
    assert out[0]["is_owner"] is True and out[0]["can_share"] is True
