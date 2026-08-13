"""The view key must never become an access identity, and two tabs must not share a view.

Both halves matter. On 2026-08-05 a VPS_SEAT env fallback made an anonymous caller look like
a seat, turning "please sign in" into "ask the owner for access" -- an identifier that existed
for bookkeeping reached an authorization path. This module introduces another such
identifier, so the separation is asserted rather than intended.
"""

import pytest

from api.request_cart import (
    CART_HEADER,
    SESSION_HEADER,
    access_seat,
    requested_cart,
    view_key,
)


class FakeRequest:
    """Minimal stand-in: headers, query params, client host."""

    def __init__(self, headers=None, query=None, host="1.2.3.4"):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = query or {}
        self.client = type("C", (), {"host": host})()


# -- access_seat: verified token only ----------------------------------------

def test_access_seat_comes_from_the_token():
    assert access_seat({"sub": "uuid-susie"}) == "uuid-susie"


@pytest.mark.parametrize("user", [None, {}, {"sub": ""}, {"sub": "   "}, {"sub": 7}, "susie"])
def test_anonymous_and_malformed_yield_no_access_seat(user):
    """None is the point: an access decision must be able to tell 'nobody' from 'someone'."""
    assert access_seat(user) is None


def test_a_session_header_cannot_produce_an_access_seat():
    """THE security assertion. A forged header must not create identity."""
    req = FakeRequest({SESSION_HEADER: "uuid-betty"})
    assert access_seat(None) is None
    assert "uuid-betty" not in (access_seat(None) or "")
    # It may key a VIEW, and that is all it may do.
    assert view_key(req, None) == "anon:uuid-betty"


# -- view_key: who is looking -------------------------------------------------

def test_signed_in_callers_key_on_their_seat():
    req = FakeRequest({SESSION_HEADER: "tab-1"})
    assert view_key(req, {"sub": "uuid-susie"}) == "seat:uuid-susie"


def test_two_anonymous_tabs_get_two_views():
    """Without this, every anonymous visitor to the demo shares one cart -- Susie and Betty
    again, minus the names."""
    a = view_key(FakeRequest({SESSION_HEADER: "tab-a"}), None)
    b = view_key(FakeRequest({SESSION_HEADER: "tab-b"}), None)
    assert a != b


def test_a_caller_sending_nothing_still_gets_a_key():
    """An un-migrated screen must keep working, not 500."""
    assert view_key(FakeRequest(), None).startswith("anon-noheader:")


def test_session_ids_are_sanitised_and_capped():
    """Junk must not reach cache keys or logs; this is hygiene, not a control."""
    req = FakeRequest({SESSION_HEADER: "../../etc/passwd\n<script>" + "x" * 500})
    key = view_key(req, None)
    assert key.startswith("anon:")
    assert "/" not in key and "<" not in key and "\n" not in key
    assert len(key) < 100


def test_an_empty_session_header_falls_back_rather_than_keying_on_nothing():
    assert view_key(FakeRequest({SESSION_HEADER: "   "}), None).startswith("anon-noheader:")


# -- requested_cart -----------------------------------------------------------

def test_no_cart_named_is_not_an_error():
    """None means 'whatever the server has' -- the un-migrated path, deliberately supported."""
    assert requested_cart(FakeRequest()) is None


def test_the_header_names_the_cart():
    assert requested_cart(FakeRequest({CART_HEADER: "redwood-finance.cart.npz"})) \
        == "redwood-finance.cart.npz"


def test_the_query_param_works_for_pasteable_links():
    assert requested_cart(FakeRequest(query={"cart": "gutenberg-poetry"})) == "gutenberg-poetry"


def test_the_header_wins_over_the_query_param():
    req = FakeRequest({CART_HEADER: "from-header"}, {"cart": "from-query"})
    assert requested_cart(req) == "from-header"


@pytest.mark.parametrize("raw,expected", [
    ("/etc/passwd", "passwd"),
    ("../../../etc/passwd", "passwd"),
    ("/opt/vector-plus-studio/cartridges/redwood-finance.cart.npz", "redwood-finance.cart.npz"),
    ("C:\\carts\\finance.cart.npz", "finance.cart.npz"),
])
def test_a_path_shaped_cart_id_is_reduced_to_a_name(raw, expected):
    """A cart id is a NAME. Separators reaching the loader is how 2026-08-12 happened.

    This is defence in depth, not the control -- `_refuse_path_shaped_filename` on the mount
    path is. Two independent places refusing the same thing is the point.
    """
    got = requested_cart(FakeRequest({CART_HEADER: raw}))
    assert got == expected
    assert "/" not in got and "\\" not in got and ".." not in got


def test_whitespace_only_is_treated_as_absent():
    assert requested_cart(FakeRequest({CART_HEADER: "   "})) is None
