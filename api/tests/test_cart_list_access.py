"""The cart list shows what you may OPEN, not what is on the disk.

THE FINDING (Andy, on the droplet, 2026-08-15): *"when not signed in the redwood company
carts are all visible. Luckily they can't be mounted however."*

The mount gate was working. It was also the only thing working: `/api/cartridges` took no
user parameter at all, so it enumerated every cart on the box to every caller. A signed-out
visitor could not open `redwood-finance`, but learned it exists -- a fact about someone's
business rather than about our demo, and exactly the "no knowledge of file structures" line.

⚠ THE INVARIANT IS THAT ONE AUTHORITY DECIDES BOTH. Listing and opening now both go through
`cart_guard.resolve_named`. Two separate rules would drift, and the drift is silent in the
dangerous direction: a cart that mount hides but the list still names.
"""

import inspect

import pytest

from api import cart_access, cart_guard, main


def test_the_route_consults_the_guard_at_all():
    """⚠ THE REGRESSION, in its simplest form: the route used to take no user."""
    sig = inspect.signature(main.get_cartridges)
    assert "user" in sig.parameters, "the cart list takes no caller identity again"
    assert "request" in sig.parameters


def test_listing_and_opening_share_one_authority():
    """Not a second rule that happens to agree today."""
    src = inspect.getsource(main.get_cartridges)
    assert "cart_guard.resolve_named" in src, (
        "the list decides visibility by some other means than the mount gate")


def test_a_refused_cart_is_not_named(monkeypatch):
    """The anonymous droplet case, end to end through the route's decision path."""
    listed = [{"name": "redwood-finance", "filename": "redwood-finance.cart.npz",
               "size_mb": 1.0, "has_brain": False, "has_signatures": False,
               "has_manifest": False},
              {"name": "attention-is-all-you-need", "filename": "attention.cart.npz",
               "size_mb": 0.1, "has_brain": False, "has_signatures": False,
               "has_manifest": False}]
    monkeypatch.setattr(main, "_list_cartridges", lambda: listed)

    def _decide(request, user, name):
        if name == "redwood-finance":          # registered, no grant for this caller
            return cart_access.decide(registered=True, owner_id="someone-else",
                                      grant_level=None, seat=None)
        return cart_access.decide(registered=False, owner_id=None,   # legacy demo cart
                                  grant_level=None, seat=None)

    monkeypatch.setattr(cart_guard, "resolve_named", _decide)
    names = _names_from(main.get_cartridges)

    assert "attention-is-all-you-need" in names, "the public demo cart vanished"
    assert "redwood-finance" not in names, (
        "a cart this caller cannot open is still named to them")


@pytest.fixture
def one_private_cart(monkeypatch):
    monkeypatch.setattr(main, "_list_cartridges", lambda: [
        {"name": "private-cart", "filename": "private-cart.cart.npz", "size_mb": 1.0,
         "has_brain": False, "has_signatures": False, "has_manifest": False}])


def test_a_public_host_hides_a_cart_it_could_not_check(monkeypatch, one_private_cart):
    """Fail CLOSED where it matters. A lookup outage must not become an amnesty on every
    name at once, which is the same direction cart_access.lookup_failed() already chose."""
    monkeypatch.setattr(main, "PUBLIC_HOST", True)
    monkeypatch.setattr(cart_guard, "resolve_named",
                        lambda request, user, name: cart_access.lookup_failed())
    assert _names_from(main.get_cartridges) == []


def test_a_local_studio_keeps_a_cart_it_could_not_check(monkeypatch, one_private_cart):
    """⚠ "COULD NOT CHECK" IS NOT "MAY NOT SEE". Andy, 2026-08-15, an hour after the filter
    shipped: Susie could not see her own carts in the dropdown on her own machine, while
    "Open from my computer" still found them. Hiding an owner's carts because Supabase
    blinked is a worse outcome than briefly naming a cart on a single-user box."""
    monkeypatch.setattr(main, "PUBLIC_HOST", False)
    monkeypatch.setattr(cart_guard, "resolve_named",
                        lambda request, user, name: cart_access.lookup_failed())
    assert _names_from(main.get_cartridges) == ["private-cart"]


def test_a_definitive_refusal_hides_the_cart_even_locally(monkeypatch, one_private_cart):
    """The distinction is COULD-NOT-CHECK vs MAY-NOT-SEE, not public vs local. An actual
    no-grant answer hides the cart everywhere, or the local studio becomes a way to
    enumerate carts you were told you cannot open."""
    monkeypatch.setattr(main, "PUBLIC_HOST", False)
    monkeypatch.setattr(cart_guard, "resolve_named",
                        lambda request, user, name: cart_access.decide(
                            registered=True, owner_id="someone-else",
                            grant_level=None, seat="susie"))
    assert _names_from(main.get_cartridges) == []


def test_a_raising_guard_still_hides_on_a_public_host(monkeypatch, one_private_cart):
    """An exception is even less of an answer than lookup_failed."""
    monkeypatch.setattr(main, "PUBLIC_HOST", True)

    def _boom(request, user, name):
        raise RuntimeError("supabase unreachable")

    monkeypatch.setattr(cart_guard, "resolve_named", _boom)
    assert _names_from(main.get_cartridges) == []


def test_the_local_studio_still_sees_everything(monkeypatch):
    """With no enforcement configured, `decide` allows -- the single-user studio is
    unaffected, which is what makes this safe to turn on everywhere."""
    monkeypatch.setattr(main, "_list_cartridges", lambda: [
        {"name": f"cart-{i}", "filename": f"cart-{i}.cart.npz", "size_mb": 1.0,
         "has_brain": False, "has_signatures": False, "has_manifest": False}
        for i in range(3)])
    monkeypatch.setattr(cart_guard, "resolve_named",
                        lambda request, user, name: cart_access.decide(
                            registered=False, owner_id=None, grant_level=None,
                            seat=None, enforced=False))
    assert len(_names_from(main.get_cartridges)) == 3


def test_an_externally_mounted_cart_is_not_named_by_its_path(monkeypatch):
    """⚠ THE DOOR THE EARLIER FIX MISSED. The scanned branch was changed to a basename and
    this one still published `engine.mounted_path` -- so mounting a sandbox upload put an
    absolute server path straight back into the list."""
    src = inspect.getsource(main.get_cartridges)
    assert "filename=engine.mounted_path" not in src, (
        "the external-mount branch publishes an absolute server path again")
    assert "os.path.basename(engine.mounted_path)" in src


def _names_from(route) -> list:
    """Drive the route and return the cart names it published."""
    import asyncio

    class _Req:
        headers: dict = {}
        cookies: dict = {}

    resp = asyncio.run(route(_Req(), None))
    return [c.name for c in resp.cartridges]
