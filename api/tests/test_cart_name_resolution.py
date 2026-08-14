"""A cart you can mount is a cart you can name in a header.

THE BUG (Andy, 2026-08-14). Tabs 404'd on every request -- `/api/status` included -- while
other tabs on the same server got 200s. The mixed result was the tell: the refusal came from
`bind_caller_cart`, which turns a loader `FileNotFoundError` into a 404 BEFORE the endpoint
runs, so a tab holding a stale-looking cart id fails uniformly and a tab holding a resolvable
one is fine.

The cart ids were not stale. `.pkl` carts were simply unresolvable by name: the candidate
list tried `name` and `name.cart.npz` and nothing else. `wiki_nomic_100k` could be MOUNTED
(the Open dialog sends an absolute path) but never REBOUND (the header sends the name), so
the tab that mounted it 404'd from its very next poll.

⚠ AND THE ACCESS GUARD HAD THE SAME OMISSION, which is why this is one fix and not two. A
`.pkl` cart registered in `user_carts` resolved to no candidate, came back "unregistered",
and was treated as a legacy cart -- the fail-open path. Sharing the list closes the 404 and
that gap together, and these tests exist to keep them shared.
"""

import pytest

from api import cart_guard, cartridge_io, main
from api.cartridge_io import name_candidates


# -- the shape of the list ---------------------------------------------------------------

def test_a_bare_name_offers_both_on_disk_spellings():
    assert name_candidates("finance") == ["finance", "finance.cart.npz", "finance.pkl"]


def test_pkl_is_offered_at_all():
    """THE REGRESSION. Without this, every .pkl cart 404s the moment a tab names it."""
    assert "wiki_nomic_100k.pkl" in name_candidates("wiki_nomic_100k")


def test_the_newest_format_wins_when_a_cart_exists_in_both():
    """A cart saved in both spellings must resolve the way mounting by name always has."""
    candidates = name_candidates("wiki_nomic_10k")
    assert candidates.index("wiki_nomic_10k.cart.npz") < candidates.index("wiki_nomic_10k.pkl")


@pytest.mark.parametrize("already_suffixed", ["finance.cart.npz", "finance.pkl", "finance.npz"])
def test_a_name_that_already_carries_a_suffix_is_left_alone(already_suffixed):
    """Not a bare name, so appending to it would invent files that cannot exist."""
    assert name_candidates(already_suffixed) == [already_suffixed]


# -- the two callers must not drift ------------------------------------------------------

def test_the_guard_and_the_loader_use_the_SAME_list():
    """⚠ THE INVARIANT. Different orders means access-checked against one file, served
    another -- a bypass rather than a bug. Sharing one function is what makes that
    unrepresentable, so this asserts they both actually call it."""
    import inspect

    guard_src = inspect.getsource(cart_guard.resolve_named)
    loader_src = inspect.getsource(main.load_cart_fields)

    assert "name_candidates(" in guard_src, "the guard built its own candidate list again"
    assert "name_candidates(" in loader_src, "the loader built its own candidate list again"
    for src, who in ((guard_src, "guard"), (loader_src, "loader")):
        assert ".cart.npz\"" not in src and ".cart.npz'" not in src, (
            f"the {who} hardcodes a suffix -- that is how the two copies drifted before")


# -- against what is actually on disk ----------------------------------------------------

def test_every_cart_on_disk_can_be_rebound_from_a_header():
    """Dogfood, not fixtures: the real cartridge directories.

    Every cart the Open dialog will list must survive being named in `X-VPS-Cart`, because
    that is the round trip a browser tab makes on every single request after it mounts.
    Signature sidecars are companions, not carts, and the dialog does not offer them.
    """
    import os

    unreachable = []
    for directory in cartridge_io.get_cartridge_dirs():
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith((".cart.npz", ".pkl")):
                continue
            if filename.endswith("_signatures.npz"):
                continue
            stem = filename.replace(".cart.npz", "").replace(".pkl", "")
            if not any(cartridge_io.find_cartridge_path(c) for c in name_candidates(stem)):
                unreachable.append(f"{stem} (on disk as {filename})")

    assert not unreachable, (
        "these carts can be mounted but not rebound, so a tab that opens one 404s on every "
        "later request including /api/status:\n  " + "\n  ".join(unreachable))
