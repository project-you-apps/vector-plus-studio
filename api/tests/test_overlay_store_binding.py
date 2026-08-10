"""The overlay store must follow the mounted cart, not the first one mounted.

Found 2026-08-10 while inventorying what lives inside a cart versus beside it. Neither bug
had ever fired, because no `*.overlay.json` file exists anywhere on disk — per-seat
attention shipped on 08-03 and has never written a byte. That is the whole reason both
survived: a mechanism nobody exercises cannot report that it is wrong.

Bug 1 — the store was cached on `DATA_DIR` alone, so the FIRST cart mounted in the process
created it and froze `parent_cart` for the life of the server. Mount finance, then company,
and every seat's attention on company is filed under finance. Same single-global family as
`engine.mounted_path`.

Bug 2 — `parent_cart` was built as `f"{mounted_name}.pkl"` unconditionally, so a `.cart.npz`
cart recorded a parent filename that does not exist. Harmless only until something joins on
it, and the cart-sync work is explicit that joins go on the source ref.
"""

import os

import pytest

from api import main


@pytest.fixture
def stores(monkeypatch):
    """Isolate the module-level store cache so tests cannot leak into each other."""
    monkeypatch.setattr(main, "_OVERLAY_STORES", {}, raising=False)
    return main._OVERLAY_STORES


class _Store:
    """Stand-in for membot's OverlayStore — we assert on binding, not on flushing."""

    def __init__(self, directory, parent_cart="", **kw):
        self.dir = directory
        self.parent_cart = parent_cart


@pytest.fixture
def fake_modules(monkeypatch):
    class _Mod:
        OverlayStore = _Store
    monkeypatch.setattr(main, "_overlay_modules", lambda: (None, _Mod(), None))


def _mount(monkeypatch, name, path):
    monkeypatch.setattr(main.engine, "mounted_name", name, raising=False)
    monkeypatch.setattr(main.engine, "mounted_path", path, raising=False)


def test_switching_carts_gives_a_store_bound_to_the_new_cart(monkeypatch, stores,
                                                             fake_modules):
    """The bug, stated as behaviour: finance first, then company."""
    _mount(monkeypatch, "redwood-finance", os.path.join("c", "redwood-finance.cart.npz"))
    first = main._overlay_store_for_mounted()

    _mount(monkeypatch, "redwood-company", os.path.join("c", "redwood-company.cart.npz"))
    second = main._overlay_store_for_mounted()

    assert first is not second, "second cart reused the first cart's store"
    assert second.parent_cart == "redwood-company.cart.npz"


def test_parent_cart_is_the_real_filename_not_a_guessed_pkl(monkeypatch, stores,
                                                            fake_modules):
    """An npz cart must not record a `.pkl` parent that has never existed on disk."""
    _mount(monkeypatch, "redwood-finance", os.path.join("c", "redwood-finance.cart.npz"))
    store = main._overlay_store_for_mounted()
    assert store.parent_cart == "redwood-finance.cart.npz"
    assert not store.parent_cart.endswith(".pkl")


def test_remounting_the_same_cart_reuses_its_store(monkeypatch, stores, fake_modules):
    """The caching is still worth having — this is what keeps attention accumulating
    across requests instead of resetting on every search."""
    _mount(monkeypatch, "redwood-finance", os.path.join("c", "redwood-finance.cart.npz"))
    a = main._overlay_store_for_mounted()
    b = main._overlay_store_for_mounted()
    assert a is b


def test_a_legacy_pkl_cart_still_gets_a_sensible_parent(monkeypatch, stores, fake_modules):
    """Older carts really are `.pkl`; the fix must not invert the bug for them."""
    _mount(monkeypatch, "nomic_dataset_10k", os.path.join("c", "nomic_dataset_10k.pkl"))
    assert main._overlay_store_for_mounted().parent_cart == "nomic_dataset_10k.pkl"


def test_no_mounted_path_falls_back_rather_than_binding_to_empty(monkeypatch, stores,
                                                                 fake_modules):
    """`mounted_path` can be None for a catalog mount. An empty parent would file every
    seat's attention under "", silently merging carts — worse than the bug being fixed."""
    _mount(monkeypatch, "some-cart", None)
    store = main._overlay_store_for_mounted()
    assert store.parent_cart == "some-cart.pkl"


def test_nothing_mounted_yields_no_store(monkeypatch, stores, fake_modules):
    _mount(monkeypatch, None, None)
    assert main._overlay_store_for_mounted() is None
