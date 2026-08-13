"""A public host must not open carts by filesystem path.

FOUND LIVE ON THE DROPLET, 2026-08-12. `/api/cartridges/mount` did:

    if os.path.isabs(filename) and os.path.exists(filename):
        ... open the file ...

with no PUBLIC_HOST check, so an unauthenticated stranger could name `/etc/passwd`,
`/opt/vector-plus-studio/.env`, or `../../../etc/passwd` and the server would open it and
hand it to `pickle.load` / `np.load(..., allow_pickle=True)`. Verified against the running
public server: all three were opened (`exists=True`) and failed only at unpickling. Nothing
was disclosed, but the only reason is that those files are not valid pickles -- and
unpickling caller-chosen bytes is an arbitrary-code-execution primitive the moment anything
pickle-shaped reaches the disk.

The decision to forbid this dates to 2026-06-02 and was implemented in the FRONTEND, by
hiding the Open Cartridge button when `read_only_mode`. Hiding a button is not a control.

Cart ACCESS control was never bypassable this way -- a registered cart named by absolute
path still 403s -- so this is about reaching the filesystem, not about reaching other
people's carts. Both matter; only one of them was broken.
"""

import os

import pytest
from fastapi import HTTPException

from api import main


# Refusals must not depend on the file existing -- see test_refusal_is_by_shape_not_existence.
OUTSIDE_PATHS = [
    "/etc/passwd",
    "/opt/vector-plus-studio/.env",
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "/tmp/definitely-not-a-cart-9f3a.cart.npz",
]


@pytest.fixture
def public(monkeypatch):
    """PUBLIC_HOST on, without reloading the module.

    monkeypatch reverts automatically. A previous suite reloaded `api.main` under an env var
    and never reloaded it back, poisoning every test that ran after it (fixed 2026-08-10) --
    so this fixture deliberately does not do that.
    """
    monkeypatch.setattr(main, "PUBLIC_HOST", True)


@pytest.fixture
def private(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_HOST", False)


@pytest.mark.parametrize("path", OUTSIDE_PATHS)
def test_public_host_refuses_paths_outside_the_cartridge_dirs(public, path):
    with pytest.raises(HTTPException) as exc:
        main._refuse_path_shaped_filename(path)
    assert exc.value.status_code == 403


@pytest.mark.parametrize("path", OUTSIDE_PATHS)
def test_local_studio_still_opens_any_path(private, path):
    """The Open dialog is the whole point of running locally. This must NOT refuse."""
    main._refuse_path_shaped_filename(path)


def test_paths_inside_a_cartridge_dir_are_allowed(public):
    """The upload sandbox lives under DATA_DIR, so hosted upload-then-mount keeps working."""
    from api.cartridge_io import get_cartridge_dirs

    root = get_cartridge_dirs()[0]
    main._refuse_path_shaped_filename(os.path.join(root, "gutenberg-poetry.cart.npz"))
    main._refuse_path_shaped_filename(os.path.join(root, "_session_uploads", "x.cart.npz"))


def test_bare_cart_names_are_untouched(public):
    """Normal mounting is by name and must be unaffected on every host."""
    for name in ("gutenberg-poetry", "redwood-finance.cart.npz", "wiki_nomic_100k.pkl",
                 "some-cart (brain only)"):
        main._refuse_path_shaped_filename(name)


def test_refusal_is_by_shape_not_existence(public):
    """Refusing only paths that EXIST would answer 'is there a file here?' for any path.

    That is a filesystem oracle -- a smaller version of the same leak -- so a path outside
    the cart dirs is refused identically whether or not it is there.
    """
    missing = "/nonexistent-root-8c1d/definitely/not/here.cart.npz"
    assert not os.path.exists(missing)
    with pytest.raises(HTTPException) as exc:
        main._refuse_path_shaped_filename(missing)
    assert exc.value.status_code == 403


def test_traversal_out_of_a_cartridge_dir_is_refused(public):
    """`<cartdir>/../../etc/passwd` is inside the cart dir only before normalisation."""
    from api.cartridge_io import get_cartridge_dirs

    root = get_cartridge_dirs()[0]
    with pytest.raises(HTTPException) as exc:
        main._refuse_path_shaped_filename(os.path.join(root, "..", "..", "etc", "passwd"))
    assert exc.value.status_code == 403


def test_mount_actually_calls_the_guard_before_touching_disk():
    """The helper is worthless if the route does not call it, and call ORDER is the control.

    Same discipline as `test_mount_routes_gate_themselves`: a mechanism that exists and is
    never invoked reads as done in a diff and does nothing in production -- which is the
    category this entire bug came from.
    """
    import inspect

    src = inspect.getsource(main.mount_cartridge)
    assert "_refuse_path_shaped_filename(" in src, (
        "mount_cartridge never calls _refuse_path_shaped_filename")

    # The disk work moved into `_mount_plan` on 2026-08-13 (shared with the pool loader), so
    # this anchors on the CALL to it rather than on the isabs/exists line it used to contain.
    # It FAILED rather than passing vacuously when the code moved, because `.index()` raises
    # on a missing anchor -- worth preserving. A boolean `in` check would have gone quiet.
    guard_at = src.index("_refuse_path_shaped_filename(")
    disk_at = src.index("_mount_plan(")
    assert guard_at < disk_at, (
        "the guard runs AFTER the filesystem dispatch; it must refuse before any disk access")


def test_the_pool_loader_cannot_be_handed_a_path():
    """`load_cart_fields` is reachable from a request HEADER, so it takes a name, not a path.

    The mount route is protected by `_refuse_path_shaped_filename`. The pool loader is a
    second door to the same loaders, and a second door needs its own lock -- it resolves
    through `find_cartridge_path`, which only searches whitelisted cartridge directories.
    """
    import inspect

    src = inspect.getsource(main.load_cart_fields)
    assert "find_cartridge_path(" in src, (
        "load_cart_fields does not resolve through find_cartridge_path; a caller-supplied "
        "cart id could reach an arbitrary path")
