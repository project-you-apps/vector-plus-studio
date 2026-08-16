"""What the server tells an anonymous caller must not describe our filesystem.

Andy's standing rule for the public box (2026-08-15): *"Users not logged in should not be
able to gain any knowledge of file structures or have any directory access of any sort."*

THE SAME DISCLOSURE ESCAPED THROUGH THREE DOORS, and the 08-10/08-12 sweep only closed one:

  1. `/api/status`            -> `mounted_path`            CLOSED 08-12
  2. `/api/cartridges`        -> `filename`                open until 08-15 (8 of 10 carts)
  3. mount response `message` -> `from <cart_dir>`         open until 08-15

Door 2 was inconsistent -- the `.pkl` branch had always sent a basename, the `.cart.npz`
branch sent the absolute path -- which is why it read as intentional and survived review.

⚠ AND CLOSING DOOR 2 BROKE MOUNTING, which is the part worth remembering. The client mounts
by whatever `filename` holds, and `_mount_plan` only dispatched by suffix for ABSOLUTE paths,
so a bare `redwood-finance.cart.npz` fell through to `_mount_pkl` -> `pickle.load` -> a 500.
Sending a basename without fixing resolution would have replaced a disclosure with an
outage. Both halves are pinned here, together, for that reason.
"""

import os

import pytest

from api import cartridge_io, main


# -- door 2: the cart list ---------------------------------------------------------------

def test_the_cart_list_never_sends_a_path():
    """Dogfood against the real cartridge dirs; a fixture would not have caught this."""
    leaky = [c for c in cartridge_io.list_cartridges()
             if isinstance(c.get("filename"), str)
             and ("/" in c["filename"] or "\\" in c["filename"])]
    assert not leaky, (
        "these carts publish a server path to every anonymous caller:\n  "
        + "\n  ".join(f"{c['name']} -> {c['filename']}" for c in leaky))


def test_the_list_still_carries_the_absolute_path_internally():
    """`path` is for us and is not serialised. Removing it would break report resolution."""
    carts = cartridge_io.list_cartridges()
    if not carts:
        pytest.skip("no carts on this machine")
    assert any(os.path.isabs(c.get("path", "")) for c in carts)


def test_the_response_model_cannot_serialise_the_internal_path():
    """The safety net under the field above: even if `path` grew a leak, it cannot escape."""
    from api.models import CartridgeInfo

    assert "path" not in CartridgeInfo.model_fields, (
        "CartridgeInfo now serialises `path` -- the absolute path is public again")


# -- the half that makes door 2 safe to close --------------------------------------------

def test_a_bare_npz_name_resolves_instead_of_reaching_the_pickle_loader():
    """⚠ THE OUTAGE THIS PREVENTS. Bare `.cart.npz` used to fall through to `_mount_pkl`."""
    npz = next((c for c in cartridge_io.list_cartridges()
                if c["filename"].endswith(".cart.npz")), None)
    if npz is None:
        pytest.skip("no .cart.npz cart on this machine")

    plan = main._mount_plan(npz["filename"])
    assert plan is not None, (
        f"{npz['filename']} does not resolve, so the mount route falls through to "
        "_mount_pkl and hands an npz to pickle.load -- a 500 for every npz cart")
    helper, _ = plan
    assert helper is main._mount_membot_npz, f"npz dispatched to {helper.__name__}"


def test_a_bare_pkl_name_is_deliberately_left_to_the_by_name_path():
    """⚠ DO NOT 'FIX' THIS. `_mount_pkl` finds companions with `find_companion_file` (all
    cart dirs PLUS DATA_DIR); `_mount_pkl_by_path` looks only beside the file. Routing bare
    .pkl names through the plan would silently drop the brain for `wiki_nomic_100k.pkl`,
    whose .pkl is in sample_data and whose _brain.npy is in cartridges/."""
    assert main._mount_plan("wiki_nomic_100k.pkl") is None


@pytest.mark.parametrize("hostile", [
    "../../../etc/passwd", "/etc/passwd", "..\\..\\windows\\system32\\config\\sam",
    "cartridges/../../../etc/shadow",
])
def test_resolution_does_not_rehabilitate_a_path(hostile):
    """The new resolution step only accepts separator-free names, so nothing path-shaped
    gets a second chance at reaching the filesystem through it."""
    assert main._mount_plan(hostile) is None or os.path.isabs(hostile), (
        f"{hostile!r} was resolved by the bare-name path")


# -- door 3: the mount message -----------------------------------------------------------

def test_the_mount_message_names_no_directory_on_a_public_host(monkeypatch):
    monkeypatch.setattr(main, "PUBLIC_HOST", True)
    assert main._mount_origin("/opt/vector-plus-studio/cartridges") == []


def test_the_local_studio_keeps_the_directory(monkeypatch):
    """It answers 'why is this the wrong version' when a cart exists in two dirs."""
    monkeypatch.setattr(main, "PUBLIC_HOST", False)
    assert main._mount_origin("/carts") == ["from /carts"]


def test_no_mount_helper_hardcodes_the_directory_any_more():
    """All three helpers said `from {cart_dir}` inline. One helper now, or the next one
    added quietly reopens the door."""
    import inspect

    src = inspect.getsource(main)
    body = src.split("def _mount_origin", 1)[1].split("\n\n\n", 1)[1]
    assert 'f"from {cart_dir}"' not in body, (
        "a mount helper builds the directory fragment inline again")
