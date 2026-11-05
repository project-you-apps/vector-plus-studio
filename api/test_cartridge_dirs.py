"""Does VPS_EXTRA_CARTRIDGE_DIRS actually widen the cart search path -- and refuse to do
anything else?

⛔ WHY A TEST FOR A SIX-LINE FUNCTION. The feature it unblocks (membot's carts and the
Studio's carts being ONE file on disk, so journal/notes sidecars sit beside the cart the
Studio mounted) fails SILENTLY in both directions. A dir that is read but empty reports
"no carts" exactly like a dir that was never read, and a dir created by a typo reports the
same. Neither raises. So the control cases below matter more than the happy path.

Run from the repo root:  python -m pytest api/test_cartridge_dirs.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from api import cartridge_io


@pytest.fixture()
def extra(tmp_path, monkeypatch):
    d = tmp_path / "membot-carts"
    d.mkdir()
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, str(d))
    return d


def test_unset_env_changes_nothing(monkeypatch):
    monkeypatch.delenv(cartridge_io._EXTRA_DIRS_ENV, raising=False)
    dirs = cartridge_io.get_cartridge_dirs()
    assert cartridge_io.DATA_DIR in dirs
    assert all("membot" not in d.lower() for d in dirs), dirs


def test_an_extra_dir_is_scanned(extra):
    dirs = cartridge_io.get_cartridge_dirs()
    assert str(extra) in dirs, dirs
    # ⭐ DATA_DIR stays first: it is the Studio's own folder and must win name collisions.
    assert dirs[0] == cartridge_io.DATA_DIR


def test_several_dirs_split_on_the_platform_separator(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, f"{a}{os.pathsep}{b}")
    dirs = cartridge_io.get_cartridge_dirs()
    assert str(a) in dirs and str(b) in dirs, dirs


def test_a_nonexistent_dir_is_skipped_and_NOT_created(tmp_path, monkeypatch):
    """⛔⛔ THE CONTROL THAT MATTERS. list_cartridges() calls os.makedirs(exist_ok=True) on
    everything this returns, so handing back an unchecked path turns a typo into a new
    empty folder -- which then reports zero carts as if that were an answer."""
    ghost = tmp_path / "typo-dir"
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, str(ghost))
    dirs = cartridge_io.get_cartridge_dirs()
    assert str(ghost) not in dirs, "a path that does not exist was returned as a cart dir"
    assert not ghost.exists(), "get_cartridge_dirs created a directory"


def test_empty_and_whitespace_entries_are_ignored(tmp_path, monkeypatch):
    a = tmp_path / "real"
    a.mkdir()
    # Trailing separator and a stray space are what a hand-edited .bat file actually
    # produces; an empty entry must not become '' (which os.path.isdir rejects) or '.'.
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, f"{a}{os.pathsep} {os.pathsep}")
    dirs = cartridge_io.get_cartridge_dirs()
    assert str(a) in dirs
    assert "" not in dirs and "." not in dirs, dirs


def test_quoted_path_is_accepted(tmp_path, monkeypatch):
    """A Windows path with spaces gets quoted by anyone editing a .bat by hand, and the
    quotes arrive as part of the value."""
    a = tmp_path / "has space"
    a.mkdir()
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, f'"{a}"')
    assert str(a) in cartridge_io.get_cartridge_dirs()


def test_the_same_place_spelled_twice_appears_once(monkeypatch):
    """⭐ Otherwise list_cartridges() scans it twice. It dedupes FILENAMES via `seen`, so a
    duplicated dir would not double the listing -- it would just do the work twice and,
    worse, make DATA_DIR's authority over a name depend on scan order."""
    doubled = cartridge_io.DATA_DIR + os.sep
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, doubled)
    dirs = cartridge_io.get_cartridge_dirs()
    canonical = [os.path.normcase(os.path.realpath(d)) for d in dirs]
    assert len(canonical) == len(set(canonical)), dirs


def test_the_env_is_read_per_call_not_cached_at_import(tmp_path, monkeypatch):
    """⛔ The server is long-running. A value frozen at import means changing it requires
    hunting down and restarting the process -- the same 'running thing is older than the
    config' trap that has already cost two diagnoses."""
    before = cartridge_io.get_cartridge_dirs()
    d = tmp_path / "late"
    d.mkdir()
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, str(d))
    after = cartridge_io.get_cartridge_dirs()
    assert str(d) not in before
    assert str(d) in after, "get_cartridge_dirs cached the environment"


def test_a_cart_in_an_extra_dir_is_actually_LISTED(tmp_path, monkeypatch):
    """⭐⭐ EXISTENCE IS NOT OPERATION. Every test above checks what the resolver RETURNS.
    This one checks that the thing downstream of it -- the listing the dropdown is built
    from -- really picks a cart up out of an extra directory."""
    d = tmp_path / "membot-carts"
    d.mkdir()
    cart = d / "extradir-probe.cart.npz"
    cart.write_bytes(b"\0" * 4096)          # over MIN_CART_BYTES so it is not a stub
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, str(d))

    names = [c.get("name") for c in cartridge_io.list_cartridges()]
    assert "extradir-probe" in names, f"cart in an extra dir never reached the listing: {names}"


def test_a_stub_in_an_extra_dir_is_still_filtered(tmp_path, monkeypatch):
    """The control for the test above: if it listed everything, that test would pass for
    the wrong reason."""
    d = tmp_path / "stubs"
    d.mkdir()
    (d / "tiny-stub.cart.npz").write_bytes(b"\0" * 16)
    monkeypatch.setenv(cartridge_io._EXTRA_DIRS_ENV, str(d))
    names = [c.get("name") for c in cartridge_io.list_cartridges()]
    assert "tiny-stub" not in names, "a sub-1KB placeholder was listed as a cart"
