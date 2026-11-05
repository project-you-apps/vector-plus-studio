"""Can a note actually be WRITTEN through the route -- and refused to the right people?

⭐⭐ THE TWO TESTS THAT MATTER are `test_a_viewer_is_refused` and
`test_a_commenter_may_annotate`. They are the pair, and neither means anything alone:
a route that allowed everyone would pass the commenter test, and a route guarded with
`require_cart_write` would pass the viewer test while refusing the one role the feature
exists for. Only both together show that `annotate` is the gate.

Run from the repo root:  python -m pytest api/test_provenance_write.py -q
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

from api import cart_access, cart_guard, cartridge_io   # noqa: E402
from api import provenance_routes as pr                 # noqa: E402
from api.main import app                                # noqa: E402

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("provenance_link", _HERE / "provenance_link.py")
link = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(link)

CART = "provwrite-probe"
PASSAGE = "Retention: AP documents are held for 7 years from period close."
SEAT_BETTY = "39525aff-f490-5848-8a65-5ab42e7ac08a"


@pytest.fixture()
def cart(monkeypatch):
    if link.modules() is None:
        pytest.skip(f"provenance layer unavailable: {link.availability()['error']}")
    d = Path(tempfile.mkdtemp())
    p = d / f"{CART}.cart.npz"
    p.write_bytes(b"")
    monkeypatch.setattr(cartridge_io, "get_cartridge_dirs", lambda: [str(d)])
    return p


@pytest.fixture()
def client():
    return TestClient(app)


def _as(monkeypatch, level: str | None, seat: str = SEAT_BETTY, enforced: bool = True):
    """Pretend the caller holds `level` on every cart, and is `seat`.

    Patches the ACCESS LOOKUP, not the guard: `require_named_cart_annotate` and
    `may_annotate` both still run, so the capability table is what decides. Stubbing the
    guard itself would test nothing -- it would assert that a function we disabled
    allows things.
    """
    decision = cart_access.MountDecision(
        allowed=True, reason="granted" if level else "unenforced",
        level=level, enforced=enforced)
    monkeypatch.setattr(cart_guard, "resolve_named", lambda *a, **k: decision)
    monkeypatch.setattr(pr, "_seat", lambda user: seat)
    monkeypatch.setattr(pr, "cart_access_enforced", lambda: enforced)


def _post(client, **over):
    body = {"passage": PASSAGE, "kind": "decision", "body": "Because the examiner said so.",
            "scope": "shared"}
    body.update(over)
    return client.post(f"/api/provenance/{CART}/notes", json=body)


# ── the pair ────────────────────────────────────────────────────────────────────────

def test_a_commenter_may_annotate(cart, client, monkeypatch):
    """⭐ commenter == ['annotate', 'read']. This is the role the route exists for."""
    _as(monkeypatch, "commenter")
    r = _post(client)
    assert r.status_code == 201, r.text
    assert r.json()["note"]["by"] == SEAT_BETTY


def test_a_viewer_is_refused(cart, client, monkeypatch):
    """⛔ viewer == ['read']. Read is necessary and NOT sufficient: a viewer can see the
    passage and must not be able to attach an attributed statement to it."""
    _as(monkeypatch, "viewer")
    r = _post(client)
    assert r.status_code == 403, r.text
    assert "viewer" in r.text


def test_an_editor_may_also_annotate(cart, client, monkeypatch):
    _as(monkeypatch, "editor")
    assert _post(client).status_code == 201


def test_the_refused_viewer_wrote_NOTHING(cart, client, monkeypatch):
    """⛔⛔ EXISTENCE IS NOT OPERATION. A 403 proves the response; it does not prove the
    sidecar is untouched. A route that stored first and checked after would pass the
    test above."""
    _as(monkeypatch, "viewer")
    _post(client)
    data = link.modules()[1].load(str(cart))
    assert data.get("notes", []) == [], "a refused note was stored anyway"


# ── identity ────────────────────────────────────────────────────────────────────────

def test_anonymous_is_refused_where_sign_in_EXISTS(cart, client, monkeypatch):
    """An unattributed note is worse than no note -- on a server that HAS sign-in, an
    anonymous writer declined to use it."""
    _as(monkeypatch, "commenter", seat="", enforced=True)
    r = _post(client)
    assert r.status_code == 401, r.text


def test_anonymous_is_allowed_on_an_UNENFORCED_studio_and_marked_claimed(
        cart, client, monkeypatch):
    """⭐ The other half, and the reason this is not just 'refuse anonymous'. A local
    single-user studio has nobody to restrict; refusing would break the product. What
    keeps it honest is the RECORD, not the gate."""
    _as(monkeypatch, None, seat="", enforced=False)
    r = _post(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["identity_source"] == "claimed", body
    assert body["note"]["by"] == pr.LOCAL_SEAT
    assert body["note"]["by"] != "", "an empty author would be invisible to its own writer"


def test_a_verified_seat_is_recorded_as_verified(cart, client, monkeypatch):
    """The control for the test above: if identity_source were hardcoded, both pass."""
    _as(monkeypatch, "commenter")
    assert _post(client).json()["identity_source"] == "verified"


# ── the record ──────────────────────────────────────────────────────────────────────

def test_the_note_is_journalled_with_op_note(cart, client, monkeypatch):
    _as(monkeypatch, "commenter")
    note_id = _post(client).json()["note"]["id"]
    journal = link.modules()[0]
    entries = journal.replay(journal.journal_path_for(str(cart)))
    ops = [e.get("op") for e in entries]
    assert "note" in ops, f"the note was never journalled: {ops}"
    assert any(e.get("payload", {}).get("note_id") == note_id for e in entries)


def test_the_chain_still_verifies_after_a_write(cart, client, monkeypatch):
    _as(monkeypatch, "commenter")
    for i in range(3):
        assert _post(client, body=f"note number {i}").status_code == 201
    journal = link.modules()[0]
    v = journal.verify(journal.journal_path_for(str(cart)))
    assert v.get("ok") is True, v


def test_a_secret_in_the_note_body_is_redacted(cart, client, monkeypatch):
    """⚠ A typed note is a NEW TEXT INGRESS. Our own corpus already holds an API key
    because nothing stood between paste and store."""
    _as(monkeypatch, "commenter")
    leak = "the key is sk-ant-api03-" + ("A" * 95)
    r = _post(client, body=leak)
    assert r.status_code == 201, r.text
    stored = r.json()["note"]["body"]
    assert "sk-ant-api03-" + ("A" * 95) not in stored, f"a secret was stored verbatim: {stored}"
    assert r.json()["redactions"], "the author was not told their text was altered"


def test_an_ordinary_note_is_not_mangled(cart, client, monkeypatch):
    """The control for redaction: a filter that ate everything would pass the test above."""
    _as(monkeypatch, "commenter")
    text = "Moved from 5 to 7 years because the examiner cited FDICIA."
    r = _post(client, body=text)
    assert r.json()["note"]["body"] == text
    assert r.json()["redactions"] == []


# ── append-only ─────────────────────────────────────────────────────────────────────

def test_there_is_no_delete_route(cart, client, monkeypatch):
    """⛔ By ruling: notes are permanent. A DELETE that 405s is the design, not a gap."""
    _as(monkeypatch, "editor")
    note_id = _post(client).json()["note"]["id"]
    r = client.delete(f"/api/provenance/{CART}/notes/{note_id}")
    assert r.status_code in (404, 405), f"a delete path exists: {r.status_code}"


def test_superseding_keeps_both_notes(cart, client, monkeypatch):
    """⭐ A correction is an APPEND carrying a link, never an edit. Both notes survive and
    the superseded one is flagged on READ -- derived, never stored."""
    _as(monkeypatch, "commenter")
    first = _post(client, body="Retention is 5 years.").json()["note"]["id"]
    second = _post(client, body="Retention is 7 years -- FDICIA.", supersedes=first)
    assert second.status_code == 201, second.text

    listed = client.get(f"/api/provenance/{CART}/notes").json()["notes"]
    assert len(listed) == 2, f"superseding removed a note: {listed}"
    by_id = {n["id"]: n for n in listed}
    assert by_id[first]["superseded"] is True
    assert by_id[second.json()["note"]["id"]]["superseded"] is False


def test_the_key_written_matches_the_key_read_back(cart, client, monkeypatch):
    """⭐⭐ The whole feature turns on this. A note stored under a key the search path
    never produces is invisible -- and invisible without raising."""
    _as(monkeypatch, "commenter")
    written = _post(client).json()["target_key"]
    found = client.post(f"/api/provenance/{CART}/for-results",
                        json={"passages": [PASSAGE]}).json()
    assert found["keys"][0] == written
    assert found["by_key"][written]["notes"], "the note did not ride back with the passage"


def test_an_unknown_kind_is_422_not_a_stored_junk_note(cart, client, monkeypatch):
    _as(monkeypatch, "commenter")
    assert _post(client, kind="whatever").status_code == 422


# ── read-only: three different things say it, and they get different answers ─────────

def test_a_LOCKED_cart_can_still_be_annotated(cart, client, monkeypatch):
    """⭐⭐ ANDY'S QUESTION, 2026-11-04: should a read-only document accept a WHY?

    YES for the MOUNT LOCK. It is a click-to-unlock safety catch on cart CONTENT, and a
    note is a sidecar beside the cart. ⛔ Refusing here would mean you must UNLOCK the
    cart -- making every passage editable -- just to write a comment: a more dangerous
    state required to do a safer thing, which inverts the control. An immutable record you
    may still annotate is the audit-workpaper shape, not a compromise.
    """
    from api import main as main_mod
    monkeypatch.setattr(main_mod.engine, "read_only", True, raising=False)
    _as(monkeypatch, "commenter")
    r = _post(client)
    assert r.status_code == 201, f"a locked cart refused a note: {r.text}"


def test_a_SERVER_WIDE_read_only_refuses_notes(cart, client, monkeypatch):
    """⛔ The other answer. VPS_READ_ONLY is not about carts: the operator declared that
    this process does not write to disk, and notes.json is disk. An operator who set it
    did not consent to a growing sidecar."""
    from api import main as main_mod
    monkeypatch.setattr(main_mod, "READ_ONLY_MODE", True)
    _as(monkeypatch, "commenter")
    r = _post(client)
    assert r.status_code == 403, r.text
    assert "read-only" in r.text.lower()


def test_the_server_wide_refusal_wrote_NOTHING(cart, client, monkeypatch):
    """Existence is not operation: a 403 proves the response, not the file."""
    from api import main as main_mod
    monkeypatch.setattr(main_mod, "READ_ONLY_MODE", True)
    _as(monkeypatch, "commenter")
    _post(client)
    assert link.modules()[1].load(str(cart)).get("notes", []) == []
