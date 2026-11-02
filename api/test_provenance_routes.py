"""Do the provenance routes OPERATE, through the real app?

⛔ `test_provenance_link.py` proves the bridge computes the right keys, and membot's own
suites prove the modules work. Neither proves a route reaches them. This drives the app
with a real cart, notes and journal entries written by membot's own modules, and asserts
what comes back over HTTP.

The cart directory is patched rather than the resolver, so `name_candidates` and
`find_cartridge_path` stay real -- a test that stubbed the lookup would pass even if a bare
name never resolved to a filename, which is the exact bug `resolve_named` exists to avoid.

Run from the repo root:  python -m pytest api/test_provenance_routes.py -q
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

from api import cartridge_io                         # noqa: E402
from api.main import app                             # noqa: E402

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("provenance_link", _HERE / "provenance_link.py")
link = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(link)

CART_NAME = "provroute-probe"
SEAT_A = "seat-alice-route"
SEAT_B = "seat-bob-route"

PASSAGE = "Retention: AP documents are held for 7 years from period close."
OTHER = "AP control 4.2: invoices over $10,000 require dual approval."


@pytest.fixture()
def seeded(monkeypatch):
    """A cart file with notes and a journal beside it, discoverable by the real resolver."""
    if link.modules() is None:
        pytest.skip(f"provenance layer unavailable: {link.availability()['error']}")

    d = Path(tempfile.mkdtemp())
    cart = d / f"{CART_NAME}.cart.npz"
    cart.write_bytes(b"")                    # only its PATH matters to the sidecars
    monkeypatch.setattr(cartridge_io, "get_cartridge_dirs", lambda: [str(d)])

    mods = link.modules()
    journal, note_store = mods[0], mods[1]
    key = link.content_key_for(PASSAGE)

    note_store.add_note(str(cart), key, "decision",
                        "Moved from 5 to 7 years because the examiner cited FDICIA.",
                        by=SEAT_A, surface="WEB", scope="shared")
    note_store.add_note(str(cart), key, "caution",
                        "PRIVATE-CANARY not sure this reading is right yet",
                        by=SEAT_B, surface="WEB", scope="private")
    journal.append_to_cart(str(cart), writer=SEAT_A, surface="WEB", op="store",
                           target={"cart": CART_NAME, "idx": 0, "target_key": key},
                           payload={"chars": len(PASSAGE), "identity_source": "verified"})
    return {"dir": d, "cart": cart, "key": key}


@pytest.fixture()
def client():
    return TestClient(app)


def test_status_reports_a_verifying_chain(seeded, client):
    r = client.get(f"/api/provenance/{CART_NAME}/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance_available"] is True, body
    assert body["integrity"]["ok"] is True, body["integrity"]
    assert body["integrity"]["entries"] >= 1
    assert body["notes"]["readable"] is True


def test_for_results_returns_provenance_and_the_why(seeded, client):
    """⭐⭐ THE ROUTE THE RESULT CARD CALLS."""
    r = client.post(f"/api/provenance/{CART_NAME}/for-results",
                    json={"passages": [PASSAGE, OTHER]})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["keys"][0] == seeded["key"], "server-side keying disagrees with the bridge"
    entry = body["by_key"][seeded["key"]]
    assert entry["provenance"], f"no journal entries rode back with the hit: {body}"
    assert any("examiner cited FDICIA" in n["body"] for n in entry["notes"]), (
        f"the WHY did not travel with the result: {entry['notes']}"
    )
    assert body["entries_withheld"] is False


def test_an_unknown_passage_gets_a_key_but_no_rows(seeded, client):
    """The control for for-results. A route that returned the same payload for everything
    would pass the test above."""
    r = client.post(f"/api/provenance/{CART_NAME}/for-results",
                    json={"passages": [OTHER]})
    body = r.json()
    k = body["keys"][0]
    assert k, "a perfectly ordinary passage failed to key"
    assert body["by_key"][k]["notes"] == []
    assert body["by_key"][k]["provenance"] == []


def test_anonymous_callers_never_see_a_private_note(seeded, client):
    """⭐⭐ THE SURVEILLANCE BOUNDARY, over HTTP. These requests carry no token, so the seat
    is '' -- which `note_store` reads as unidentified and answers with SHARED only."""
    r = client.get(f"/api/provenance/{CART_NAME}/notes")
    assert r.status_code == 200, r.text
    text = r.text
    assert "PRIVATE-CANARY" not in text, f"a private note leaked over HTTP:\n{text}"
    assert "examiner cited FDICIA" in text, "the shared note should be visible"

    r2 = client.post(f"/api/provenance/{CART_NAME}/for-results",
                     json={"passages": [PASSAGE]})
    assert "PRIVATE-CANARY" not in r2.text, (
        f"a private note leaked through for-results:\n{r2.text}"
    )


def test_a_broken_chain_withholds_entries_and_says_so(seeded, client):
    """⛔ Entries read out of a record that may have been edited are not evidence."""
    mods = link.modules()
    jpath = mods[0].journal_path_for(str(seeded["cart"]))
    key = seeded["key"]
    # Three entries, then remove the middle one: an interior deletion breaks the chain.
    for i in (1, 2):
        mods[0].append_to_cart(str(seeded["cart"]), writer=SEAT_A, surface="WEB", op="note",
                               target={"cart": CART_NAME, "idx": 0, "target_key": key},
                               payload={"n": i})
    lines = Path(jpath).read_text(encoding="utf-8").splitlines()
    Path(jpath).write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    r = client.post(f"/api/provenance/{CART_NAME}/for-results",
                    json={"passages": [PASSAGE]})
    body = r.json()
    assert body["integrity"]["ok"] is False, "a broken chain verified clean"
    assert body["entries_withheld"] is True
    assert body["by_key"][key]["provenance"] == [], "entries served from a broken journal"
    # ⭐ But the human statement survives: a note is not a claim about an unaltered history.
    assert body["by_key"][key]["notes"], "the notes were suppressed along with the entries"


def test_an_unknown_cart_is_404_not_an_empty_answer(seeded, client):
    r = client.get("/api/provenance/no-such-cart-anywhere/status")
    assert r.status_code == 404, r.text


def test_too_many_passages_is_refused(seeded, client):
    r = client.post(f"/api/provenance/{CART_NAME}/for-results",
                    json={"passages": [f"p{i}" for i in range(500)]})
    assert r.status_code == 413, r.text


def test_what_changed_is_honest_for_an_anonymous_caller(seeded, client):
    """An anonymous caller watches nothing. It must say so rather than fall back to
    'everything that changed', which would be a different feature wearing this one's name."""
    r = client.get(f"/api/provenance/{CART_NAME}/what-changed")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changes"] == []
    assert body.get("seat") is None
    assert "Sign in" in (body.get("verdict_line") or "")
