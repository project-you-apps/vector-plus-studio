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


def test_keys_path_returns_the_same_rows_as_the_text_path(seeded, client):
    """⭐ The frontend's path. A search response carries `content_key`; handing it straight
    back must give exactly what sending the full body gives."""
    by_text = client.post(f"/api/provenance/{CART_NAME}/for-results",
                          json={"passages": [PASSAGE]}).json()
    by_key = client.post(f"/api/provenance/{CART_NAME}/for-results",
                         json={"keys": [seeded["key"]]}).json()

    assert by_key["keys"] == by_text["keys"] == [seeded["key"]]
    assert by_key["by_key"][seeded["key"]]["notes"], "keys path returned no notes"
    assert (len(by_key["by_key"][seeded["key"]]["provenance"])
            == len(by_text["by_key"][seeded["key"]]["provenance"]))


def test_a_preview_derived_key_still_finds_nothing(seeded, client):
    """⛔⛔ THE CONTROL FOR THE WHOLE DESIGN. The reason the server sends the key is that a
    key over a SHORTENED body matches nothing and raises nothing. If this ever starts
    passing, truncation stopped mattering and the design can be revisited -- until then it
    is the measured justification for SearchResult.content_key."""
    preview = PASSAGE[:40]
    assert preview != PASSAGE
    res = client.post(f"/api/provenance/{CART_NAME}/for-results",
                      json={"passages": [preview]}).json()
    k = res["keys"][0]
    assert k and k != seeded["key"], "a truncated body produced the full body's key"
    assert res["by_key"][k]["notes"] == [], "a preview-derived key matched notes"
    assert res["by_key"][k]["provenance"] == []


def test_keys_and_passages_together_stay_parallel(seeded, client):
    """Both may be supplied; `keys` must stay parallel to (keys + passages) in order."""
    res = client.post(f"/api/provenance/{CART_NAME}/for-results",
                      json={"keys": [seeded["key"]], "passages": [OTHER]}).json()
    assert len(res["keys"]) == 2
    assert res["keys"][0] == seeded["key"]
    assert res["keys"][1] and res["keys"][1] != seeded["key"]


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


def test_status_reconciles_notes_against_the_journal(seeded, client):
    """⭐ The seeded cart has two notes written by note_store DIRECTLY, with no matching
    journal entries, so they are unwitnessed: `ok` (nothing removed) but NOT `attested`."""
    body = client.get(f"/api/provenance/{CART_NAME}/status").json()
    ni = body["notes_integrity"]
    assert ni["ok"] is True, ni
    assert ni["attested"] is False, "unwitnessed notes must not read as verified"
    assert ni["checked"] == 2


def test_status_DETECTS_a_note_spliced_out_of_the_sidecar(seeded, client):
    """⛔⛔ THE ATTACK THE HASH CHAIN ALONE CANNOT SEE. Removing an object from notes.json
    leaves the journal verifying perfectly -- only the reconciliation notices."""
    import json as _json
    mods = link.modules()
    journal, note_store = mods[0], mods[1]
    cart = str(seeded["cart"])

    n = note_store.add_note(cart, seeded["key"], "why", "the inconvenient one",
                            by=SEAT_A, surface="WEB", scope="shared")
    journal.append_to_cart(cart, writer=SEAT_A, surface="WEB", op="note",
                           target={"target_key": seeded["key"]},
                           payload={"note_id": n["id"]})

    p = Path(note_store.sidecar_path(cart))
    data = _json.loads(p.read_text(encoding="utf-8"))
    notes = data["notes"] if isinstance(data, dict) else data
    kept = [x for x in notes if x.get("id") != n["id"]]
    if isinstance(data, dict):
        data["notes"] = kept
    else:
        data = kept
    p.write_text(_json.dumps(data), encoding="utf-8")

    body = client.get(f"/api/provenance/{CART_NAME}/status").json()
    # The chain is still fine -- that is the whole point.
    assert body["integrity"]["ok"] is True, "the splice broke the chain; wrong test"
    ni = body["notes_integrity"]
    assert ni["ok"] is False, ni
    assert n["id"] in ni["orphaned_in_journal"], ni
    assert "MISSING" in ni["verdict_line"]


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
