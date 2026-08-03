"""The studio edit verb, and the succession it records.

WHY THIS TEST FILE EXISTS. Succession shipped on 2026-08-03 morning with `record_succession`,
`build_succession`, `follow_succession` and resolver integration — all tested, and all with
ZERO callers. The tests passed because each one built its own chain by hand. Nothing in any
product ever recorded an edit, so a mechanism described as "closing the edit bug" closed
nothing.

These tests are about the WIRING, not the algorithm: does the endpoint actually call it.

    PYTHONIOENCODING=utf-8 python -m pytest api/tests/test_edit_succession.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from api import main as m
from api.models import AddPassageRequest, MessageResponse


V1 = "Redwood's on-call rotation is weekly, handed over on Mondays."
V2 = "Redwood's on-call rotation is FORTNIGHTLY, handed over on Mondays."


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    """A mounted, writable studio with two passages and a real DATA_DIR."""
    monkeypatch.setattr(m, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(m.engine, "mounted_name", "redwood")
    monkeypatch.setattr(m.engine, "read_only", False)
    monkeypatch.setattr(m.engine, "passages", ["unrelated passage", V1])
    monkeypatch.setattr(m.engine, "deleted_ids", set())
    monkeypatch.setattr(m, "_enforce_writable", lambda **kw: None)

    def fake_add(text):
        m.engine.passages.append(text)
        return MessageResponse(success=True, message="added")
    monkeypatch.setattr(m, "_add_passage_sync", fake_add)
    return tmp_path


# ------------------------------------------------------------------ the wiring

def test_edit_appends_tombstones_and_records(studio):
    resp = run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))
    assert resp.success
    assert m.engine.passages[-1] == V2          # appended
    assert 1 in m.engine.deleted_ids            # old tombstoned

    succ = m._succession_module()
    mapping = succ.build_map(studio / "redwood.pkl")
    import subcart as sc  # noqa: E402  (membot is on sys.path via the loader)
    assert mapping.get(sc.content_key(V1)) == sc.content_key(V2)


def test_a_cluster_holding_the_old_key_follows_the_edit(studio):
    """The bug in one test: a curated cluster must not shed a member because someone
    fixed a sentence in it."""
    run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))

    import subcart as sc  # noqa: E402
    succ = m._succession_module()
    manifest = sc.new_manifest("oncall-notes", "redwood.pkl",
                               keys=[sc.content_key(V1)], created_at=1.0)

    without = sc.resolve(manifest, m.engine.passages)
    with_succ = sc.resolve(manifest, m.engine.passages,
                           succession=succ.build_map(studio / "redwood.pkl"))

    assert m.engine.passages[with_succ["indices"][0]] == V2
    assert with_succ["stats"]["followed"] == 1
    # Without the map it lands on the OLD text — which is what shipped this morning.
    assert m.engine.passages[without["indices"][0]] == V1


def test_the_seat_is_recorded_as_the_editor(studio):
    run(m.edit_pattern(1, AddPassageRequest(text=V2),
                       user={"sub": "3579e6ee-1111"}))
    succ = m._succession_module()
    entries = succ.load_log(studio / "redwood.pkl")["entries"]
    assert entries[0]["by"] == "3579e6ee-1111"


def test_an_anonymous_edit_is_attributed_to_the_studio(studio, monkeypatch):
    monkeypatch.delenv("VPS_SEAT", raising=False)
    run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))
    succ = m._succession_module()
    assert succ.load_log(studio / "redwood.pkl")["entries"][0]["by"] == "studio"


# ------------------------------------------------------------------ refusals

def test_editing_out_of_range_changes_nothing(studio):
    resp = run(m.edit_pattern(99, AddPassageRequest(text=V2), user=None))
    assert not resp.success and "Invalid index" in resp.message
    assert m.engine.deleted_ids == set()


def test_empty_replacement_is_refused(studio):
    resp = run(m.edit_pattern(1, AddPassageRequest(text="   "), user=None))
    assert not resp.success and "empty" in resp.message.lower()
    assert m.engine.deleted_ids == set()


def test_an_unchanged_edit_is_refused(studio):
    """Otherwise the succession log fills with entries where nothing was superseded."""
    resp = run(m.edit_pattern(1, AddPassageRequest(text=V1), user=None))
    assert not resp.success and "unchanged" in resp.message.lower()


def test_a_read_only_cart_refuses(studio, monkeypatch):
    monkeypatch.setattr(m.engine, "read_only", True)
    resp = run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))
    assert not resp.success and "read-only" in resp.message.lower()
    assert m.engine.deleted_ids == set()


# ------------------------------------------------------------------ ordering

def test_nothing_is_tombstoned_when_the_append_fails(studio, monkeypatch):
    """Tombstoning first would leave the passage deleted with no replacement."""
    monkeypatch.setattr(m, "_add_passage_sync",
                        lambda text: MessageResponse(success=False, message="embed failed"))
    resp = run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))
    assert not resp.success
    assert m.engine.deleted_ids == set()
    assert m.engine.passages == ["unrelated passage", V1]


def test_a_failed_succession_record_warns_the_caller(studio, monkeypatch):
    """An edit whose link was NOT recorded is the silent-orphan case this endpoint exists
    to prevent — the caller must be told, not left assuming it worked."""
    class Broken:
        @staticmethod
        def record(*a, **k):
            raise OSError("disk gone")
    monkeypatch.setattr(m, "_succession_module", lambda: Broken)
    resp = run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))
    assert resp.success                      # the edit itself DID happen
    # STRUCTURED, not prose. A client must be able to detect this without string-matching.
    assert resp.succession_recorded is False
    assert "not follow this edit" in resp.message


def test_a_successful_edit_reports_the_link_was_recorded(studio):
    resp = run(m.edit_pattern(1, AddPassageRequest(text=V2), user=None))
    assert resp.succession_recorded is True


def test_non_edit_responses_leave_the_flag_unset(studio):
    """None means 'not applicable', so a client cannot mistake a delete for a failed edit."""
    resp = run(m.delete_pattern(1))
    assert resp.succession_recorded is None
