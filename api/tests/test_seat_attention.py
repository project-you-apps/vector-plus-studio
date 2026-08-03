"""Per-seat attention in the STUDIO.

Until 2026-08-03 the studio recorded none. memory_server and membot both tracked it, so
"Ben's hot stack versus Karthik's" existed everywhere except the product with the UI — the
one place you would actually show someone.

    PYTHONIOENCODING=utf-8 python -m pytest api/tests/test_seat_attention.py -q
"""

from __future__ import annotations

import os

import pytest

from api import main as m


# ------------------------------------------------------------------ seat resolution

def test_a_signed_in_user_is_their_own_seat():
    assert m._seat_for({"sub": "3579e6ee-6412-4099-8d66-a205d9be7849"}) == \
        "3579e6ee-6412-4099-8d66-a205d9be7849"


def test_the_jwt_beats_the_local_fallback(monkeypatch):
    monkeypatch.setenv("VPS_SEAT", "local-andy")
    assert m._seat_for({"sub": "real-uuid"}) == "real-uuid"


def test_local_fallback_is_used_when_nobody_is_signed_in(monkeypatch):
    """A single-user local studio should still build a hot stack."""
    monkeypatch.setenv("VPS_SEAT", "local-andy")
    assert m._seat_for(None) == "local-andy"


def test_an_anonymous_visitor_with_no_local_seat_is_not_tracked(monkeypatch):
    """On the public demo, an anonymous visitor must not quietly accumulate into
    somebody's attention profile."""
    monkeypatch.delenv("VPS_SEAT", raising=False)
    assert m._seat_for(None) is None
    assert m._seat_for({}) is None
    assert m._seat_for({"sub": ""}) is None
    assert m._seat_for("not a dict") is None


def test_whitespace_only_seat_is_not_a_seat(monkeypatch):
    monkeypatch.setenv("VPS_SEAT", "   ")
    assert m._seat_for(None) is None


# ------------------------------------------------------------------ recording

def test_recording_without_a_seat_is_a_no_op(monkeypatch):
    """Must not raise, must not write, must not invent a seat."""
    called = []
    monkeypatch.setattr(m, "_overlay_store_for_mounted",
                        lambda: called.append(1) or None)
    m._record_seat_attention(None, ["some passage"])
    assert called == []


def test_recording_with_no_passages_is_a_no_op(monkeypatch):
    called = []
    monkeypatch.setattr(m, "_overlay_store_for_mounted",
                        lambda: called.append(1) or None)
    m._record_seat_attention("seat-a", [])
    assert called == []


def test_recording_touches_the_store_with_content_keys(monkeypatch, tmp_path):
    mods = m._overlay_modules()
    assert mods is not None, "overlay modules should be importable in-repo"
    _, store_mod, sc = mods
    store = store_mod.OverlayStore(tmp_path, parent_cart="brain.pkl")
    monkeypatch.setattr(m, "_overlay_store_for_mounted", lambda: store)

    m._record_seat_attention("seat-a", ["alpha body", "beta body"])
    overlay = store.get_overlay("seat-a")
    assert len(overlay["entries"]) == 2
    assert sc.content_key("alpha body") in overlay["entries"]


def test_two_seats_do_not_share_studio_attention(monkeypatch, tmp_path):
    mods = m._overlay_modules()
    _, store_mod, sc = mods
    store = store_mod.OverlayStore(tmp_path, parent_cart="brain.pkl")
    monkeypatch.setattr(m, "_overlay_store_for_mounted", lambda: store)

    m._record_seat_attention("seat-a", ["shared passage"])
    key = sc.content_key("shared passage")
    assert key in store.get_overlay("seat-a")["entries"]
    assert key not in store.get_overlay("seat-b")["entries"]


def test_a_store_failure_never_propagates(monkeypatch):
    """Attention is additive to a product that worked without it."""
    class Exploding:
        stats = {}
        def touch_many(self, *a, **k):
            raise OSError("disk gone")
        def flush(self, *a, **k):
            raise OSError("disk gone")
    monkeypatch.setattr(m, "_overlay_store_for_mounted", lambda: Exploding())
    m._record_seat_attention("seat-a", ["body"])          # must not raise


def test_keys_come_from_full_bodies_not_previews(monkeypatch, tmp_path):
    """The trap membot had: a key computed from truncated display text matches nothing in
    the cart, silently and permanently."""
    mods = m._overlay_modules()
    _, store_mod, sc = mods
    store = store_mod.OverlayStore(tmp_path, parent_cart="brain.pkl")
    monkeypatch.setattr(m, "_overlay_store_for_mounted", lambda: store)

    body = "SESSION: x\n" + ("long enterprise passage " * 60)
    m._record_seat_attention("seat-a", [body])
    entries = store.get_overlay("seat-a")["entries"]
    assert sc.content_key(body) in entries
    assert sc.content_key(body[:500] + "...") not in entries


# ------------------------------------------------------------------ status surface

def test_status_says_why_it_is_off_when_no_cart_is_mounted(monkeypatch):
    monkeypatch.setattr(m.engine, "mounted_name", None)
    status = m._seat_attention_status()
    assert status["enabled"] is False and "no cart mounted" in status["reason"]


def test_status_says_why_it_is_off_when_modules_are_missing(monkeypatch):
    monkeypatch.setattr(m, "_overlay_modules", lambda: None)
    status = m._seat_attention_status()
    assert status["enabled"] is False and "not importable" in status["reason"]


def test_status_reports_counters_when_on(monkeypatch, tmp_path):
    mods = m._overlay_modules()
    _, store_mod, _ = mods
    store = store_mod.OverlayStore(tmp_path, parent_cart="brain.pkl")
    monkeypatch.setattr(m, "_overlay_store_for_mounted", lambda: store)
    monkeypatch.setattr(m.engine, "mounted_name", "brain")

    m._record_seat_attention("seat-a", ["one", "two"])
    status = m._seat_attention_status()
    assert status["enabled"] is True
    assert status["touches"] == 2 and status["flush_errors"] == 0
    assert status["rows"]["seat-a"] == 2
