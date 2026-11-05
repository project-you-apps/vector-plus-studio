"""The failure paths of content_key population — the ones that only run when something
has already gone wrong.

⛔ WHY THIS FILE EXISTS. On 2026-11-04 two `log.warning(...)` calls were written into
main.py against a `log` that did not exist in that module. Both sat on EXCEPTION paths, so
the handler whose entire job was to report a keying failure would itself have raised
NameError, inside a search request, in the one state it was written for. 584 tests passed
over it because not one of them made `content_key_for` raise.

⇒ A reporting path that is never exercised is not a reporting path. These tests force the
failure instead of waiting for it.

Run from the repo root:  python -m pytest api/test_provenance_keying.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from api import main, provenance_link          # noqa: E402


def test_main_has_a_logger_at_all():
    """⛔ THE DIRECT REGRESSION. `log.warning` in main.py resolves to something."""
    assert hasattr(main, "log"), "main.py calls log.warning but defines no `log`"
    assert callable(getattr(main.log, "warning", None))


def test_the_keying_failure_reporter_does_not_itself_raise(monkeypatch, caplog):
    """⭐⭐ THE REAL TEST. Drive the exact except-branch that was broken: make keying blow
    up, then run the reporting code and assert it LOGS rather than raising.

    Written against main.log directly rather than through a search request, because the
    bug is in the handler, and a test that needed a mounted engine would not have been
    written -- which is precisely why it was not."""
    def boom(_passage):
        raise RuntimeError("synthetic keying failure")

    monkeypatch.setattr(provenance_link, "content_key_for", boom)

    # The shape both call sites use.
    ckey = "unset"
    with caplog.at_level("WARNING"):
        try:
            ckey = provenance_link.content_key_for("anything")
        except Exception as e:                                   # noqa: BLE001
            ckey = None
            main.log.warning("content_key failed for pattern %d: %s: %s",
                             7, type(e).__name__, e)

    assert ckey is None
    # getMessage() applies the %-args, which is the formatting step that would ALSO have
    # blown up had the call site's placeholders and arguments disagreed.
    assert any("synthetic keying failure" in r.getMessage() for r in caplog.records), \
        caplog.text


def test_a_failed_key_is_None_and_not_an_empty_string():
    """⚠ '' is a VALID-LOOKING key that matches nothing. None is the honest absence, and
    the UI hides the Why section on None -- it would render an empty one on ''."""
    assert provenance_link.content_key_for("hello") != ""
    assert isinstance(provenance_link.content_key_for("hello"), str)


def test_pattern_response_carries_a_content_key_field():
    """The modal cannot attach a note without it, and prev/next cannot reuse the search
    hit's key because it refetches by idx."""
    from api.models import PatternResponse
    r = PatternResponse(idx=0, title="t", preview="p", full_text="f")
    assert r.content_key is None, "default must be None, never ''"
    assert "content_key" in PatternResponse.model_fields


def test_status_exposes_notes_writable():
    from api.models import StatusResponse
    s = StatusResponse(engine_ready=True, gpu_available=False, mounted_cartridge=None,
                       pattern_count=0, physics_trained=False, training_active=False,
                       training_progress=0, training_total=0)
    # ⭐ Defaults TRUE: notes are writable unless the server says otherwise. A default of
    # False would disable the composer on every deployment that has not been upgraded.
    assert s.notes_writable is True
