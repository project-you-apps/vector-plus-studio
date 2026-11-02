"""Does the Studio compute the SAME content key as Membot?

⛔⛔ THIS IS THE TEST THE WHOLE BRIDGE EXISTS FOR, and it is here because the failure it
guards against is silent. Notes and journal entries are addressed by
`content_key(passage_text)`. If the Studio's idea of that key differs from Membot's by one
stripped header or one trailing newline, then:

    - every lookup returns zero rows
    - the panel renders "no notes on this passage"
    - nothing raises, nothing logs, and the feature looks finished

That is indistinguishable from a passage genuinely having no notes, which makes it the
worst class of defect this project has: a mechanism that is complete, reachable, and
quietly answering the wrong question. `seat_hot_stack.py` carries the same warning about
hashing a 500-char truncated display string instead of the body -- "every overlay looking
populated, every resolve returning zero, no error anywhere."

⇒ So the key is not reimplemented, and this asserts the pass-through really is one.

Run from the repo root:  python -m pytest api/test_provenance_link.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the bridge as a plain module, so this file runs without the FastAPI app.
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("provenance_link", _HERE / "provenance_link.py")
provenance_link = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provenance_link)


PASSAGES = [
    "AP control 4.2: invoices over $10,000 require dual approval.",
    "Retention: AP documents are held for 7 years from period close.",
    # The cases most likely to diverge between two implementations:
    "SESSION: 2026-11-01 | live-store\nThe body after a session header.",
    "trailing newline at the end\n",
    "   leading and trailing whitespace   ",
    "unicode: café — naïve — 日本語",
    "",
    "a" * 5000,
]


def _membot_subcart():
    """Membot's module, imported by its own path -- NOT through the bridge.

    Deliberately a second, independent import: if the bridge were accidentally
    reimplementing the key, comparing the bridge against itself would pass.
    """
    d = provenance_link.membot_dir()
    if d is None:
        pytest.skip(f"membot not found: {provenance_link.availability()['error']}")
    # ⛔ Load by FILE PATH and do not touch sys.path. An earlier version of this helper
    # appended membot to sys.path, which made `test_membot_is_not_left_on_sys_path` fail
    # against the test's own pollution rather than the bridge's -- the instrument
    # measuring itself. `subcart` has no cross-imports (checked), so a bare
    # spec_from_file_location load is sufficient.
    spec = importlib.util.spec_from_file_location("_membot_subcart_probe", d / "subcart.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_bridge_is_available():
    """If this fails, every provenance route will 503 -- which is correct behavior, but
    the deployment is misconfigured and the reason should be legible."""
    a = provenance_link.availability()
    assert a["available"], f"provenance layer unavailable: {a['error']}"
    assert a["modules"]["subcart"] is True


@pytest.mark.parametrize("text", PASSAGES)
def test_content_key_matches_membot_exactly(text):
    """⭐⭐ THE ONE THAT MATTERS."""
    mine = provenance_link.content_key_for(text)
    theirs = _membot_subcart().content_key(text)
    assert mine == theirs, (
        f"CONTENT KEY MISMATCH -- every note and journal lookup from the Studio would "
        f"return zero rows with no error.\n  studio: {mine}\n  membot: {theirs}\n"
        f"  input: {text[:60]!r}"
    )


def test_keys_differ_for_different_text():
    """The control. A function returning a constant would pass every test above."""
    a = provenance_link.content_key_for(PASSAGES[0])
    b = provenance_link.content_key_for(PASSAGES[1])
    assert a != b, "the key is not a function of the content"


def test_key_is_stable_across_calls():
    t = PASSAGES[0]
    assert provenance_link.content_key_for(t) == provenance_link.content_key_for(t)


def test_session_header_is_stripped_the_same_way():
    """The specific divergence risk named in subcart's docstring: a synced passage can
    accumulate a second SESSION: header, so the key is computed over the body. If the two
    trees disagree about that, notes written by an agent would not be found by the web UI
    for the very passages most likely to carry notes."""
    sub = _membot_subcart()
    headed = "SESSION: 2026-11-01 | live-store\nSame body text here."
    bare = "Same body text here."
    assert provenance_link.content_key_for(headed) == sub.content_key(headed)
    # And pin the behaviour itself, so a change in subcart is visible here rather than
    # only in membot's own suite.
    assert sub.content_key(headed) == sub.content_key(bare), (
        "subcart no longer strips the SESSION header; the Studio's keys are still in "
        "agreement, but anything that stored a key for a bare body has drifted"
    )


def test_unavailable_layer_raises_rather_than_returning_a_key():
    """⛔ Degrade loudly. A bridge that returned '' or a hash of nothing when membot was
    missing would make 'cannot read' look like 'nothing there'."""
    saved = provenance_link._LOADED
    try:
        provenance_link._LOADED = None
        with pytest.raises(RuntimeError):
            provenance_link.content_key_for("anything")
        assert provenance_link.availability()["available"] is False
    finally:
        provenance_link._LOADED = saved


def test_membot_is_not_left_on_sys_path():
    """⛔⛔ THE HOUSE RULE, now testable. `main.py:295`: membot's directory must NEVER stay
    on sys.path, because VPS-local modules exist in BOTH repos -- `multi_lattice_wrapper_v7`
    is the named case -- and a permanent entry can shadow them.

    The first version of this bridge appended and left it there. It worked that day, and
    would have broken whenever a later import happened to resolve from membot first. That
    is the class of bug that gets blamed on the wrong change six weeks later.
    """
    d = provenance_link.membot_dir()
    if d is None:
        pytest.skip("membot not found")

    # Force a load, then assert the path is clean again.
    assert provenance_link.modules() is not None
    assert str(d) not in sys.path, (
        f"membot was left on sys.path after loading: {d}. VPS-local modules that also "
        f"exist in membot can now be shadowed."
    )


def test_modules_load_once_and_are_cached():
    """Re-entry must not repeat the sys.path dance on every request."""
    first = provenance_link.modules()
    second = provenance_link.modules()
    assert first is second, "modules() is reloading rather than caching"
