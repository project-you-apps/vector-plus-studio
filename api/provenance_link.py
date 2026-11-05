"""The bridge from the Studio to Membot's provenance modules.

WHY THIS FILE EXISTS AT ALL
---------------------------
Membot grew four provenance modules on 2026-11-01 -- `note_store` (the human "why"),
`journal` (an append-only hash-chained record of what happened), `subscriptions`
("what changed on the things I watch"), and `seat_identity` (verified vs merely claimed).
All four are stdlib-only and address a cart by its FILESYSTEM PATH, so they work unchanged
against any cart the Studio can already see. Nothing had to be written twice.

⛔⛔ AND NOTHING MAY BE. These sidecars are keyed by `content_key(passage_text)`, which
already exists in SIX places in this project (`membot/subcart.py` plus five copies under
`tools/`). A seventh would be the obvious mistake, and the failure mode is the quiet kind:
if the Studio computed a key even slightly differently -- one stripped header, one trailing
newline -- every lookup would return zero rows, the panel would render empty, and NOTHING
would raise. "No notes on this passage" and "we cannot find the notes on this passage"
would be indistinguishable.

⚠ `main.py` already knows this. `_credit_seat` carries the same warning about hashing a
truncated display body -- *"content keys that match nothing in the parent -- silently,
forever. Same trap membot had; same fix."* This module inherits that discipline rather
than re-deriving it, and `test_provenance_link.py` asserts both trees agree.

SYS.PATH DISCIPLINE -- THE HOUSE RULE, NOT MY PREFERENCE
--------------------------------------------------------
`main.py:295` states it: membot's directory must NEVER be left on `sys.path`, because
VPS-local modules exist in BOTH repos (`multi_lattice_wrapper_v7.py` is the named case) and
a permanent entry can shadow them. My first version appended and left it there, which
would have worked today and broken whenever a later import happened to resolve first from
membot.

⚠ But the modules cannot be loaded in total isolation either: `subscriptions` imports
`journal` and `note_store` BY BARE NAME. So this follows `_overlay_modules()` exactly --
insert membot at the front of `sys.path`, import inside the block, and remove it in a
`finally`. Loaded once, cached, path clean afterwards.

ⓘ `main.py` derives `_membot_dir` the same way, three lines of path join. Consolidating the
two resolvers into this module is a tidy follow-up, deliberately NOT done the night before
a demo: it means editing the import region of a 160 KB file for no behavioural gain.

GUARDED IMPORT, AND THE OPPOSITE CHOICE FROM MEMBOT'S OWN
---------------------------------------------------------
In `membot_server.py` these same modules are imported UNGUARDED, because there they are
controls on a write path -- ingest redaction and seat identity. A missing file there must
be a loud boot failure, never a server that keeps accepting writes while the logs imply a
filter ran.

Here they back READ-ONLY display. The correct failure is to degrade that one feature, not
to refuse to start a product other people are using. So the import is guarded and
`availability()` reports exactly why it is unavailable.

⛔ But an unavailable layer must NEVER answer with an empty list. That turns "we cannot
read the record" into "there is no record", the one translation a provenance feature must
never make. Callers get `available: False` plus a reason; the routes turn that into a
response that says so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "availability", "membot_dir", "content_key_for", "modules",
    "journal", "note_store", "subscriptions", "seat_identity",
]

_ENV_VAR = "MEMBOT_PATH"


def membot_dir() -> Path | None:
    """Where membot lives, or None if we cannot find it.

    `MEMBOT_PATH` wins so a deployment can place the trees anywhere. The fallback matches
    the working copy: `project-you/vector-plus-studio-repo/api/` sits two levels under
    `project-you/`, which also holds `membot/`.
    """
    env = (os.environ.get(_ENV_VAR) or "").strip()
    if env:
        p = Path(env).expanduser()
        return p if (p / "subcart.py").exists() else None

    here = Path(__file__).resolve()
    candidate = here.parent.parent.parent / "membot"      # api/ -> repo/ -> project-you/
    return candidate if (candidate / "subcart.py").exists() else None


_LOADED: tuple | None | str = "unloaded"
_IMPORT_ERROR: str | None = None
_DIR = membot_dir()


def modules() -> tuple | None:
    """(journal, note_store, subscriptions, seat_identity, subcart) or None. Cached."""
    global _LOADED, _IMPORT_ERROR
    if _LOADED != "unloaded":
        return _LOADED

    if _DIR is None:
        _IMPORT_ERROR = (
            f"membot not found. Set {_ENV_VAR} to the directory containing subcart.py "
            f"(looked for a sibling 'membot' directory beside this repo)."
        )
        _LOADED = None
        return None

    d = str(_DIR)
    added = d not in sys.path
    if added:
        # Front of the path for the duration of this block only -- `subscriptions` imports
        # `journal` and `note_store` by bare name and must resolve them here.
        sys.path.insert(0, d)
    try:
        import journal as _journal                     # noqa: E402
        import note_integrity as _note_integrity       # noqa: E402
        import note_store as _note_store               # noqa: E402
        import redact as _redact                       # noqa: E402
        import seat_identity as _seat_identity         # noqa: E402
        import subcart as _subcart                     # noqa: E402
        import subscriptions as _subscriptions         # noqa: E402
        # ⚠ APPEND ONLY. Indices are positional and `content_key_for` reads mods[4]; a
        # reorder here silently repoints that at a different module.
        _LOADED = (_journal, _note_store, _subscriptions, _seat_identity, _subcart,
                   _redact, _note_integrity)
    except Exception as e:                             # noqa: BLE001
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        _LOADED = None
    finally:
        # ⛔ Always, including on the exception path. A half-failed import that leaves
        # membot on sys.path is exactly the shadowing main.py:295 warns about.
        if added and d in sys.path:
            sys.path.remove(d)
    return _LOADED


class _Lazy:
    """Attribute access that loads on first use.

    Keeps `link.journal.verify(...)` readable at the call sites while still deferring the
    import, so merely importing this module cannot fail the app's startup.
    """

    __slots__ = ("_i", "_name")

    def __init__(self, index: int, name: str):
        self._i = index
        self._name = name

    def __getattr__(self, attr):
        mods = modules()
        if mods is None:
            raise RuntimeError(_IMPORT_ERROR or "membot provenance modules unavailable")
        return getattr(mods[self._i], attr)

    def __bool__(self):
        return modules() is not None


journal = _Lazy(0, "journal")
note_store = _Lazy(1, "note_store")
subscriptions = _Lazy(2, "subscriptions")
seat_identity = _Lazy(3, "seat_identity")
# ⭐ A typed note is a NEW TEXT INGRESS into the cart's sidecars, so it gets the same
# treatment as an ingested passage. Our own corpus is the proven case: it already holds
# an API key and OAuth tokens because nothing stood between paste and store.
redact = _Lazy(5, "redact")
# ⭐ Answers "did anyone remove a note from this discussion?" by reconciling notes.json
# against the hash-chained journal, which already witnesses every note creation. No second
# chain: the witness existed, nobody was asking it.
note_integrity = _Lazy(6, "note_integrity")


def availability() -> dict:
    """Whether the provenance layer can answer, and why not when it cannot.

    ⛔ Callers must branch on this rather than treating an empty result as "nothing here".
    """
    mods = modules()
    ok = mods is not None
    return {
        "available": ok,
        "membot_dir": str(_DIR) if _DIR else None,
        "error": None if ok else (_IMPORT_ERROR or "provenance modules not loaded"),
        "modules": {
            name: ok for name in
            ("journal", "note_store", "subscriptions", "seat_identity", "subcart",
             "redact", "note_integrity")
        },
    }


def content_key_for(passage: str) -> str:
    """Membot's content key for a passage, unsalted.

    ⛔ A thin pass-through ON PURPOSE. Resist every temptation to normalise, strip or
    lowercase on the way in: `subcart.content_key` already decides what to strip (leading
    `SESSION:` headers, because /store prepends a fresh one on arrival), and a second
    opinion here produces keys that match nothing while raising nothing.
    """
    mods = modules()
    if mods is None:
        raise RuntimeError(_IMPORT_ERROR or "subcart unavailable")
    return mods[4].content_key(passage)
