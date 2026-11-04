"""Read-only provenance routes: where a passage came from, and WHY it says what it says.

Route mapping
-------------
- ``GET  /api/provenance/{cart_name}/status``       -> chain verdict, counts, availability
- ``POST /api/provenance/{cart_name}/for-results``  -> provenance + notes for a set of hits
- ``GET  /api/provenance/{cart_name}/notes``        -> notes this seat may see
- ``GET  /api/provenance/{cart_name}/what-changed`` -> what changed on what this seat watches

⭐⭐ WHY THERE IS NO "PROVENANCE" SCREEN, AND WHY `for-results` IS THE IMPORTANT ROUTE
`DESIGN-WAL-2026-09-02` section 8 considered three ways to surface this and rejected two:
a changelog window ("a log nobody reads") and a rolling strip on the Hot Stack ("ambient,
mostly irrelevant to the task in hand, trains the reader to skip it"). What it chose was
*the N most recent entries referencing a passage, returned WITH that passage* -- because
the reader cannot skip what is attached to the thing they asked for, and it answers the
question that actually matters at that moment, **should I trust this passage right now**,
rather than the one a dashboard answers, *what has been happening lately*.

⇒ `for-results` is that ruling as an endpoint. It takes the passages a search returned and
hands back provenance keyed by content so the UI can render it inline. The modules were
built with `for_results()` helpers for exactly this shape, so nothing here reformats or
re-queries per hit.

⛔ WRITES ARE NOT HERE, DELIBERATELY (Andy, 2026-11-01: "one thing at a time")
No note creation, no sharing, no acknowledging. Those need the write-path decisions --
which identity may annotate which cart, and what a revoked grant does to an existing note
-- and a read layer can ship and be useful without settling them. The MCP tools in membot
already write; this reads what they wrote.

⛔ EVERY ROUTE IS CART-SCOPED IN THE PATH, ON PURPOSE
`tests/test_route_guards.py` requires each route to declare a guard or carry a written
exemption. Putting the cart name in the path means every route here takes
`Depends(cart_guard.require_named_cart_read)` and needs no exemption -- including the POST,
whose payload is only the passage list. A provenance route that answered for a cart the
caller cannot mount would be a THIRD DOOR into content the mount gate refuses, and the
listing route learned that lesson the hard way on 2026-08-15: it enumerated every cart on
the box to everybody, and "luckily they can't be mounted" was the only thing saving it.

⚠ AN UNREADABLE RECORD IS NOT AN EMPTY ONE
Every response carries `provenance_available` and, when false, a reason. A route that
returned `[]` because membot was missing would turn "we cannot read the record" into
"there is no record", which is the one translation a provenance feature must never make.
Same distinction `note_store.load` draws between unreadable and empty.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import cart_guard
from . import provenance_link as link
from .auth import get_current_user
from .cartridge_io import find_cartridge_path, name_candidates
from .profiles import seat_id

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/provenance", tags=["provenance"])

# How many journal entries ride back with one passage. Small on purpose: this renders
# inline next to a search hit, and the question is "should I trust this", not "replay the
# history of this document".
ENTRIES_PER_PASSAGE = 3
MAX_PASSAGES_PER_CALL = 200


class ForResultsRequest(BaseModel):
    """Identify the hits to look up, EITHER by key or by text.

    ⭐ `keys` is the preferred path and the one the frontend uses: a search response now
    carries `content_key` on every result (`SearchResult.content_key`, added 2026-11-04),
    computed server-side from the engine's passage. Handing those straight back means the
    client never touches passage text for this purpose and cannot get it wrong.

    ⚠ `passages` remains for callers that have text and no key — an agent assembling its own
    context, a script, a test. It is NOT the fallback the UI should use: the UI has keys.
    """

    keys: list[str] = Field(
        default_factory=list,
        description="Content keys from SearchResult.content_key, passed back verbatim. "
                    "Preferred: the server computed them from the authoritative passage.",
    )
    passages: list[str] = Field(
        default_factory=list,
        description="FULL passage texts, for callers that have no key. ⛔ Must be the whole "
                    "body -- a key over a preview or any shortened text matches nothing, "
                    "returns zero rows, and raises nothing.",
    )


def _cart_path_or_404(cart_name: str) -> str:
    """Resolve a bare cart name to a path on disk.

    Uses `name_candidates` for the same reason the mount path does: a caller sends `foo`
    and the file on disk is `foo.cart.npz`. Guarding one spelling and loading another is
    the bug `resolve_named` exists to avoid, so the resolution order here matches it.
    """
    for cand in name_candidates(cart_name):
        p = find_cartridge_path(cand)
        if p:
            return p
    raise HTTPException(status_code=404, detail=f"no cart named {cart_name!r} on this server")


def _unavailable() -> dict:
    a = link.availability()
    return {"provenance_available": False, "reason": a["error"]}


def _seat(user) -> str:
    """This caller's seat, or '' for anonymous.

    ⛔ '' is not "no filter" -- `note_store` reads it as an unidentified caller and returns
    SHARED notes only. Fail-closed on the surveillance boundary: "I do not know who you
    are" must never widen to "here is everyone's private reasoning".
    """
    return seat_id(user) or ""


def _integrity(cart_path: str) -> dict:
    """The chain verdict for this cart's journal, in a shape a UI can render."""
    v = link.journal.verify(link.journal.journal_path_for(cart_path))
    return {
        "ok": bool(v.get("ok")),
        "verdict_line": v.get("verdict_line"),
        "entries": v.get("entries", 0),
        "first_break": v.get("first_break"),
        "hwm_present": v.get("hwm_present"),
    }


@router.get("/{cart_name}/status")
async def provenance_status(cart_name: str, request: Request,
                            user: dict | None = Depends(get_current_user),
                            _guard=Depends(cart_guard.require_named_cart_read)):
    """Is there a record for this cart, does it verify, and how much is in it.

    ⭐ The honest answer to "do you have provenance for this?" includes the case where a
    journal exists and is BROKEN, which is different from both "yes" and "no".
    """
    if not link.availability()["available"]:
        return _unavailable()

    cart_path = _cart_path_or_404(cart_name)
    integrity = _integrity(cart_path)
    notes = link.note_store.stats(cart_path)
    jstats = link.journal.stats(link.journal.journal_path_for(cart_path))

    return {
        "provenance_available": True,
        "cart": cart_name,
        "integrity": integrity,
        "journal": jstats,
        # ⚠ Counts only, and `visible_to_you` rather than a total, because a count of
        # private notes is itself a signal about what other people are thinking.
        "notes": {
            "readable": notes.get("readable", True),
            "visible_to_you": len(link.note_store.list_notes(cart_path, seat=_seat(user))),
            "by_kind": notes.get("by_kind", {}),
        },
    }


@router.post("/{cart_name}/for-results")
async def provenance_for_results(cart_name: str, payload: ForResultsRequest,
                                 request: Request,
                                 user: dict | None = Depends(get_current_user),
                                 _guard=Depends(cart_guard.require_named_cart_read)):
    """⭐⭐ THE ROUTE THE RESULT CARD CALLS. Provenance for a set of hits, keyed by content.

    Returns `by_key[content_key] = {provenance: [...], notes: [...]}` plus a parallel
    `keys` list in the same order the passages arrived, so the caller can map a hit to its
    entry without recomputing anything.

    ⛔ IF THE CHAIN IS BROKEN, THE JOURNAL ENTRIES ARE WITHHELD and `integrity.ok` is
    false. Entries read out of a record that may have been edited are not evidence, and a
    caller who cannot tell the difference will render them as though they were. Notes are
    still returned: a note is a human statement, not a claim about an unaltered history,
    and suppressing someone's explanation because a hash chain is damaged would lose the
    one thing that is still true.
    """
    if not link.availability()["available"]:
        return _unavailable()

    n_in = len(payload.keys) + len(payload.passages)
    if n_in > MAX_PASSAGES_PER_CALL:
        raise HTTPException(
            status_code=413,
            detail=f"{n_in} items; limit is {MAX_PASSAGES_PER_CALL} per call. "
                   f"A search page is tens of hits, not hundreds.",
        )

    cart_path = _cart_path_or_404(cart_name)
    seat = _seat(user)
    integrity = _integrity(cart_path)

    # ⭐ Keys first and verbatim — the server computed them, so there is nothing to redo.
    # `keys` and `passages` may both be supplied; the response's `keys` list stays parallel
    # to (keys + passages) in that order, so a caller can always map a hit to its entry.
    keys: list[str] = [str(k or "").strip() for k in payload.keys]
    for text in payload.passages:
        try:
            keys.append(link.content_key_for(text))
        except Exception as e:                         # noqa: BLE001
            # One unkeyable passage must not fail the page. An empty key is reported in
            # `keys` so the caller can see WHICH hit has no provenance rather than finding
            # a silently short list.
            log.warning("content_key failed for a passage: %s", type(e).__name__)
            keys.append("")

    wanted = [k for k in keys if k]

    notes_by_key: dict = {}
    if wanted:
        try:
            notes_by_key = link.note_store.for_results(cart_path, wanted, seat=seat) or {}
        except Exception as e:                         # noqa: BLE001
            log.error("note_store.for_results failed: %s", e)
            notes_by_key = {}

    prov_by_key: dict = {}
    if wanted and integrity["ok"]:
        jpath = link.journal.journal_path_for(cart_path)
        try:
            # `for_results` on the journal takes the targets it should match. Keys are what
            # the write path stamps onto `target.target_key` as of 2026-11-01; older
            # index-only entries simply will not match, which is a gap rather than an error.
            prov_by_key = link.journal.for_results(jpath, wanted,
                                                   n=ENTRIES_PER_PASSAGE) or {}
        except Exception as e:                         # noqa: BLE001
            log.error("journal.for_results failed: %s", e)
            prov_by_key = {}

    by_key = {}
    for k in wanted:
        by_key[k] = {
            "provenance": prov_by_key.get(k, []),
            "notes": notes_by_key.get(k, []),
        }

    return {
        "provenance_available": True,
        "cart": cart_name,
        "integrity": integrity,
        "entries_withheld": not integrity["ok"],
        "keys": keys,
        "by_key": by_key,
    }


@router.get("/{cart_name}/notes")
async def provenance_notes(cart_name: str, request: Request,
                           user: dict | None = Depends(get_current_user),
                           _guard=Depends(cart_guard.require_named_cart_read)):
    """Every note on this cart THIS SEAT may see: its own private ones plus all shared.

    ⛔ Never another seat's private notes. That boundary is the reason a note can be
    written candidly, and it is enforced in `note_store`, not here -- this route passes a
    seat and does no filtering of its own, so there is one implementation of the rule.
    """
    if not link.availability()["available"]:
        return _unavailable()

    cart_path = _cart_path_or_404(cart_name)
    data = link.note_store.load(cart_path)
    if not data.get("readable", True):
        # Unreadable is not empty. Say which it is.
        return {"provenance_available": True, "cart": cart_name,
                "notes_readable": False, "reason": data.get("error"), "notes": []}

    notes = link.note_store.list_notes(cart_path, seat=_seat(user))
    superseded = link.note_store.superseded_ids(notes)
    return {
        "provenance_available": True,
        "cart": cart_name,
        "notes_readable": True,
        "notes": [dict(n, superseded=(n.get("id") in superseded)) for n in notes],
    }


@router.get("/{cart_name}/what-changed")
async def provenance_what_changed(cart_name: str, request: Request, since: int = 0,
                                  user: dict | None = Depends(get_current_user),
                                  _guard=Depends(cart_guard.require_named_cart_read)):
    """What changed on the things this seat watches, and why.

    ⚠ Does NOT acknowledge anything. Reading is not acknowledging: advancing the seen-mark
    here would mean a client that failed midway through rendering lost those changes with
    nothing able to tell afterwards. Acknowledging is a write, and writes are not in this
    module yet.

    ⚠ An anonymous caller watches nothing, so this is honestly empty for them rather than
    falling back to "everything that changed" -- which would be a different feature wearing
    this one's name.
    """
    if not link.availability()["available"]:
        return _unavailable()

    cart_path = _cart_path_or_404(cart_name)
    seat = _seat(user)
    if not seat:
        return {"provenance_available": True, "cart": cart_name, "seat": None,
                "verdict_line": "Sign in to watch passages and see what changed.",
                "changes": [], "watching": []}

    res = link.subscriptions.what_changed(cart_path, seat, since_seq=since)
    # Pass the module's own verdict and integrity through unmodified. It already withholds
    # rows on a broken chain and says so; re-deciding that here would be a second opinion
    # on the same question.
    return {"provenance_available": True, **res}
