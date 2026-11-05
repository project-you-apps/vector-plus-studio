"""Provenance routes: where a passage came from, and WHY it says what it says.

Route mapping
-------------
- ``GET  /api/provenance/{cart_name}/status``       -> chain verdict, counts, availability
- ``POST /api/provenance/{cart_name}/for-results``  -> provenance + notes for a set of hits
- ``GET  /api/provenance/{cart_name}/notes``        -> notes this seat may see
- ``POST /api/provenance/{cart_name}/notes``        -> ADD a note (append-only; no delete)
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

ⓘ WRITES ARRIVED 2026-11-04, AND ONLY ONE OF THEM
This module shipped read-only on 11-01 ("one thing at a time") because note creation
needed a decision this file could not make: WHICH IDENTITY MAY ANNOTATE WHICH CART. That
is now settled -- `annotate` is a real capability that `commenter` has and `viewer` does
not, so `cart_guard.require_named_cart_annotate` is the gate and the read guard is not.

⛔ STILL NOT HERE, AND NOT AN OVERSIGHT: there is **no delete and no edit**, by ruling
(Andy, 2026-11-04). Notes are permanent and appendable; a correction is a new note with
`supersedes`. Also absent: note SHARING (scope change after the fact) and what a REVOKED
grant does to an existing note -- the second is a real open question, because a note
written legitimately by a seat whose access was later removed is evidence, not a leak, and
deleting it to tidy up access control would destroy the record on purpose.

⛔ EVERY ROUTE IS CART-SCOPED IN THE PATH, ON PURPOSE
`tests/test_route_guards.py` requires each route to declare a guard or carry a written
exemption. Putting the cart name in the path means every route here can `Depends(...)` its
guard and needs no exemption -- including both POSTs, since neither names its cart in the
body. (The read routes take `require_named_cart_read`; the note POST takes
`require_named_cart_annotate`, which is strictly stronger.) A provenance route that answered for a cart the
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

from . import cart_access
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

# The author recorded for a note written on a studio with no auth configured at all.
#
# ⭐ A NAME, NOT AN EMPTY STRING, and the difference is load-bearing in two directions.
# `note_store` reads '' as "unidentified caller" and answers with SHARED notes only, so
# storing '' as the author would create a note whose own writer could not then see it if
# it were private. And on the reading side a named author makes the record legible: "this
# was written on a single-user install" is a fact worth keeping, where a blank author
# merely looks like data loss.
#
# ⛔ It is deliberately NOT a seat-shaped UUID, so it can never be mistaken for a verified
# identity, and it is paired with identity_source='claimed' everywhere it appears.
LOCAL_SEAT = "local-studio"


def cart_access_enforced() -> bool:
    """Does this deployment have an identity system whose answers could bind?

    Wrapped rather than called inline so the note route reads as a policy question, and
    so a test can monkeypatch one name. Thin on purpose: `enforcement_available` reads the
    environment at call time, and caching the answer here would freeze whichever state
    happened to exist at import.
    """
    return cart_access.enforcement_available()


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


class AddNoteRequest(BaseModel):
    """Attach a note to one passage, identified by key or by full text."""

    target_key: str = Field("", description="content_key of the passage. Preferred.")
    passage: str = Field("", description="FULL passage text, when the caller has no key. "
                                         "⛔ A key over a preview matches nothing.")
    kind: str = Field("why", description="why | context | caution | decision | prediction")
    body: str = Field(..., min_length=1, description="The note itself.")
    scope: str = Field("private", description="private (default) | shared")
    supersedes: str = Field("", description="id of a note this one replaces. The prior note "
                                            "is NEVER deleted; this records a link.")


@router.post("/{cart_name}/notes", status_code=201)
async def provenance_add_note(cart_name: str, payload: AddNoteRequest, request: Request,
                              user: dict | None = Depends(get_current_user),
                              guard=Depends(cart_guard.require_named_cart_annotate)):
    """Write a note. ⭐ THE FIRST WRITE ROUTE IN THIS MODULE (2026-11-04).

    ⭐⭐ GUARDED ON `annotate`, NOT `write`. `commenter` is exactly the level that may
    annotate and may not write, so the write guard would refuse the one role this route
    exists for. See cart_access.MountDecision.may_annotate for the capability table.

    ⛔ NOTES ARE APPEND-ONLY AND THERE IS NO DELETE ROUTE, BY RULING (Andy, 2026-11-04):
    "They shouldn't be deletable... the notes should be visible and appendable... so all
    the notes are permanent, just they can be added to." A correction is a NEW note
    carrying `supersedes`, which records an attributed link rather than mutating or
    removing anything. `superseded` is then DERIVED on read, never stored -- so the
    history of a disagreement survives the disagreement being settled.

    ⚠ THE BODY IS REDACTED ON THE WAY IN. A typed note is a new text ingress into the
    cart's sidecars, and the proven case is our own corpus, which already contains an API
    key and OAuth tokens because nothing stood between paste and store. Redaction belongs
    at the door and nowhere else (redact.redact_text's own warning), and this is a door.
    """
    if not link.availability()["available"]:
        raise HTTPException(status_code=503, detail=link.availability()["error"])

    # ── MAY THIS SERVER WRITE TO DISK AT ALL? ───────────────────────────────────────
    # ⭐⭐ THREE DIFFERENT THINGS SAY "READ-ONLY" IN THIS APP AND THEY GET DIFFERENT
    # ANSWERS HERE (Andy asked on 2026-11-04, looking at the header badge):
    #
    #   engine.read_only          the MOUNT LOCK -- a click-to-unlock safety catch on
    #                             cart CONTENT. ⇒ NOTES ALLOWED. A note is a sidecar, not
    #                             a passage. Blocking it would mean you must UNLOCK the
    #                             cart -- making every passage editable -- just to write a
    #                             comment. That forces a MORE dangerous state to do a
    #                             SAFER thing, which inverts the control.
    #   cart .permissions.json    per-cart "r". Also about cart content. ⇒ ALLOWED, same
    #                             reasoning; it collapses into the mount lock anyway.
    #   READ_ONLY_MODE            VPS_READ_ONLY, server-wide. ⇒ REFUSED, below. This one
    #                             is not about carts at all: the operator declared that
    #                             this process does not write to disk, and notes.json is
    #                             disk. An operator who set it did not consent to a
    #                             growing sidecar.
    #
    # ⇒ And an immutable record you may still annotate is the COMPLIANCE case, not a
    # compromise: audit workpapers are exactly "the record is sealed, your commentary is
    # appended alongside it". Read-only carts are where notes matter MOST.
    from .main import READ_ONLY_MODE                  # late: avoids a circular import
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail="This server runs read-only (VPS_READ_ONLY), so it does not write to "
                   "disk at all -- including note sidecars. Notes can be added on a "
                   "writable deployment of this cart.")

    cart_path = _cart_path_or_404(cart_name)

    # ── WHO is writing ──────────────────────────────────────────────────────────────
    # ⛔ An unattributed note is worse than no note: the whole value of the record is that
    # a statement has an author. But "anonymous" means two different things and they get
    # opposite answers, the same split cart_access draws between a denial and UNENFORCED:
    #
    #   enforcement available + no token -> there IS an identity system and this caller
    #                                       declined to use it. Refuse.
    #   no enforcement at all            -> local single-user studio. Allow, and let the
    #                                       RECORD carry the weakness as identity_source
    #                                       'claimed' rather than refusing a product that
    #                                       has nobody to restrict.
    seat = _seat(user)
    enforced = bool(getattr(guard, "enforced", False)) and getattr(guard, "level", None)
    if not seat:
        if cart_access_enforced():
            raise HTTPException(
                status_code=401,
                detail="Sign in to add a note. A provenance note records WHO said it, so "
                       "it cannot be written anonymously on a server that has sign-in.")
        seat = LOCAL_SEAT
    identity_source = "verified" if (seat and seat != LOCAL_SEAT) else "claimed"

    # ── WHICH passage ───────────────────────────────────────────────────────────────
    target_key = (payload.target_key or "").strip()
    if not target_key:
        if not payload.passage.strip():
            raise HTTPException(status_code=422,
                                detail="give either target_key or the full passage text")
        target_key = link.content_key_for(payload.passage)

    # ── Redact, then store ──────────────────────────────────────────────────────────
    clean, findings = link.redact.redact_text(payload.body, origin="")
    try:
        note = link.note_store.add_note(
            cart_path, target_key, payload.kind, clean,
            by=seat, surface="WEB", scope=payload.scope,
            supersedes=payload.supersedes,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except PermissionError as e:
        # Superseding a note this seat cannot see is a write into someone else's private
        # reasoning, which note_store refuses. 403, not 404: saying "no such note" would
        # leak whether it exists.
        raise HTTPException(status_code=403, detail=str(e)) from e

    # ── Journal it ──────────────────────────────────────────────────────────────────
    # ⛔ AFTER the store, never before: a journal entry for a note that failed to save
    # would make the chain assert something that did not happen. `op="note"` is a
    # deliberate member of journal.VALID_OPS -- it was silently rejected until a test
    # caught it, because _journal_event swallows its own failures by design.
    try:
        link.journal.append_to_cart(
            cart_path, writer=seat, surface="WEB", op="note",
            target={"cart": cart_name, "target_key": target_key},
            payload={"note_id": note.get("id"), "kind": payload.kind,
                     "scope": payload.scope, "chars": len(clean),
                     "identity_source": identity_source,
                     "redactions": len(findings),
                     "supersedes": payload.supersedes or None},
        )
    except Exception as e:                                          # noqa: BLE001
        # The note is saved. Say so loudly rather than failing the request: losing the
        # note because its audit line failed would be the worse of the two outcomes, and
        # a silent miss is the one thing this feature must not do.
        log.error("note %s stored but NOT journalled: %s: %s",
                  note.get("id"), type(e).__name__, e)

    return {
        "provenance_available": True,
        "cart": cart_name,
        "note": note,
        "target_key": target_key,
        "identity_source": identity_source,
        # ⚠ Surfaced so the author SEES that their text was altered. A redaction the
        # writer never learns about is content silently eaten.
        "redactions": [{"kind": f.get("kind"), "verdict": f.get("verdict")}
                       for f in findings],
        "enforced": bool(enforced),
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
