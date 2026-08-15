"""Clicking a file in the Pattern-0 TOC must find that file's passages.

THE BUG (Andy, 2026-08-14). The TOC listed a file as "6p" and clicking it said "No passages
found for this file. 0 active / 0 total". The count and the drill-down disagreed about the
same six passages, which is the shape that says two code paths are answering one question
differently.

The TOC counts by `source_hash` in the h-row. `/api/patterns?source=` matched by reading the
FIRST LINE of each passage, assuming the chunker prepended `<filename>` there. That holds for
LocalCart and the wiki carts. It does not hold for anything built by build_office.py, whose
chunks begin with prose -- so the filter compared a filename against a sentence, and no cart
of that shape could ever drill down.

⚠ THIS TESTS AGAINST THE REAL CARTS, not a fixture. A fixture would have encoded the same
assumption the code did -- I would have written label lines into it, because that is what I
believed carts looked like, and it would have passed while the product stayed broken. The
carts on disk are the only witness that disagrees.
"""

import asyncio
import json
import pathlib

import numpy as np
import pytest

from api import main
from api.cartridge_io import parse_hippocampus

CARTS = pathlib.Path(__file__).resolve().parents[2] / "cartridges"


def _carts_with_a_named_toc():
    """Real carts that ship a Pattern-0 TOC of filenames. Skips if none are present."""
    found = []
    for path in sorted(CARTS.glob("*.cart.npz")):
        try:
            with np.load(path, allow_pickle=True) as z:
                if "pattern0" not in z.files or "hippocampus" not in z.files:
                    continue
                toc = json.loads(str(z["pattern0"]))
                if toc.get("files"):
                    found.append(path)
        except Exception:                                   # noqa: BLE001
            continue
    return found


CARTS_WITH_TOC = _carts_with_a_named_toc()
needs_carts = pytest.mark.skipif(
    not CARTS_WITH_TOC, reason="no built carts on this machine; nothing to dogfood against")


@needs_carts
@pytest.mark.parametrize("cart_path", CARTS_WITH_TOC, ids=lambda p: p.name)
def test_every_toc_entry_finds_exactly_the_passages_it_claims(cart_path, monkeypatch):
    """THE REGRESSION, stated as the invariant: the TOC's count IS the drill-down's count.

    Every file in every real cart, through the real route. The TOC and the filter are two
    code paths answering one question, and the bug was that nothing ever asked them the
    same question and compared.
    """
    with np.load(cart_path, allow_pickle=True) as z:
        toc = json.loads(str(z["pattern0"]))
        passages = [str(p) for p in z["passages"]]
        hippo = parse_hippocampus(z)

    assert hippo, f"{cart_path.name} has no parseable hippocampus"

    _stand_up_engine(monkeypatch, passages, hippo)

    mismatches = []
    for entry in toc["files"]:
        found = _drill_down(entry["name"]).total
        if found != entry["chunks"]:
            mismatches.append(f"{entry['name']}: TOC says {entry['chunks']}p, filter finds {found}")

    assert not mismatches, (
        f"{cart_path.name}: the TOC and the drill-down disagree, so clicking these files "
        f"shows nothing or the wrong thing:\n  " + "\n  ".join(mismatches[:10]))


def _stand_up_engine(monkeypatch, passages, hippo):
    """Give the route the cart state it reads, and nothing else.

    A stand-in rather than a real mount: this is about how a source name RESOLVES, and
    booting the embedder to prove it would make a fast test slow and a focused one broad.
    """
    class _FakeEngine:
        pass

    fake = _FakeEngine()
    fake.passages = passages
    fake.hippocampus = hippo
    fake.deleted_ids = set()
    monkeypatch.setattr(main, "engine", fake)


def _drill_down(source_name: str, limit: int = 25):
    """Call THE ACTUAL ROUTE. ⚠ Do not reimplement the ladder here.

    The first draft of this file counted matches with its own copy of the matching rules,
    which would have gone on passing if the route lost a rung -- the exact "tests prove
    logic, not shape" trap. The route's `Depends(...)` defaults are inert when it is called
    directly, and its body never reads them.
    """
    return asyncio.run(main.list_patterns(offset=0, limit=limit, q=None, source=source_name))


def test_the_route_still_uses_the_full_ladder():
    """All four rungs present in the route. Losing the hash rung silently re-breaks
    every cart whose chunks are not labelled, and the symptom is an empty panel rather
    than an error."""
    import inspect

    src = inspect.getsource(main.list_patterns)
    for rung in ("engine_source_paths", "source_path", "_label_line", "source_hash"):
        assert rung in src, f"the source-matching ladder lost its {rung!r} rung"


@needs_carts
def test_a_name_that_is_in_no_cart_matches_nothing(monkeypatch):
    """The hash rung must not turn an unknown name into results."""
    with np.load(CARTS_WITH_TOC[0], allow_pickle=True) as z:
        passages = [str(p) for p in z["passages"]]
        hippo = parse_hippocampus(z)

    _stand_up_engine(monkeypatch, passages, hippo)
    assert _drill_down("no-such-file-anywhere.txt").total == 0


@needs_carts
def test_a_labelled_cart_still_resolves_by_its_label_line(monkeypatch):
    """⚠ THE RUNG THE FIX COULD HAVE BROKEN. LocalCart and the wiki carts prepend
    `<filename>` to each chunk, and that path worked before today. Adding the hash rung
    below it must leave it exactly as it was."""
    labelled = next((p for p in sorted(CARTS.glob("*.cart.npz"))
                     if p.name.startswith("wiki_")), None)
    if labelled is None:
        pytest.skip("no labelled cart on this machine")

    with np.load(labelled, allow_pickle=True) as z:
        passages = [str(p) for p in z["passages"]]
        hippo = parse_hippocampus(z) or []

    _stand_up_engine(monkeypatch, passages, hippo)
    label = passages[0].split("\n")[0].strip()
    assert _drill_down(label).total >= 1, f"label-line resolution regressed for {label!r}"
