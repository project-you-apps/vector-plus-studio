"""Source-file link helpers — Phase A of the report drill-down UX.

Shared helpers for emitting ``vps://source/{slug}`` markdown links from
every Wave-1 report module. Landed 2026-07-13 as part of the "live
source-file links" pass.

Every place a report currently emits a source-file name in its markdown
output should route through :func:`source_link` so the rendered markdown
carries a real link the frontend can intercept. The frontend's
``ReportResultsView`` custom ``a`` component watches for the
``vps://source/{slug}`` scheme and dispatches a Search-tab focus action
instead of following the URL. Exports also carry the ``vps://`` link
syntax as-is — MD/HTML keep them, TXT strips markdown link syntax by
design (see ``frontend/src/lib/exportReport.ts``).

Slug rule (kept in ONE place so every report agrees):

- lowercase
- strip any leading ``Poem:`` / ``Doc:`` / ``PDF:`` prefix (case-insensitive)
- collapse runs of non-alphanumeric characters into a single ``-``
- collapse consecutive ``-``
- trim leading / trailing ``-``

Phase B (passage-level ``vps://passage/{pattern-id}`` links) is out of
scope for this pass. If we add it, extend this module rather than
introducing a second slug helper — one convention, one code path.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Slug rule
# ---------------------------------------------------------------------------

# Prefix labels that get stripped before slugging. Kept lowercase because
# the caller lowercases the source name first. If we grow this list,
# update the corresponding docstring rule above so downstream agents
# don't have to grep for the truth.
_STRIP_PREFIXES: tuple[str, ...] = ("poem:", "doc:", "pdf:")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DASH_RUN_RE = re.compile(r"-+")


def source_slug(source_name: Optional[str]) -> str:
    """Deterministic URL slug for a source-file name.

    Empty / whitespace / all-punctuation inputs return ``""`` so callers
    can decide whether to emit a link at all.

    Examples::

        source_slug("Poem: War Poems by Siegfried Sassoon")
        # → "war-poems-by-siegfried-sassoon"

        source_slug("invoice-may04.pdf")
        # → "invoice-may04-pdf"

        source_slug("PDF: 2026 Annual Report v2")
        # → "2026-annual-report-v2"
    """
    if not source_name:
        return ""
    s = source_name.strip()
    if not s:
        return ""
    lowered = s.lower()
    for prefix in _STRIP_PREFIXES:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):].strip()
            break
    lowered = _NON_ALNUM_RE.sub("-", lowered)
    lowered = _DASH_RUN_RE.sub("-", lowered)
    return lowered.strip("-")


# ---------------------------------------------------------------------------
# Markdown link renderer
# ---------------------------------------------------------------------------

def source_link(source_name: Optional[str], *, empty_label: str = "(no source)") -> str:
    """Full markdown link syntax for a source-file reference.

    Returns ``[{source_name}](vps://source/{slug})`` — a real markdown
    link the frontend intercepts. Display text preserves whatever prefix
    was in the caller-provided name (e.g. "Poem: ...") so the rendered
    surface reads the same as it did before Phase A landed.

    Empty / None inputs collapse to ``empty_label`` (default
    ``(no source)``) rather than an empty-link ``[](vps://source/)``, so
    caller sites can drop their own bracket-wrappers unconditionally.

    Weird inputs whose slug reduces to the empty string fall back to the
    plain display text — nobody wants a live-looking link that points at
    ``vps://source/``.

    Display-text escaping: ``]`` inside the display text would prematurely
    close the markdown link; ``|`` inside a GFM table cell would break the
    row. Both are backslash-escaped so unusual source names don't
    corrupt the output. ``\\`` itself is doubled first so the escape
    passes are order-independent.
    """
    if not source_name:
        return empty_label
    display = str(source_name).strip()
    if not display:
        return empty_label
    slug = source_slug(display)
    if not slug:
        # All-punctuation input — no useful link target. Emit the display
        # text as-is (escaped so table cells still parse).
        return _escape_display(display)
    return f"[{_escape_display(display)}](vps://source/{slug})"


def _escape_display(display: str) -> str:
    """Escape characters that would break markdown link / GFM table syntax."""
    return (
        display
        .replace("\\", "\\\\")
        .replace("]", "\\]")
        .replace("|", "\\|")
    )


__all__ = ["source_slug", "source_link"]
