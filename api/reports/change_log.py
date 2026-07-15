"""Change Log report — diff two cart snapshots.

Wave-1 (no LLM) implementation of the Change Log report described in
``docs/vps-internal/Report Types Design 2026-07-10.md`` §7. Answers the
"what changed between last week's cart and this week's cart?" question
without hitting an LLM: two strategies (exact text match, or embedding
cosine ≥ 0.92) categorize each pattern as added / removed / modified /
unchanged and the output aggregates by source_path.

Design notes worth calling out:

- The Report ABC hands ``generate()`` a single ``cart_path`` but Change
  Log needs two carts. We treat ``cart_path`` as the NEW cart's hint
  and resolve ``cart_id_old`` from the same directory (see
  :func:`_resolve_cart_path`). Both cart ids are user-supplied form
  values from :attr:`ReportInput.raw`; the frontend definition in
  ``report-definitions.ts`` names them ``cart_id_old`` / ``cart_id_new``.
- Tombstoned patterns on BOTH sides are skipped up front via
  :py:meth:`CartHandle.is_tombstoned`. Otherwise a pattern tombstoned in
  the NEW cart would look like a removal when it is really a suppression
  — the semantics differ (a tombstone means "I hid this", a removal
  means "I never had this").
- The semantic strategy is O(N × M). For carts of a few hundred to a
  couple thousand patterns this is well under a second, but Grant's
  Sysco carts approach 10k and paired snapshots of a personal Heartbeat
  cart could go much higher. We hard-cap the pairwise matrix at
  ``_SEMANTIC_MAX_SIDE = 5000`` per side for Wave 1 and surface a
  warning if we clipped; the design doc explicitly authorizes the cap
  and defers a streaming / batched implementation to a later wave.
- The cosine threshold ``0.92`` is copied verbatim from the design doc.
  Do not tune it here — that choice is a spec question, not a code
  question, and drifts in the threshold need to move in lockstep with
  the "modified" definition (a matched-embedding + differing-text pair).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import Report, ReportInput, ReportOptions, ReportOutput
from .cart_reader import CartHandle
from .registry import register_report
from .source_link import source_link


# ---------------------------------------------------------------------------
# Tuning constants (per §7 of the design doc)
# ---------------------------------------------------------------------------

# Cosine threshold above which two embeddings are considered "the same
# pattern" for the semantic diff strategy. Value comes from the design
# doc; changing it changes the definition of "modified" so do not touch
# without a spec update.
_SEMANTIC_COSINE_THRESHOLD = 0.92

# Hard cap on the per-side count for the pairwise similarity matrix.
# 5000 × 5000 float32 = ~100 MB — big but tolerable on the target dev
# machine. Anything larger gets clipped with a warning; Wave-2 will
# replace this with a batched / ANN implementation.
_SEMANTIC_MAX_SIDE = 5000

# Structural-difference warning fires when the two carts differ in size
# by more than this ratio. 0.5 = one side has ≥50% more patterns than
# the other. Empirical — feel free to loosen later.
_STRUCTURAL_SIZE_RATIO = 0.5

# Rebuild-artifact detection: if this fraction (or more) of the new-cart
# patterns match into the old cart under the chosen strategy AND the raw
# embedding-array fingerprints differ, we flag "carts appear to be
# rebuilds, not content changes." Tighter than "identical" (which is
# 1.0) so re-embeddings with minor churn still trigger the note.
_REBUILD_MATCH_RATIO = 0.98

# Character budget for the "short_summary" line in each output entry.
# Chosen to fit in one wrapped terminal line without truncating the
# source label + pattern index tail.
_SHORT_SUMMARY_CHARS = 80

# Before/after excerpt length in the Modified section. Wider than the
# short summary because the caller is specifically asking "what
# changed" and needs enough context to see the diff.
_MODIFIED_EXCERPT_CHARS = 160

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _PatternRef:
    """Compact per-pattern record for the diff pass.

    Kept small on purpose — we hold one of these per non-tombstoned
    pattern on each side of the diff, and for large carts even a handful
    of extra fields adds up.
    """

    idx: int
    text: str
    source: str


def _live_refs(cart: CartHandle) -> list[_PatternRef]:
    """Enumerate non-tombstoned patterns as ``_PatternRef`` records.

    Order is preserved from the cart so the "pattern #N" indices in the
    output line up with what the user sees in the cart browser.
    """
    return [
        _PatternRef(idx=i, text=cart.get_passage(i), source=cart.get_source(i))
        for i in range(cart.count)
        if not cart.is_tombstoned(i)
    ]


def _short_summary(text: str, max_chars: int = _SHORT_SUMMARY_CHARS) -> str:
    """Trim ``text`` to ~``max_chars`` at a word boundary.

    Empty / whitespace-only text collapses to "(empty passage)" so the
    output stays readable when the source is a brain-only cart. We prefer
    a trailing whitespace break when one is available inside the budget;
    if not, hard-cut with an ellipsis.

    Passage text on disk occasionally carries stray unpaired UTF-16
    surrogates (byproduct of upstream capture from mixed-encoding
    sources). Left in place they crash any downstream encoder — round-
    trip through UTF-8 with ``surrogatepass`` disabled to scrub them.
    """
    # Scrub unpaired surrogates before anything else so downstream
    # renderers / loggers can encode the result.
    if text:
        text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    # Collapse newlines to spaces so the summary sits on one line.
    flat = " ".join(text.split())
    if not flat:
        return "(empty passage)"
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    # Prefer the last whitespace boundary within budget so we don't slice
    # a word in half. Only pull back to that boundary if it's not too
    # aggressive (i.e. still yields a meaningful snippet).
    space = cut.rfind(" ")
    if space >= max_chars // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"


def _resolve_cart_path(cart_id: str, hint_path: str) -> Optional[str]:
    """Resolve a user-supplied cart identifier to an absolute file path.

    Resolution strategy (in order):

    1. If ``cart_id`` is already an existing absolute or relative path,
       return the absolute form.
    2. Look in the same directory as ``hint_path`` (the primary cart
       the executor handed us) for ``{cart_id}`` or ``{cart_id}.cart.npz``.
    3. Give up and return ``None``.

    Returning ``None`` (rather than raising) lets the caller surface a
    friendly "cart not found" message via the report's warnings + a
    markdown body that names the missing side.
    """
    if not cart_id:
        return None
    # (1) Try as-is.
    if os.path.exists(cart_id):
        return os.path.abspath(cart_id)
    # (2) Sibling of the hint cart. Handle both bare-name and
    # already-suffixed inputs so "banana-boat" and "banana-boat.cart.npz"
    # both resolve.
    if hint_path:
        parent = os.path.dirname(os.path.abspath(hint_path))
        candidates = [
            os.path.join(parent, cart_id),
            os.path.join(parent, cart_id + ".cart.npz"),
            os.path.join(parent, cart_id + ".npz"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return None


def _fingerprint(embeddings: np.ndarray) -> str:
    """Compact SHA-16 of the embeddings buffer — mirrors the shape used
    by ``api/cartridge_io.py::compute_cartridge_fingerprint`` so the
    rebuild-artifact check is comparable across the codebase."""
    import hashlib

    if embeddings.size == 0:
        return "empty"
    first = embeddings[0].tobytes()
    last = embeddings[-1].tobytes() if len(embeddings) > 1 else first
    combined = first + last + str(len(embeddings)).encode()
    return hashlib.sha256(combined).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Diff strategies
# ---------------------------------------------------------------------------

def _diff_exact(
    old_refs: list[_PatternRef],
    new_refs: list[_PatternRef],
) -> tuple[list[_PatternRef], list[_PatternRef], list[tuple[_PatternRef, _PatternRef]], int]:
    """Match patterns by identical passage text.

    Returns ``(added, removed, modified, unchanged_count)``. Exact mode
    never produces "modified" — matching text is unchanged, non-matching
    text is a fresh add / removal.

    Uses a multiset per side so duplicate passages (which do happen in
    text-only rebuilds where the same source paragraph shows up in two
    chunks) are handled without double-crediting one side.
    """
    # Bucket old refs by text; pop as we consume matches so duplicates
    # on the new side are still matched 1:1.
    old_by_text: dict[str, list[_PatternRef]] = {}
    for ref in old_refs:
        old_by_text.setdefault(ref.text, []).append(ref)

    added: list[_PatternRef] = []
    unchanged = 0
    for ref in new_refs:
        bucket = old_by_text.get(ref.text)
        if bucket:
            bucket.pop()
            if not bucket:
                del old_by_text[ref.text]
            unchanged += 1
        else:
            added.append(ref)

    # Anything left in old_by_text was never matched → removed.
    removed = [ref for bucket in old_by_text.values() for ref in bucket]
    return added, removed, [], unchanged


def _diff_semantic(
    old_refs: list[_PatternRef],
    new_refs: list[_PatternRef],
    old_emb: np.ndarray,
    new_emb: np.ndarray,
) -> tuple[
    list[_PatternRef],
    list[_PatternRef],
    list[tuple[_PatternRef, _PatternRef]],
    int,
    bool,
]:
    """Match patterns by embedding cosine ≥ ``_SEMANTIC_COSINE_THRESHOLD``.

    Returns ``(added, removed, modified, unchanged_count, capped)``.

    Matching is greedy: for each new pattern we take the best old pattern
    still available, then remove it from the pool. Not globally optimal
    (that's the assignment problem — Hungarian is O(n^3)), but good
    enough for Wave 1 and behaves sanely on realistic diffs where each
    old pattern has one obvious new counterpart.

    ``capped`` is True when we clipped either side to
    :data:`_SEMANTIC_MAX_SIDE` to keep memory bounded — the caller
    surfaces the warning.
    """
    capped = False
    old_slice = old_refs
    new_slice = new_refs
    old_emb_slice = old_emb
    new_emb_slice = new_emb
    if len(old_refs) > _SEMANTIC_MAX_SIDE:
        capped = True
        old_slice = old_refs[:_SEMANTIC_MAX_SIDE]
        old_emb_slice = old_emb[:_SEMANTIC_MAX_SIDE]
    if len(new_refs) > _SEMANTIC_MAX_SIDE:
        capped = True
        new_slice = new_refs[:_SEMANTIC_MAX_SIDE]
        new_emb_slice = new_emb[:_SEMANTIC_MAX_SIDE]

    # Normalize once upfront so the pairwise cosine is a plain dot
    # product. Guard against zero-norm rows (embedding for an empty
    # passage) so we don't divide by zero.
    old_norm = np.linalg.norm(old_emb_slice, axis=1, keepdims=True)
    new_norm = np.linalg.norm(new_emb_slice, axis=1, keepdims=True)
    old_norm[old_norm == 0] = 1.0
    new_norm[new_norm == 0] = 1.0
    old_u = (old_emb_slice / old_norm).astype(np.float32)
    new_u = (new_emb_slice / new_norm).astype(np.float32)

    # Pairwise cosine: [n_new, n_old]. Contiguous float32 keeps this
    # around ~n_new * n_old * 4 bytes.
    sims = new_u @ old_u.T

    # Greedy best-match: for each new pattern, find its argmax over
    # still-available old indices, accept iff cosine >= threshold.
    old_available = np.ones(len(old_slice), dtype=bool)
    added: list[_PatternRef] = []
    modified: list[tuple[_PatternRef, _PatternRef]] = []
    unchanged = 0
    matched_old_idx: set[int] = set()

    for i, new_ref in enumerate(new_slice):
        # Mask unavailable old indices with -inf so they can't win the argmax.
        row = np.where(old_available, sims[i], -np.inf)
        best = int(np.argmax(row))
        best_score = float(row[best])
        if best_score >= _SEMANTIC_COSINE_THRESHOLD:
            old_available[best] = False
            matched_old_idx.add(best)
            old_ref = old_slice[best]
            if old_ref.text == new_ref.text:
                unchanged += 1
            else:
                modified.append((old_ref, new_ref))
        else:
            added.append(new_ref)

    removed = [ref for j, ref in enumerate(old_slice) if j not in matched_old_idx]

    # Anything past the cap on the new side is unaccounted for; treat it
    # as "added" so the count is honest, but the capped warning tells the
    # user the comparison is incomplete.
    if len(new_refs) > len(new_slice):
        added.extend(new_refs[len(new_slice):])
    if len(old_refs) > len(old_slice):
        removed.extend(old_refs[len(old_slice):])

    return added, removed, modified, unchanged, capped


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_added(added: list[_PatternRef]) -> str:
    """Bulleted list of added patterns, sorted by source then index."""
    if not added:
        return ""
    lines = [f"## Added ({len(added)})"]
    for ref in sorted(added, key=lambda r: (r.source, r.idx)):
        lines.append(
            f"- {source_link(ref.source)}: {_short_summary(ref.text)} "
            f"(pattern #{ref.idx})"
        )
    return "\n".join(lines)


def _render_removed(removed: list[_PatternRef]) -> str:
    """Bulleted list of removed patterns, sorted by source then index."""
    if not removed:
        return ""
    lines = [f"## Removed ({len(removed)})"]
    for ref in sorted(removed, key=lambda r: (r.source, r.idx)):
        lines.append(
            f"- {source_link(ref.source)}: {_short_summary(ref.text)} "
            f"(was pattern #{ref.idx})"
        )
    return "\n".join(lines)


def _render_modified(modified: list[tuple[_PatternRef, _PatternRef]]) -> str:
    """Bulleted before/after list, sorted by NEW-side source then index."""
    if not modified:
        return ""
    lines = [f"## Modified ({len(modified)})"]
    for old_ref, new_ref in sorted(modified, key=lambda pair: (pair[1].source, pair[1].idx)):
        lines.append(f"- {source_link(new_ref.source)}: {_short_summary(new_ref.text)}")
        lines.append(f"  - **Before**: {_short_summary(old_ref.text, _MODIFIED_EXCERPT_CHARS)}")
        lines.append(f"  - **After**: {_short_summary(new_ref.text, _MODIFIED_EXCERPT_CHARS)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@register_report
class ChangeLogReport(Report):
    """§7 Change Log — pattern-level diff of two cart snapshots."""

    name = "change_log"
    display_name = "Change Log"
    description = "What changed between two cart snapshots — added / removed / modified."
    llm_dependency = False
    # Comparison of specific snapshots, not a recurring brief — the
    # scheduler surface would need a "which two carts" resolution
    # policy that's out of scope for Wave 1.
    supports_scheduling = False

    # Mirrors the ``change_log`` entry in
    # ``frontend/src/reports/report-definitions.ts`` verbatim. Wave-2
    # will auto-generate the frontend from the backend describe() call,
    # at which point this stays the source of truth.
    input_schema: list[dict[str, Any]] = [
        {
            "name": "cart_id_old",
            "label": "Old cart",
            "type": "text",
            "required": True,
            "placeholder": "cart name or path (older snapshot)",
            "helpText": "The \"before\" cart to diff from.",
        },
        {
            "name": "cart_id_new",
            "label": "New cart",
            "type": "text",
            "required": True,
            "placeholder": "cart name or path (newer snapshot)",
            "helpText": "The \"after\" cart to diff to.",
        },
        {
            "name": "diff_strategy",
            "label": "Diff strategy",
            "type": "select",
            "required": True,
            "default": "semantic",
            "options": ["exact", "semantic"],
            "helpText": "exact = string-match passages; semantic = embedding cosine >= 0.92.",
        },
    ]

    def generate(
        self,
        cart_path: str,
        inputs: ReportInput,
        options: ReportOptions,
    ) -> ReportOutput:
        cart_id_old = inputs.get_str("cart_id_old") or ""
        cart_id_new = inputs.get_str("cart_id_new") or ""
        strategy = (inputs.get_str("diff_strategy") or "semantic").lower()
        if strategy not in ("exact", "semantic"):
            strategy = "semantic"

        # ---- Resolve the two cart paths --------------------------------
        # cart_path from the executor is the hint anchor: probably matches
        # cart_id_new, but we don't force it — the user may have entered
        # two explicit paths.
        old_path = _resolve_cart_path(cart_id_old, cart_path)
        new_path = _resolve_cart_path(cart_id_new, cart_path) or (
            os.path.abspath(cart_path) if cart_path and os.path.exists(cart_path) else None
        )

        warnings: list[str] = []

        if not old_path or not new_path:
            missing_sides = []
            if not old_path:
                missing_sides.append(f"cart_id_old={cart_id_old!r}")
            if not new_path:
                missing_sides.append(f"cart_id_new={cart_id_new!r}")
            msg = (
                "Cart not found: " + ", ".join(missing_sides)
                + f". Tried the value verbatim and as a sibling of {cart_path!r}."
            )
            warnings.append(msg)
            markdown = (
                "# Change Log — cart not found\n\n"
                f"{msg}\n\n"
                "Enter an absolute path, or a filename that lives in the "
                "same directory as the primary cart the executor was "
                "handed. Example: `banana-boat.cart.npz`."
            )
            return ReportOutput(
                markdown=markdown,
                metadata={"strategy": strategy, "old_path": old_path, "new_path": new_path},
                warnings=warnings,
            )

        # ---- Load both carts -------------------------------------------
        try:
            old_cart = CartHandle(old_path)
        except (FileNotFoundError, ValueError) as exc:
            markdown = (
                f"# Change Log — could not load old cart\n\n"
                f"Failed to open {old_path!r}: {exc}"
            )
            return ReportOutput(
                markdown=markdown,
                metadata={"strategy": strategy, "old_path": old_path, "new_path": new_path},
                warnings=[str(exc)],
            )
        try:
            new_cart = CartHandle(new_path)
        except (FileNotFoundError, ValueError) as exc:
            markdown = (
                f"# Change Log — could not load new cart\n\n"
                f"Failed to open {new_path!r}: {exc}"
            )
            return ReportOutput(
                markdown=markdown,
                metadata={"strategy": strategy, "old_path": old_path, "new_path": new_path},
                warnings=[str(exc)],
            )

        # Surface cart-loader shape warnings if either side flagged
        # them at load time (e.g. mismatched parallel-array lengths).
        for w in old_cart.length_warnings:
            warnings.append(f"(old cart) {w}")
        for w in new_cart.length_warnings:
            warnings.append(f"(new cart) {w}")

        # ---- Enumerate non-tombstoned patterns -------------------------
        # Skip tombstones on BOTH sides so a suppression in NEW doesn't
        # masquerade as a removal (and vice versa on OLD).
        old_refs = _live_refs(old_cart)
        new_refs = _live_refs(new_cart)
        n_old, n_new = len(old_refs), len(new_refs)

        # ---- Run the chosen strategy -----------------------------------
        capped = False
        if strategy == "exact":
            added, removed, modified, unchanged = _diff_exact(old_refs, new_refs)
        else:
            # Slice embeddings to match the non-tombstoned refs so the
            # index alignment carried in _PatternRef stays sound.
            old_emb = old_cart.embeddings[[ref.idx for ref in old_refs]] if old_refs else np.zeros((0, 1), dtype=np.float32)
            new_emb = new_cart.embeddings[[ref.idx for ref in new_refs]] if new_refs else np.zeros((0, 1), dtype=np.float32)
            # Dim mismatch would kill the dot product — surface as a
            # hard warning and fall back to exact.
            if old_emb.size and new_emb.size and old_emb.shape[1] != new_emb.shape[1]:
                warnings.append(
                    f"Embedding dims differ (old={old_emb.shape[1]}, "
                    f"new={new_emb.shape[1]}); falling back to exact match."
                )
                added, removed, modified, unchanged = _diff_exact(old_refs, new_refs)
                strategy = "exact"
            else:
                added, removed, modified, unchanged, capped = _diff_semantic(
                    old_refs, new_refs, old_emb, new_emb
                )
                if capped:
                    warnings.append(
                        f"Semantic diff capped at {_SEMANTIC_MAX_SIDE}×{_SEMANTIC_MAX_SIDE}; "
                        f"patterns beyond the cap are reported as added/removed. "
                        f"For carts this size a streaming diff is a Wave-2 upgrade."
                    )

        # ---- Structural / rebuild-artifact heuristics ------------------
        structural_note: Optional[str] = None
        if n_old and n_new:
            ratio = abs(n_new - n_old) / max(n_new, n_old)
            if ratio > _STRUCTURAL_SIZE_RATIO:
                structural_note = (
                    f"**Structural note**: the two carts differ in size by "
                    f"{ratio * 100:.0f}% ({n_old} → {n_new} live patterns). "
                    f"Change Log below is at pattern granularity; if the "
                    f"content model changed (chunking, source set), that "
                    f"explains most of the churn."
                )

        rebuild_note: Optional[str] = None
        old_fp = _fingerprint(old_cart.embeddings)
        new_fp = _fingerprint(new_cart.embeddings)
        matched = unchanged + len(modified)
        denom = max(n_new, n_old, 1)
        if (
            matched / denom >= _REBUILD_MATCH_RATIO
            and old_fp != new_fp
            and (added or removed or modified)
        ):
            rebuild_note = (
                "**Rebuild note**: these carts appear to be rebuilds of the "
                "same source content, not content changes. Nearly every "
                "pattern matched under the chosen strategy, but the raw "
                "embeddings differ (fingerprints "
                f"`{old_fp}` → `{new_fp}`). Small \"modified\" deltas below "
                "are likely re-chunking or re-embedding artifacts."
            )

        # ---- Assemble markdown -----------------------------------------
        header = f"# Change Log: {old_cart.cart_name} → {new_cart.cart_name}"
        subheader_lines = [
            f"Strategy: **{strategy}**"
            + (f" (cosine ≥ {_SEMANTIC_COSINE_THRESHOLD})" if strategy == "semantic" else "")
            + f" • {n_old} → {n_new} live patterns "
            + f"(tombstoned skipped: {old_cart.count - n_old} old, "
            + f"{new_cart.count - n_new} new)"
        ]

        # Zero-changes short-circuit per the design doc.
        if not added and not removed and not modified:
            body = "These carts are functionally identical."
        else:
            sections = [_render_added(added), _render_modified(modified), _render_removed(removed)]
            body = "\n\n".join(s for s in sections if s)

        parts = [header, "", "\n".join(subheader_lines)]
        if structural_note:
            parts.extend(["", structural_note])
        if rebuild_note:
            parts.extend(["", rebuild_note])
        parts.extend(["", body])
        markdown = "\n".join(parts).rstrip() + "\n"

        # ---- Metadata --------------------------------------------------
        metadata: dict[str, Any] = {
            "strategy": strategy,
            "old_path": old_path,
            "new_path": new_path,
            "old_cart_name": old_cart.cart_name,
            "new_cart_name": new_cart.cart_name,
            "old_count": old_cart.count,
            "new_count": new_cart.count,
            "old_live": n_old,
            "new_live": n_new,
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
            "unchanged": unchanged,
            "old_fingerprint": old_fp,
            "new_fingerprint": new_fp,
            "semantic_cap": _SEMANTIC_MAX_SIDE if strategy == "semantic" else None,
            "semantic_capped": capped,
        }

        if options.include_source_refs:
            metadata["added_source_refs"] = [ref.idx for ref in added]
            metadata["removed_source_refs"] = [ref.idx for ref in removed]
            metadata["modified_source_refs"] = [
                {"old_idx": old.idx, "new_idx": new.idx} for old, new in modified
            ]

        return ReportOutput(markdown=markdown, metadata=metadata, warnings=warnings)


__all__ = ["ChangeLogReport"]
