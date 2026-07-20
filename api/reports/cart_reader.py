"""Cart-file reader shared across report modules.

Every report needs a way to walk the patterns of a mounted cart without
re-implementing the NPZ-loading dance. :class:`CartHandle` is that
shared substrate: load once on construction, cache the derived views,
expose read-only helpers.

Cart format (``.cart.npz``) supported keys, adapted from the actual
carts Andy has on disk (Cart Builder GUI + membot cartridge_builder.py
formats — see ``api/cartbuilder/*.py`` for writers):

- ``embeddings`` (float32, [n, dim])
- ``passages`` (unicode object array, [n])  OR
- ``compressed_texts`` (zlib-blob object array, [n]) fallback
- ``source_paths`` (unicode array, [n]) — per-pattern source filename
- ``hippocampus`` (uint8, [n, 64]) — HIPPO_FORMAT rows; parsed via
  ``api.cartridge_io.parse_hippocampus``
- ``pattern0`` (unicode 0-d array of JSON, OR uint8 header bytes) —
  cart-level metadata. See ``_parse_pattern0_from_npz`` in main.py for
  the two live shapes; we support the JSON-string flavor here (Cart
  Builder GUI carts). Uint8-header-only carts fall through to ``None``.
- ``per_pattern_meta`` (unicode 0-d array of JSON list) — per-pattern
  extended metadata (image_b64, table_json, tags). Optional.

Thread-safety: after ``__init__`` the arrays are numpy views into a
closed NPZ file (we ``.copy()`` at load time so the underlying handle
can be released) and all mutation stays on the caller side. Reports do
read-only work, so concurrent ``get_passage`` calls across threads are
safe.
"""
from __future__ import annotations

import json
import os
import zlib
from typing import Any, Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Hippocampus flags — canonical layout
# ---------------------------------------------------------------------------

# Offset of the flags byte within the 64-byte hippocampus row (see
# ``api/cartridge_io.py::HIPPO_FORMAT``). Bit 0 of that byte is the
# tombstone marker (``FLAG_TOMBSTONE`` in
# ``api/cartbuilder/cartridge_builder.py``).
_HIPPO_FLAGS_OFFSET = 28
_FLAG_TOMBSTONE = 0x01

# Byte 30 = lifecycle byte (truth_status + URGENT/IMPORTANT/TO-CONSOLIDATE).
# Imported from cartridge_io so the layout stays single-sourced.
from api.cartridge_io import (
    LIFECYCLE_BYTE_OFFSET as _LIFECYCLE_BYTE_OFFSET,
    TRUTH_STATUS_MASK as _TRUTH_STATUS_MASK,
    TRUTH_STATUS_ACTIVE as _TRUTH_STATUS_ACTIVE,
    LIFECYCLE_FLAG_URGENT as _LIFECYCLE_FLAG_URGENT,
    LIFECYCLE_FLAG_IMPORTANT as _LIFECYCLE_FLAG_IMPORTANT,
    LIFECYCLE_FLAG_TO_CONSOLIDATE as _LIFECYCLE_FLAG_TO_CONSOLIDATE,
    TRUTH_STATUS_NAMES as _TRUTH_STATUS_NAMES,
    parse_lifecycle_byte as _parse_lifecycle_byte,
    PERMS_BYTE_OFFSET_READ as _PERMS_BYTE_OFFSET,
    PERM_R as _PERM_R,
    FORMAT_VERSION_PROVENANCE as _FORMAT_VERSION_PROVENANCE,
)

# H-row byte offsets shared across v1/v2/v3 (see HIPPO_FORMAT_V* in
# api/cartridge_io.py). format_version at byte 4 discriminates the layout;
# timestamp lives at bytes 24-27 (uint32 LE) in all versions.
_HROW_FORMAT_VERSION_OFFSET = 4
_HROW_TIMESTAMP_OFFSET = 24
# v3 h-row: bytes 18-19 = source_idx uint16 LE (byte 20-21 reserved).
_HROW_V3_SOURCE_IDX_OFFSET = 18


# ---------------------------------------------------------------------------
# CartHandle
# ---------------------------------------------------------------------------

class CartHandle:
    """Loaded, in-memory handle to a ``.cart.npz`` file.

    Report modules construct this once at the top of ``generate()`` and
    use the accessors below to walk patterns. All heavy I/O (NPZ decode,
    JSON parse of pattern0 / per_pattern_meta, compressed-text
    inflation) happens at construction; per-passage access is O(1).

    Attributes are private-by-convention (leading underscore); consumers
    should use the ``get_*`` / ``iter_*`` helpers to stay future-proof
    against shape changes.
    """

    def __init__(self, cart_path: str):
        """Load ``cart_path`` and populate the in-memory views.

        Raises :class:`FileNotFoundError` if the path doesn't exist,
        :class:`ValueError` if the NPZ is present but missing the
        ``embeddings`` array (that's the one field we treat as required
        — no embeddings means no cart, period).
        """
        if not os.path.exists(cart_path):
            raise FileNotFoundError(f"Cart file not found: {cart_path}")

        self._cart_path = cart_path
        # Cart name = filename stem, with the double-suffix stripped.
        base = os.path.basename(cart_path)
        for suffix in (".cart.npz", ".npz", ".pkl", ".cart.pkl"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        self._cart_name = base

        # Load NPZ, materialize what we need, close the handle. `.copy()`
        # decouples us from the zip archive so the handle can be released
        # without dangling numpy views.
        with np.load(cart_path, allow_pickle=True) as data:
            files = set(data.files)

            if "embeddings" not in files:
                raise ValueError(
                    f"Cart {cart_path!r} has no 'embeddings' array; "
                    f"present keys: {sorted(files)}"
                )
            self._embeddings: np.ndarray = np.asarray(data["embeddings"]).copy()

            # Passages: prefer the plain-text 'passages' array. Fall back
            # to 'compressed_texts' (zlib-blob per pattern) if only the
            # compressed form is present — this is how the split-cart /
            # cartridge_builder.py format lays it out.
            self._passages: list[str] = []
            if "passages" in files:
                arr = data["passages"]
                # Object/unicode arrays deserialize element-by-element.
                self._passages = [str(x) for x in arr.tolist()]
            elif "compressed_texts" in files:
                arr = data["compressed_texts"]
                inflated: list[str] = []
                for blob in arr.tolist():
                    try:
                        if isinstance(blob, (bytes, bytearray)):
                            inflated.append(zlib.decompress(blob).decode("utf-8", errors="replace"))
                        elif hasattr(blob, "tobytes"):
                            inflated.append(zlib.decompress(bytes(blob)).decode("utf-8", errors="replace"))
                        else:
                            # np.void wraps the zlib blob; str(blob) is unusable —
                            # coerce through bytes(). If that fails, leave placeholder.
                            inflated.append("")
                    except Exception:
                        inflated.append("")
                self._passages = inflated
            else:
                # No text at all — this is a brain-only cart. Reports that
                # need text can check ``count > 0 and get_passage(0) == ""``
                # and warn the user gracefully.
                self._passages = ["" for _ in range(len(self._embeddings))]

            # Hippocampus (H-block per pattern) — optional. Only decode if
            # present; otherwise every ``get_meta`` call falls back to {}.
            # Loaded BEFORE source_paths so the v3 resolution below can peek
            # at the format_version byte + source_idx per row.
            self._hippocampus_raw: Optional[np.ndarray] = None
            if "hippocampus" in files:
                self._hippocampus_raw = np.asarray(data["hippocampus"]).copy()

            # Source paths — per-pattern source filename. Resolution order:
            #   1. v3-native: ``source_strings`` (dedup table) + h-row
            #      ``source_idx`` (bytes 18-19 uint16 LE). Preferred when the
            #      cart's h-row format_version = FORMAT_VERSION_PROVENANCE.
            #      Retires the sidecar dependency for v3 carts.
            #   2. v1/v2 sidecar: ``source_paths.npy`` unicode array.
            #      Backward-compat for older browser-built carts + carts that
            #      were re-emitted with both surfaces during the v3 transition.
            #   3. Empty — falls back to first-line parsing at ``get_source``
            #      call time (Cart Builder GUI carts embed source name as
            #      line 0 of each passage).
            self._source_paths: list[str] = []
            v3_resolved = False
            if ("source_strings" in files
                    and self._hippocampus_raw is not None
                    and self._hippocampus_raw.shape[0] > 0
                    and self._hippocampus_raw.shape[1] > _HROW_V3_SOURCE_IDX_OFFSET + 1):
                try:
                    fmt_ver = int(self._hippocampus_raw[0, _HROW_FORMAT_VERSION_OFFSET])
                    if fmt_ver == _FORMAT_VERSION_PROVENANCE:
                        strings_table = [str(x) for x in data["source_strings"].tolist()]
                        lo = self._hippocampus_raw[:, _HROW_V3_SOURCE_IDX_OFFSET].astype(np.uint32)
                        hi = self._hippocampus_raw[:, _HROW_V3_SOURCE_IDX_OFFSET + 1].astype(np.uint32)
                        source_idx = (lo | (hi << 8)).tolist()
                        derived: list[str] = []
                        for idx in source_idx:
                            if 0 <= idx < len(strings_table):
                                derived.append(strings_table[idx])
                            else:
                                derived.append("")
                        self._source_paths = derived
                        v3_resolved = True
                except Exception:
                    # Fall through to sidecar path on any parse failure.
                    v3_resolved = False
            if not v3_resolved and "source_paths" in files:
                self._source_paths = [str(x) for x in data["source_paths"].tolist()]

            # Per-pattern ingestion timestamp (uint32 Unix epoch) — h-row
            # bytes 24-27 in all format versions (v1/v2/v3). Populated at
            # cart-build time; represents when the pattern was embedded,
            # not necessarily when its source file was created. Surface
            # via ``get_timestamp`` / ``all_timestamps``. Empty list when
            # no hippocampus present (older carts).
            self._timestamps: list[int] = []
            if (self._hippocampus_raw is not None
                    and self._hippocampus_raw.shape[0] > 0
                    and self._hippocampus_raw.shape[1] > _HROW_TIMESTAMP_OFFSET + 3):
                try:
                    b0 = self._hippocampus_raw[:, _HROW_TIMESTAMP_OFFSET + 0].astype(np.uint32)
                    b1 = self._hippocampus_raw[:, _HROW_TIMESTAMP_OFFSET + 1].astype(np.uint32)
                    b2 = self._hippocampus_raw[:, _HROW_TIMESTAMP_OFFSET + 2].astype(np.uint32)
                    b3 = self._hippocampus_raw[:, _HROW_TIMESTAMP_OFFSET + 3].astype(np.uint32)
                    self._timestamps = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).tolist()
                except Exception:
                    self._timestamps = []

            # pattern0 — JSON cart-level metadata. Two live shapes:
            #   1. Unicode NPY (Cart Builder GUI) — JSON string, load via json.loads
            #   2. Uint8 header bytes (membot cartridge_builder.py) — binary,
            #      returns None here (reports that need it can call
            #      main.py::_parse_pattern0_from_npz directly).
            self._pattern0: Optional[dict[str, Any]] = None
            if "pattern0" in files:
                self._pattern0 = _decode_pattern0(data["pattern0"])

            # per_pattern_meta — JSON list, one dict per pattern. Optional.
            # Mirrors the shape written by api/cartbuilder/builder.py.
            self._per_pattern_meta: Optional[list[dict[str, Any]]] = None
            if "per_pattern_meta" in files:
                self._per_pattern_meta = _decode_per_pattern_meta(data["per_pattern_meta"])

        # Sanity: parallel-array lengths should match embeddings count when
        # present. Log-only (via metadata warnings — reports surface it) —
        # don't hard-fail so weird historic carts stay openable.
        self._length_warnings: list[str] = []
        n = len(self._embeddings)
        if self._passages and len(self._passages) != n:
            self._length_warnings.append(
                f"passages length {len(self._passages)} != embeddings {n}"
            )
        if self._source_paths and len(self._source_paths) != n:
            self._length_warnings.append(
                f"source_paths length {len(self._source_paths)} != embeddings {n}"
            )

    # -- properties -------------------------------------------------------
    @property
    def cart_path(self) -> str:
        """Absolute path this handle was constructed from."""
        return self._cart_path

    @property
    def cart_name(self) -> str:
        """Filename stem (``foo.cart.npz`` → ``foo``). Reports that also
        surface Pattern-0's ``cart_name`` should prefer that over this,
        since the stem may not match what the creator titled the cart."""
        return self._cart_name

    @property
    def count(self) -> int:
        """Number of patterns in the cart."""
        return int(self._embeddings.shape[0])

    @property
    def pattern0(self) -> Optional[dict[str, Any]]:
        """Cart-level metadata (creator, description, files, etc.) or
        ``None`` if the cart has no Pattern-0 / has a binary-header
        Pattern-0 we can't decode from JSON."""
        return self._pattern0

    @property
    def embeddings(self) -> np.ndarray:
        """Raw embeddings array (float32, shape [n, dim]). Reports doing
        semantic diff / clustering (Change Log, Summary theme cluster)
        reach for this directly."""
        return self._embeddings

    @property
    def length_warnings(self) -> list[str]:
        """Non-fatal shape warnings detected at load. Reports should
        surface these via ``ReportOutput.warnings`` when non-empty."""
        return list(self._length_warnings)

    # -- per-pattern accessors -------------------------------------------
    def get_passage(self, idx: int) -> str:
        """Return the raw text of pattern ``idx``. Empty string if the
        cart is brain-only (no passages / compressed_texts arrays)."""
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Pattern idx {idx} out of range [0, {self.count})")
        if idx >= len(self._passages):
            return ""
        return self._passages[idx]

    def get_source(self, idx: int) -> str:
        """Return the source filename for pattern ``idx``.

        Prefers the ``source_paths`` array. Falls back to parsing the
        first line of the passage text — Cart Builder carts prepend
        ``<filename>`` (optionally with " (part N/M)") as line 0. Empty
        string if neither surface has it.
        """
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Pattern idx {idx} out of range [0, {self.count})")
        if self._source_paths and idx < len(self._source_paths):
            return self._source_paths[idx]
        # Fallback: first line of passage. This mirrors the label-line
        # parsing in main.py's per-source filter (line 1917).
        text = self.get_passage(idx)
        if not text:
            return ""
        first_line = text.split("\n", 1)[0].strip()
        # Strip " (part N/M)" suffix if present.
        if " (part " in first_line:
            first_line = first_line.split(" (part ", 1)[0].strip()
        return first_line

    def get_timestamp(self, idx: int) -> int:
        """Return the h-row ingestion timestamp (uint32 Unix epoch) for
        pattern ``idx``, or ``0`` if unavailable (no hippocampus, or the
        h-row was never populated with a timestamp).

        Populated by the writer at cart-build time; represents when the
        pattern was embedded, not necessarily when the source file was
        created or last modified. For batch-built carts (CLI
        ``cartridge_builder.py``), every pattern in one build shares the
        same timestamp (batch-scope). Incremental writers may populate
        per-pattern granular timestamps.
        """
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Pattern idx {idx} out of range [0, {self.count})")
        if not self._timestamps or idx >= len(self._timestamps):
            return 0
        return int(self._timestamps[idx])

    def get_meta(self, idx: int) -> dict[str, Any]:
        """Return the per-pattern metadata record for ``idx``, or ``{}``
        if the cart has no ``per_pattern_meta`` sidecar.

        Record shape mirrors ``api/cartbuilder/builder.py``; expected
        keys include ``content_type``, ``image_b64``, ``table_json``,
        ``tags``, ``chunk_index``. Reports treat all keys as optional.
        """
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Pattern idx {idx} out of range [0, {self.count})")
        if not self._per_pattern_meta or idx >= len(self._per_pattern_meta):
            return {}
        return self._per_pattern_meta[idx]

    def is_tombstoned(self, idx: int) -> bool:
        """True iff pattern ``idx`` is tombstoned.

        Reads bit 0 (``FLAG_TOMBSTONE``) of the flags byte at offset 28
        of the 64-byte hippocampus row. Returns ``False`` when the cart
        has no hippocampus array or the row is too short to carry a
        flags byte — legacy Cart Builder GUI / brain-only carts pre-date
        the tombstone convention and read as fully live.

        Raises :class:`IndexError` if ``idx`` is out of range on a cart
        that DOES carry a hippocampus (mirrors
        :py:meth:`get_hippocampus_row`).
        """
        row = self.get_hippocampus_row(idx)
        if row is None or len(row) <= _HIPPO_FLAGS_OFFSET:
            return False
        try:
            return bool(int(row[_HIPPO_FLAGS_OFFSET]) & _FLAG_TOMBSTONE)
        except (IndexError, TypeError, ValueError):
            return False

    # -- lifecycle byte (byte 30) — truth_status + behavioral flags ------
    # Hot Stack preliminary — 2026-07-17. See
    # ``docs/vps-internal/hot-stack-preliminary-2026-07-17-recap.md``.
    # Every accessor here returns the "default / safe" value when the cart's
    # hippocampus row is absent or shorter than 31 bytes, matching the
    # tombstone idiom: legacy carts read as ACTIVE with no flags set, so
    # existing behavior is completely preserved.

    def _lifecycle_byte(self, idx: int) -> int:
        """Return the raw byte at offset 30 for pattern ``idx``, or 0 if
        the cart has no hippocampus / the row is too short. Used by every
        higher-level lifecycle accessor below."""
        row = self.get_hippocampus_row(idx)
        if row is None or len(row) <= _LIFECYCLE_BYTE_OFFSET:
            return 0
        try:
            return int(row[_LIFECYCLE_BYTE_OFFSET]) & 0xFF
        except (IndexError, TypeError, ValueError):
            return 0

    def truth_status(self, idx: int) -> int:
        """Return the 3-bit truth_status enum value for pattern ``idx``.

        Values: 0=ACTIVE, 1=SUPERSEDED, 2=INCORRECT, 3=DEFERRED, 4=ONGOING.
        Values 5-7 are reserved. Missing byte → ACTIVE (the safe default).
        """
        return self._lifecycle_byte(idx) & _TRUTH_STATUS_MASK

    def truth_status_name(self, idx: int) -> str:
        """Return the canonical uppercase name of pattern ``idx``'s
        truth_status ("ACTIVE" / "SUPERSEDED" / ...). Reserved enum
        values 5-7 render as "UNKNOWN"."""
        return _TRUTH_STATUS_NAMES.get(self.truth_status(idx), "UNKNOWN")

    def is_urgent(self, idx: int) -> bool:
        """True iff pattern ``idx`` has the URGENT behavioral flag set."""
        return bool(self._lifecycle_byte(idx) & _LIFECYCLE_FLAG_URGENT)

    def is_important(self, idx: int) -> bool:
        """True iff pattern ``idx`` has the IMPORTANT behavioral flag set."""
        return bool(self._lifecycle_byte(idx) & _LIFECYCLE_FLAG_IMPORTANT)

    def is_to_consolidate(self, idx: int) -> bool:
        """True iff pattern ``idx`` has the TO-CONSOLIDATE flag set."""
        return bool(self._lifecycle_byte(idx) & _LIFECYCLE_FLAG_TO_CONSOLIDATE)

    def is_active(self, idx: int) -> bool:
        """Convenience: True iff pattern ``idx`` is safe to surface in
        the default "current relevant patterns" retrieval — not
        tombstoned AND truth_status == ACTIVE.

        This is the check the ``pattern_filter="active_only"`` mode in
        AgentsEngine uses. Consumers who need finer-grained filtering
        (e.g. "include DEFERRED but exclude SUPERSEDED") should compose
        ``is_tombstoned`` + ``truth_status`` themselves.
        """
        return (
            not self.is_tombstoned(idx)
            and self.truth_status(idx) == _TRUTH_STATUS_ACTIVE
        )

    def lifecycle(self, idx: int) -> dict:
        """Return the full parsed lifecycle byte for pattern ``idx`` as
        a dict (see :func:`api.cartridge_io.parse_lifecycle_byte`).

        Shape: ``{"truth_status": int, "truth_status_name": str,
        "urgent": bool, "important": bool, "to_consolidate": bool,
        "raw": int}``. Missing byte → all defaults (ACTIVE, no flags).
        """
        return _parse_lifecycle_byte(self._lifecycle_byte(idx))

    # -- perms byte (byte 29) — Step 2b permissions ---------------------
    # See ``api/cartridge_io.py`` for the full bit layout including the
    # reserved Commenter (bit 3) / Manager (bit 4) tiers.

    def is_readable(self, idx: int) -> bool:
        """True iff pattern ``idx`` is readable (PERM_R bit set).

        Mirrors the tombstone / is_active idiom: missing hippocampus /
        short row / zero perms_byte all read as True (legacy carts
        pre-Step-2b defaulted to R+W). Only an explicit non-zero
        perms_byte with PERM_R unset excludes the pattern from search
        results. Closes the invariant that ``PERM_R = 0`` should hide
        a pattern from ``retrieve_top_patterns`` (see
        ``api/agents/retrieval.py:_should_include``).
        """
        row = self.get_hippocampus_row(idx)
        if row is None or len(row) <= _PERMS_BYTE_OFFSET:
            return True  # legacy / brain-only cart — treat as readable
        try:
            perms = int(row[_PERMS_BYTE_OFFSET]) & 0xFF
        except (IndexError, TypeError, ValueError):
            return True
        if perms == 0:
            return True  # perms_byte zero → PERM_DEFAULT_LEGACY (R+W)
        return bool(perms & _PERM_R)

    def get_hippocampus_row(self, idx: int) -> Optional[np.ndarray]:
        """Return the raw 64-byte hippocampus row for ``idx`` (uint8
        array of length 64), or ``None`` if the cart has no
        hippocampus.

        Reports that need the parsed struct (flags, source_hash, prev /
        next, perms) should call ``api.cartridge_io.parse_hippocampus``
        on the full array instead; this getter is here for reports that
        want raw bytes without importing that module."""
        if self._hippocampus_raw is None:
            return None
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Pattern idx {idx} out of range [0, {self.count})")
        return self._hippocampus_raw[idx].copy()

    # -- bulk enumerators -------------------------------------------------
    def all_source_paths(self) -> list[str]:
        """Return a per-pattern list of source filenames.

        Length == ``count``. Uses the ``source_paths`` array if present,
        else falls back to per-pattern first-line parsing (via
        ``get_source``). Reports enumerating unique sources should
        dedup on the caller side; Summary + Financial Rollup are the
        two downstream consumers.
        """
        if self._source_paths and len(self._source_paths) == self.count:
            return list(self._source_paths)
        return [self.get_source(i) for i in range(self.count)]

    def unique_sources(self) -> list[str]:
        """Return unique source filenames in insertion order.

        Convenience wrapper for the common Summary Report shape.
        """
        seen: dict[str, None] = {}
        for s in self.all_source_paths():
            if s and s not in seen:
                seen[s] = None
        return list(seen.keys())

    def iter_passages(self) -> Iterator[tuple[int, str, str]]:
        """Yield ``(idx, passage_text, source_path)`` for every pattern.

        The workhorse enumerator for reports that scan every passage
        (Timeline, Trend, Entity Rollup, Financial Rollup). Skip
        tombstoned patterns externally if needed — this iterator is
        state-agnostic.
        """
        for i in range(self.count):
            yield i, self.get_passage(i), self.get_source(i)


# ---------------------------------------------------------------------------
# Helpers — pattern0 + per_pattern_meta decoders
# ---------------------------------------------------------------------------

def _decode_pattern0(arr: np.ndarray) -> Optional[dict[str, Any]]:
    """Decode a ``pattern0`` NPZ entry to a dict, if possible.

    Mirrors ``main.py::_parse_pattern0_from_npz`` for the JSON-string
    shape (Cart Builder GUI carts). Binary-header shapes return
    ``None`` — reports needing them should call the main.py helper.
    """
    if arr.dtype.kind in ("U", "O"):
        try:
            raw = arr.item() if arr.ndim == 0 else arr[0]
            payload = json.loads(str(raw))
            if isinstance(payload, dict):
                return payload
        except (ValueError, TypeError):
            return None
    # Binary uint8 header shapes: we skip here rather than duplicating
    # the 4096-byte CartridgeHeader unpack. Report modules that need it
    # can call main.py's parser directly.
    return None


def _decode_per_pattern_meta(arr: np.ndarray) -> Optional[list[dict[str, Any]]]:
    """Decode a ``per_pattern_meta`` NPZ entry to a list of dicts.

    Mirrors ``main.py::_parse_per_pattern_meta_from_npz``.
    """
    if arr.dtype.kind in ("U", "O"):
        try:
            raw = arr.item() if arr.ndim == 0 else arr[0]
            payload = json.loads(str(raw))
            if isinstance(payload, list):
                return payload
        except (ValueError, TypeError):
            return None
    return None


__all__ = ["CartHandle"]
