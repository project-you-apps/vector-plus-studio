"""Migrate a .cart.npz cart from v1/v2 (source_hash) to v3 (source_idx + strings table).

Idempotent — running twice on a v3 cart is a no-op.

Recovery policy:
- v2 cart with ``source_paths.npy`` sidecar → deduplicate into
  ``source_strings.npy`` + rewrite h-row source_idx (byte 18).
- v2 cart WITHOUT ``source_paths.npy`` → try ``per_pattern_meta`` JSON's
  ``source`` field per pattern; error out if neither is present (we can't
  invent provenance we don't have).
- v1 cart with format_version=1 rows: same logic; the wire struct at bytes
  18-21 was uint32 source_hash in both v1 and v2, so recovery source is
  identical.

Preserves everything else: embeddings, passages, compressed_texts,
pattern0, per_pattern_meta, permissions sidecar, manifest fingerprint.

Usage:

    python bin/migrate_cart_v1_v2_to_v3.py path/to/cart.cart.npz [--in-place]

Without ``--in-place`` the migrated cart is written to
``path/to/cart.v3.cart.npz`` and the original is preserved. With
``--in-place`` the original file is overwritten (a ``.bak-YYYYMMDD-HHMM``
backup is written alongside first).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


HIPPO_SIZE = 64
HIPPO_FORMAT_V1_V2 = '<I B B I I I I H I B B 34s'
HIPPO_FORMAT_V3    = '<I B B I I I H H H I B B 34s'

FORMAT_VERSION_LEGACY     = 1
FORMAT_VERSION_CANONICAL  = 2
FORMAT_VERSION_PROVENANCE = 3


def detect_format_version(hippo_rows: np.ndarray) -> int:
    """Return the format_version byte of the first hippocampus row."""
    if hippo_rows.shape[0] == 0:
        return FORMAT_VERSION_CANONICAL  # arbitrary; empty cart migrates trivially
    return int(hippo_rows[0, 4])


def gather_source_paths(cart_data: dict, n_passages: int) -> list[str] | None:
    """Recover per-pattern source paths from the best available surface.

    Preference order:
    1. ``source_paths.npy`` sidecar (v1 provenance) — canonical when present
    2. ``per_pattern_meta[i]['source']`` — fallback for pre-sidecar carts
    3. None (caller errors out)
    """
    # 1. v1 sidecar
    if "source_paths" in cart_data.files:
        try:
            arr = cart_data["source_paths"]
            paths = [str(x) for x in arr.tolist()]
            if len(paths) == n_passages:
                return paths
        except Exception:
            pass

    # 2. per_pattern_meta JSON
    if "per_pattern_meta" in cart_data.files:
        try:
            raw = cart_data["per_pattern_meta"]
            payload = raw.item() if raw.ndim == 0 else raw[0]
            records = json.loads(str(payload))
            if isinstance(records, list) and len(records) == n_passages:
                paths = [str(rec.get("source", "")) for rec in records]
                if any(paths):
                    return paths
        except Exception:
            pass

    return None


def dedupe_strings(paths: list[str]) -> tuple[list[str], list[int]]:
    """Return (strings_table, idx_per_pattern). Index 0 is reserved for "no source"."""
    table: list[str] = [""]  # sentinel at index 0
    lookup: dict[str, int] = {}
    idx_per_pattern: list[int] = []
    for p in paths:
        if p not in lookup:
            lookup[p] = len(table)
            table.append(p)
        idx_per_pattern.append(lookup[p])
    if len(table) > 0xFFFF:
        raise ValueError(
            f"Source strings table has {len(table)} entries; exceeds uint16 cap."
        )
    return table, idx_per_pattern


def rewrite_hippo_row_v1v2_to_v3(row: np.ndarray, new_source_idx: int) -> bytes:
    """Repack a single 64-byte row: replace uint32 source_hash with uint16 idx + uint16 reserved."""
    old_vals = struct.unpack(HIPPO_FORMAT_V1_V2, row.tobytes())
    (
        pattern_id, format_version, cartridge_type,
        parent_ptr, child_ptr, sibling_ptr,
        source_hash_ignored,  # dropped
        sequence_num, timestamp, flags, perms_byte, reserved,
    ) = old_vals
    return struct.pack(
        HIPPO_FORMAT_V3,
        pattern_id,
        FORMAT_VERSION_PROVENANCE,  # bump format_version
        cartridge_type,
        parent_ptr, child_ptr, sibling_ptr,
        new_source_idx & 0xFFFF,
        0,                          # reserved uint16
        sequence_num,
        timestamp,
        flags, perms_byte,
        reserved,
    )


def migrate_cart(input_path: Path, output_path: Path, in_place: bool = False) -> dict:
    """Migrate a cart in a fresh .npz. Returns a summary dict."""
    with np.load(input_path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
        source_paths = gather_source_paths(data, len(data["passages"]))

    if "hippocampus" not in arrays:
        raise ValueError(f"{input_path} has no hippocampus array; can't migrate")

    hippo = arrays["hippocampus"]  # shape (N, 64)
    current_version = detect_format_version(hippo)
    n = hippo.shape[0]

    summary = {
        "input": str(input_path),
        "n_rows": n,
        "input_version": current_version,
    }

    if current_version == FORMAT_VERSION_PROVENANCE:
        summary["action"] = "no-op (already v3)"
        summary["output"] = str(input_path)
        return summary

    if source_paths is None:
        raise ValueError(
            f"{input_path} has no source_paths.npy sidecar AND no per_pattern_meta "
            f"provenance; cannot migrate to v3 without inventing provenance."
        )

    # Deduplicate and rewrite.
    strings_table, idx_per_pattern = dedupe_strings(source_paths)
    new_hippo = np.zeros_like(hippo)
    for i in range(n):
        row_bytes = rewrite_hippo_row_v1v2_to_v3(hippo[i], idx_per_pattern[i])
        new_hippo[i] = np.frombuffer(row_bytes, dtype=np.uint8)
    arrays["hippocampus"] = new_hippo
    arrays["source_strings"] = np.array(strings_table, dtype=object)

    # v3 keeps source_paths sidecar for backward-compat with older loaders.
    # (Loader prefers strings table; sidecar is a graceful fallback.)

    # Pattern 0 header: if the cart shipped a pattern0.npy blob, its inline
    # 64-byte hippocampus struct also needs its format_version bumped.
    # Skipped here — pattern0 body is JSON in most carts, and headerless
    # carts don't have a hippo-shaped pattern0. Migration tool for those
    # cases is a follow-up if the fleet needs it.

    if in_place:
        ts = time.strftime("%Y%m%d-%H%M")
        backup = input_path.with_suffix(f".npz.bak-{ts}")
        os.replace(input_path, backup)
        np.savez_compressed(input_path, **arrays)
        summary["output"] = str(input_path)
        summary["backup"] = str(backup)
    else:
        np.savez_compressed(output_path, **arrays)
        summary["output"] = str(output_path)

    summary["action"] = "migrated"
    summary["n_unique_sources"] = len(strings_table) - 1  # subtract sentinel

    # Also refresh the manifest if it exists.
    manifest_path = input_path.parent / (input_path.stem.replace(".cart", "") + ".cart_manifest.json")
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["hippo_format_version"] = FORMAT_VERSION_PROVENANCE
            manifest["has_source_strings"] = True
            manifest["n_source_strings"] = len(strings_table)
            manifest["migrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            summary["manifest_updated"] = str(manifest_path)
        except Exception as e:
            summary["manifest_error"] = str(e)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cart", type=Path, help="Path to .cart.npz to migrate")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input cart (a .bak-YYYYMMDD-HHMM backup is written first)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>.v3.cart.npz). Ignored with --in-place.",
    )
    args = parser.parse_args()

    if not args.cart.exists():
        print(f"ERROR: cart file not found: {args.cart}", file=sys.stderr)
        return 1

    output_path = args.output or args.cart.with_name(
        args.cart.stem.replace(".cart", "") + ".v3.cart.npz"
    )

    try:
        summary = migrate_cart(args.cart, output_path, in_place=args.in_place)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
