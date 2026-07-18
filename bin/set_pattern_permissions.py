#!/usr/bin/env python3
"""
set_pattern_permissions.py — flip per-pattern RWX bits in a cart's
hippocampus row 63 (Step 2b of the RWX roadmap).

Companion to bin/set_cart_permissions.py:
  • set_cart_permissions.py    — cart-wide default via .permissions.json sidecar
  • set_pattern_permissions.py — per-pattern bits in the cart's hippocampus

The per-pattern bits live in the `flags` byte (offset 28) of the 64-byte
hippocampus row. Bit layout:
  0x01 = R (read)
  0x02 = W (write)
  0x04 = X (reserved for future "executable" / "lambda passage")

When both layers exist, the cart-level default applies UNLESS the pattern's
flags byte explicitly grants additional permissions. Pattern bits cannot
override a cart-level lock (a read-only cart stays read-only).

Usage:
    # Lock specific patterns to read-only (rwx=0x01)
    python bin/set_pattern_permissions.py mycart.cart.npz --idx 5,12,42 --perms r

    # Mark a range writable
    python bin/set_pattern_permissions.py mycart.cart.npz --idx 0:100 --perms rw

    # Set all patterns to a specific value (use with care)
    python bin/set_pattern_permissions.py mycart.cart.npz --all --perms r
"""
import argparse
import os
import sys
from pathlib import Path

# Local import — bin/ is a sibling of api/, so add the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.cartridge_io import (  # noqa: E402
    write_hippocampus_perms, PERM_R, PERM_W, PERM_X,
)


def _parse_idx_spec(spec: str, n_patterns: int) -> list[int]:
    """Parse "5,12,42" or "0:100" or "5,12-15,42" into a list of indices."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk or "-" in chunk:
            sep = ":" if ":" in chunk else "-"
            lo_s, hi_s = chunk.split(sep, 1)
            lo = int(lo_s) if lo_s else 0
            hi = int(hi_s) if hi_s else n_patterns
            for i in range(lo, hi):
                out.add(i)
        else:
            out.add(int(chunk))
    return sorted(out)


def _parse_perms(s: str) -> int:
    """Parse 'r' / 'rw' / 'rwx' / '0x07' into a flags byte value."""
    s = s.strip().lower()
    if s.startswith("0x"):
        return int(s, 16) & 0xFF
    flags = 0
    for ch in s:
        if ch == "r":
            flags |= PERM_R
        elif ch == "w":
            flags |= PERM_W
        elif ch == "x":
            flags |= PERM_X
        elif ch in ("-", "_"):
            continue
        else:
            raise ValueError(f"Unknown permission character: {ch!r} (use r/w/x or 0xNN)")
    return flags


def main() -> int:
    p = argparse.ArgumentParser(description="Set per-pattern RWX bits in a cart's hippocampus row")
    p.add_argument("cart_path", help="Path to a .cart.npz file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", help="Comma-list and/or ranges (e.g. '5,12-15,42' or '0:100')")
    g.add_argument("--all", action="store_true", help="Apply to every pattern in the cart")
    p.add_argument("--perms", required=True,
                   help="Permissions to set: 'r', 'rw', 'rwx', or '0xNN'")
    args = p.parse_args()

    if not os.path.exists(args.cart_path):
        print(f"ERROR: cart not found: {args.cart_path}", file=sys.stderr)
        return 1

    flags = _parse_perms(args.perms)
    print(f"Setting flags = 0x{flags:02x} (R={bool(flags & PERM_R)} W={bool(flags & PERM_W)} X={bool(flags & PERM_X)})")

    # Resolve indices — need to know n_patterns first
    import numpy as np
    data = np.load(args.cart_path, allow_pickle=True)
    if "hippocampus" not in data.files:
        print(f"ERROR: cart has no hippocampus array: {args.cart_path}", file=sys.stderr)
        return 1
    n_patterns = data["hippocampus"].shape[0]
    data.close()

    if args.all:
        indices = list(range(n_patterns))
    else:
        indices = _parse_idx_spec(args.idx, n_patterns)

    if not indices:
        print("No indices selected.", file=sys.stderr)
        return 1

    idx_to_flags = {i: flags for i in indices}
    updated = write_hippocampus_perms(args.cart_path, idx_to_flags)
    print(f"Updated {updated} pattern(s) in {args.cart_path}")
    if len(indices) <= 20:
        print(f"  indices: {indices}")
    else:
        print(f"  indices: {indices[:5]} … {indices[-3:]}  ({len(indices)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
