#!/usr/bin/env python3
"""
set_cart_permissions.py — write a `.permissions.json` sidecar next to a cart.

Step 2a tool from the RWX roadmap. Used to retrofit
existing carts (e.g. on the droplet) with read-only permissions for the
public demo, without rebuilding the cart.

Usage:
    python bin/set_cart_permissions.py <cart-path> --default r [--owner andy] [--description "..."]

Examples:
    # Mark every .cart.npz in /opt/membot/cartridges/ as read-only:
    for c in /opt/membot/cartridges/*.cart.npz; do
        python bin/set_cart_permissions.py "$c" --default r --owner andy
    done

    # Allow writes on a personal cart:
    python bin/set_cart_permissions.py mycart.cart.npz --default rw

The sidecar is written next to the cart with the same basename:
    mycart.cart.npz  →  mycart.permissions.json
"""
import argparse
import json
import os
import sys


def _permissions_path_for(cart_path: str) -> str:
    """Mirror cartridge_io._permissions_path_for so this CLI runs standalone."""
    base = cart_path
    for suffix in (".cart.npz", ".cart.pkl", ".npz", ".pkl", "_brain.npy"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    else:
        base = os.path.splitext(cart_path)[0]
    return base + ".permissions.json"


def main() -> int:
    p = argparse.ArgumentParser(description="Write cart .permissions.json sidecar")
    p.add_argument("cart_path", help="Path to a cart file (e.g. mycart.cart.npz)")
    p.add_argument("--default", required=True, choices=["r", "rw", "rwx"],
                   help="Cart-wide default permission")
    p.add_argument("--owner", default=None, help="Optional owner name (foundation for Step 3 user-ACL)")
    p.add_argument("--description", default=None, help="Optional human-readable note")
    args = p.parse_args()

    if not os.path.exists(args.cart_path):
        print(f"ERROR: cart not found: {args.cart_path}", file=sys.stderr)
        return 1

    payload = {"default": args.default, "version": "1.0"}
    if args.owner:
        payload["owner"] = args.owner
    if args.description:
        payload["description"] = args.description

    out_path = _permissions_path_for(args.cart_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
