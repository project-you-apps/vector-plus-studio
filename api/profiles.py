"""User profiles and seats: who is asking, and what may they reach.

The join that was missing. Two ends already existed and never met:

    auth.py     -- WHO you are        (Supabase JWT, verified against a LOCAL secret)
    access.py   -- what a LEVEL permits (can(), capabilities_for())
    db/004      -- WHICH carts, at what level  (user_carts + cart_grants)

This module is the middle: JWT -> seat -> effective access -> capability.

SEAT IDENTITY IS THE SUPABASE `sub`, AND THAT IS NOT A STYLE CHOICE
-------------------------------------------------------------------
A seat is keyed by the auth user's UUID, never by email or handle. Emails change,
handles are editable, and both are re-assignable — any of which would silently
re-point a person's entire attention history at someone else. The UUID is the only
identifier the auth system promises is stable and unique.

The cost is that overlay filenames are opaque (`8f2c….overlay.json`), which is why
`display_name_for()` exists: resolve the human label at PRESENTATION time from
`profiles`, never store it as identity.

OWNER IS NOT AN ACCESS LEVEL
----------------------------
`effective_access` in SQL returns 'owner' | 'viewer' | 'commenter' | 'editor' | NULL.
`access.py` deliberately has no ACCESS_LEVELS["owner"] entry so that `share` and
`sign` are unreachable from any grantable value. `can_here()` below preserves that
split rather than flattening it, because flattening is how a grant row becomes an
ownership transfer.

NULL MEANS DENIED
-----------------
Absence of a grant is a denial, not a default. Two fail-OPEN paths were found in the
sidecar reader on 2026-08-01 — an empty permissions file returned writable — so every
"no answer" path here resolves to no capabilities at all.
"""

from __future__ import annotations

from .access import _key, can, can_as_owner, capabilities_for

OWNER = "owner"

# Levels that may be stored in cart_grants. Mirrors the SQL check constraint in
# db/004_cart_grants.sql; a test asserts the two lists agree, because two spellings of
# one vocabulary is how a cart grants write in one layer and denies it in another.
GRANTABLE_LEVELS = ("viewer", "commenter", "editor")


# --------------------------------------------------------------------- seats

def seat_id(user: dict | None) -> str | None:
    """Stable seat identifier for a decoded JWT payload, or None when anonymous.

    `sub` is the Supabase auth user UUID. Returns None rather than a placeholder so a
    caller cannot accidentally write one anonymous seat's attention over another's.
    """
    if not isinstance(user, dict):
        return None
    sub = user.get("sub")
    if not isinstance(sub, str) or not sub.strip():
        return None
    return sub.strip()


def overlay_name(seat: str) -> str:
    """Filename stem for a seat's attention overlay. Path-safe by construction."""
    if not isinstance(seat, str) or not seat.strip():
        raise ValueError("seat id is required")
    safe = "".join(c for c in seat.strip() if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError(f"seat id {seat!r} has no path-safe characters")
    return safe


def display_name_for(profile: dict | None, user: dict | None = None) -> str:
    """Human label, resolved at presentation time. Never used as identity.

    Falls back down the chain the 001 migration set up: display_name -> full_name ->
    username -> the JWT email's local part -> "Unknown". A missing profile row is normal
    for a brand-new user, so this must never raise.
    """
    profile = profile if isinstance(profile, dict) else {}
    for field in ("display_name", "full_name", "username"):
        value = profile.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    email = (profile.get("email") or (user or {}).get("email") or "")
    if isinstance(email, str) and "@" in email:
        return email.split("@", 1)[0]
    return "Unknown"


# ------------------------------------------------------------------- access

def normalize_effective(value: object) -> str | None:
    """Normalize whatever `effective_access()` returned. Unknown values -> None (denied)."""
    key = _key(value)
    if key == OWNER:
        return OWNER
    return key if key in GRANTABLE_LEVELS else None


def is_owner(effective: object) -> bool:
    return normalize_effective(effective) == OWNER


def can_here(effective: object, capability: str) -> bool:
    """May a user with this effective access perform `capability` on this cart?

    Routes owners through `can_as_owner` and everyone else through `can`, preserving the
    split that keeps `share`/`sign` ungrantable. Any unrecognised value denies.
    """
    level = normalize_effective(effective)
    if level is None:
        return False
    if level == OWNER:
        return can_as_owner(capability)
    return can(level, capability)


def capabilities_here(effective: object) -> frozenset[str]:
    """Full capability set, for a UI badge or an API response."""
    level = normalize_effective(effective)
    if level is None:
        return frozenset()
    if level == OWNER:
        # Derived from can_as_owner rather than a second hard-coded list, so the two can
        # never drift apart.
        return frozenset(c for c in ("read", "annotate", "write", "share", "sign")
                         if can_as_owner(c))
    return capabilities_for(level)


def describe_cart_access(effective: object) -> dict:
    """One shape for the API and the User Page to render a cart's access state."""
    level = normalize_effective(effective)
    return {
        "access": level,                       # None means no access at all
        "is_owner": level == OWNER,
        "capabilities": sorted(capabilities_here(level)),
        "can_share": can_here(level, "share"),
        "read_only": bool(level) and not can_here(level, "write"),
    }


def visible_carts(rows) -> list:
    """Shape `user_carts` joined with effective access into what a User Page renders.

    Rows without a resolvable access level are DROPPED rather than shown greyed out.
    A row we cannot justify showing is one we should not be returning at all -- and per
    the revocation policy, whether an inaccessible thing is even acknowledged is a
    per-cart setting, not something a list endpoint should decide on its own.
    """
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        level = normalize_effective(row.get("effective_access") or row.get("access"))
        if level is None:
            continue
        out.append({
            "cart_id": row.get("id") or row.get("cart_id"),
            "cart_filename": row.get("cart_filename"),
            "display_name": row.get("display_name") or row.get("cart_filename"),
            "size_bytes": row.get("size_bytes"),
            "pattern_count": row.get("pattern_count"),
            **describe_cart_access(level),
        })
    return out
