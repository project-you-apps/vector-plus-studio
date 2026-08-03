"""Access levels and capabilities — the authorization layer for cart-level permissions.

Implements `docs/_canon/CARTRIDGE-FORMAT.md` §7.1 / §7.1.1. Canon is normative; if this
file disagrees with it, this file is wrong.

WHY A CAPABILITY TABLE AND NOT `if level == "editor"`
------------------------------------------------------
Canon §7.1.1: *"Roles are the interface; capabilities are the mechanism… Every enforcement
point MUST read `can(access_level, "write")` rather than `access_level == "editor"`. Adding
a level later then costs one table row instead of an audit of every call site. This is the
difference between the change being a rename and being an architecture."*

That is the whole reason this module exists as a table rather than a set of comparisons
scattered through the API.

WHAT IS *NOT* HERE: OWNERSHIP
------------------------------
Owner is deliberately absent from the sidecar and from `ACCESS_LEVELS`. Canon §7.1:
ownership is proven by holding the key that signs the cart (§4.5), not by a string in a
file — *"a non-owner cannot grant themselves ownership by editing a string, because
ownership is not a string."* Three levels, not four. `can_as_owner()` exists for call sites
that have already verified a signature; it is not reachable from sidecar data.

THE TWO LAYERS DO NOT MEET HERE
--------------------------------
This module answers *"what may this PERSON do with this CART?"* only. Whether an individual
passage is readable is a separate question answered by the per-pattern `PERM_R` bit
(h-block byte 29) in `api/agents/retrieval.py`, and canon §7.1.2 requires **both** to pass:

    A passage is readable iff access_level grants `read` AND that passage's PERM_R permits it.

Person-capability ∩ object-sensitivity. Checking only this module leaks non-readable
passages; checking only the bit ignores the caller. Do not let one substitute for the other.
"""

from __future__ import annotations

# Capability sets per access level. Canon §7.1.1's table, verbatim.
#
#   read     — search, view passages
#   annotate — write to the comment sidecar, NEVER to the cart itself
#   write    — add / edit / tombstone passages
#   share    — modify .permissions.json
#   sign     — re-sign the immutable core (owner only; possession of the key IS the grant)
ACCESS_LEVELS: dict[str, frozenset[str]] = {
    "viewer":    frozenset({"read"}),
    "commenter": frozenset({"read", "annotate"}),
    "editor":    frozenset({"read", "annotate", "write"}),
}

# Owner is not a sidecar value — see the module docstring. Kept separate so that no amount
# of sidecar editing can produce it.
_OWNER_CAPABILITIES: frozenset[str] = frozenset(
    {"read", "annotate", "write", "share", "sign"}
)

ALL_CAPABILITIES: frozenset[str] = _OWNER_CAPABILITIES

# The most restrictive level we support. Unknown input resolves here, never to a permissive
# default — canon: "fail closed, never open."
FAIL_CLOSED_LEVEL = "viewer"

# Absent sidecar is NOT the same as malformed sidecar.
#
# No sidecar at all means a cart built before Step 2a, and every one of those is currently
# writable in private use. Failing those closed would break existing local carts to defend
# against a threat that is not present on a single-user box. Canon's fail-closed rule is
# about *unrecognized values in a sidecar that exists* — a file someone wrote and we cannot
# interpret. That distinction is the difference between "safe" and "broke everybody."
LEGACY_ABSENT_LEVEL = "editor"

# Canon §7.1 migration mapping. `x` was never semantically distinct for carts, so it must
# NOT silently promote to anything beyond `rw`.
_LEGACY_DEFAULT_TO_LEVEL: dict[str, str] = {
    "r":    "viewer",
    "rw":   "editor",
    "rwx":  "editor",
}


def normalize_access_level(value: object) -> str | None:
    """Coerce a candidate access level to a known one, or None if unrecognized.

    Returns None rather than a default so callers can distinguish "unknown" (fail closed)
    from "absent" (legacy). Collapsing those two is how a malformed file quietly becomes
    full write access.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in ACCESS_LEVELS else None


def resolve_access_level(sidecar: dict | None) -> str:
    """Resolve a `.permissions.json` payload to an access level. DUAL-READ.

    Canon §7.1 migration contract:
      - prefer `access_level`
      - fall back to legacy `default` (r / rw / rwx) when it is absent
      - a sidecar with neither key, or an unrecognized value, fails CLOSED
      - no sidecar at all keeps legacy behaviour (see LEGACY_ABSENT_LEVEL)

    Note `role` is deliberately NOT consulted. In this codebase `role` means CART TYPE
    (identity / episodic / semantic / federated) and is exposed publicly as
    `multi_search(role_filter=…)`. Reading it here would conflate two unrelated concepts —
    see the naming decision in canon §7.1 (2026-08-01).
    """
    if sidecar is None:
        return LEGACY_ABSENT_LEVEL
    if not isinstance(sidecar, dict):
        return FAIL_CLOSED_LEVEL

    level = normalize_access_level(sidecar.get("access_level"))
    if level:
        return level

    legacy = sidecar.get("default")
    if isinstance(legacy, str):
        mapped = _LEGACY_DEFAULT_TO_LEVEL.get(legacy.strip().lower())
        if mapped:
            return mapped
        return FAIL_CLOSED_LEVEL      # present but uninterpretable

    # A sidecar exists but declares no permission at all — someone wrote a file we cannot
    # read. Treat as restrictive.
    return FAIL_CLOSED_LEVEL


def _key(value: object) -> str:
    """Normalize any input to a lookup key. NEVER raises.

    Non-string input must DENY, not crash — an enforcement primitive that throws on
    unexpected input turns an authorization question into a 500, and a 500 in a write path
    is indistinguishable from a permission bug at 3am. Caught by
    `test_can_denies_unknown_levels[7]` on 2026-08-01, which passed an int.
    """
    return value.strip().lower() if isinstance(value, str) else ""


def can(access_level: object, capability: object) -> bool:
    """Does this access level grant this capability?

    THE enforcement primitive. Call sites use this, never `access_level == "editor"`.

    Unknown level or unknown capability → False. An unrecognized capability returning False
    means a typo denies access rather than granting it, which is the safe direction for a
    misspelling to fail in. Non-string input of any type also denies.
    """
    caps = ACCESS_LEVELS.get(_key(access_level))
    return bool(caps) and isinstance(capability, str) and capability in caps


def can_as_owner(capability: str) -> bool:
    """Capabilities of a verified owner. Only for call sites that checked a signature.

    Separate function rather than an `ACCESS_LEVELS["owner"]` entry, so that `share` and
    `sign` are unreachable from any sidecar-derived value no matter what it contains.
    """
    return capability in _OWNER_CAPABILITIES


def capabilities_for(access_level: str | None) -> frozenset[str]:
    """Full capability set, for surfacing in an API response or UI."""
    return ACCESS_LEVELS.get(_key(access_level), frozenset())


def describe_denial(access_level: str | None, capability: str) -> str:
    """Human-readable reason a capability was denied.

    Refusals should say which rule stopped you. "You cannot write" without a reason is
    undebuggable from the far side of an API or MCP call — the same lesson as
    `membot_server._write_blocked`.
    """
    level = _key(access_level) or "unknown"
    if level not in ACCESS_LEVELS:
        return (f"Access level {level!r} is not recognized, so the request was refused "
                f"(fail-closed). Known levels: {', '.join(sorted(ACCESS_LEVELS))}.")
    granted = ", ".join(sorted(ACCESS_LEVELS[level])) or "nothing"
    need = _minimum_level_for(capability)
    if need:
        return (f"'{capability}' requires access level '{need}'; this caller is '{level}' "
                f"(grants: {granted}).")
    if capability in ("share", "sign"):
        return (f"'{capability}' is reserved to the cart owner, which is proven by holding "
                f"the signing key — not by any access level.")
    return f"'{capability}' is not a known capability."


def _minimum_level_for(capability: str) -> str | None:
    """Least-privileged level granting a capability, for error messages."""
    for level in ("viewer", "commenter", "editor"):
        if capability in ACCESS_LEVELS[level]:
            return level
    return None
