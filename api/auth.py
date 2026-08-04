"""
JWT verification for Supabase Auth — legacy HS256 and asymmetric (JWKS) side by side.

Pattern:
  - get_current_user(): FastAPI dependency returning dict|None. Endpoints that
    work both signed-in and signed-out (e.g. cart list — anon gets bundled +
    sandbox, authed gets bundled + sandbox + own private) use this.
  - require_user(): FastAPI dependency that 401s if no user. Endpoints that
    require sign-in (private cart upload, save-search, etc.) use this.

WHY TWO VERIFICATION PATHS AT ONCE
-----------------------------------
Supabase is moving projects off a single shared HS256 secret onto asymmetric
signing keys with a published JWKS. During that migration BOTH kinds of token
are in circulation: already-issued HS256 tokens stay valid while newly issued
ones are signed with the standby asymmetric key. A server that understands only
one of them forces a flag-day cutover and signs everyone out at the moment of
the switch.

So the algorithm in the token header selects the verifier:

    HS256          -> SUPABASE_JWT_SECRET          (legacy; goes away at revoke)
    RS256 / ES256  -> JWKS at SUPABASE_URL         (the destination)

After the legacy secret is revoked, `SUPABASE_JWT_SECRET` is simply removed from
the environment and the HS256 branch stops being reachable. No code change, no
second deploy. See `docs/RUNBOOK-jwt-migration-and-key-revoke-2026-08-04.md`.

ALGORITHM CONFUSION — WHY THE BRANCHES DO NOT SHARE AN `algorithms` LIST
-------------------------------------------------------------------------
`jwt.get_unverified_header()` reads attacker-controlled data. It is used ONLY to
choose a branch; each branch then verifies with its own fixed algorithm list and
its own key. That ordering matters.

A JWKS is public by design. The classic attack is to take the published public
key and sign a forged token with HS256, using that public key as the shared
secret — which succeeds against any verifier that accepts a caller-supplied
`alg` against a single key. Here the HS256 branch verifies against
`SUPABASE_JWT_SECRET` and nothing else, so a token signed with the public key
fails. Once the secret is revoked and unset, HS256 is refused outright.

CONFIGURATION IS READ AT CALL TIME, NOT IMPORT
-----------------------------------------------
The previous version captured the secret in a module constant at import. That
made the migration below untestable without reimporting the module, and it hid a
live fact behind a startup-time snapshot. Every accessor here re-reads the
environment; the only cached thing is the JWKS client, keyed by URL so it drops
itself when the project changes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request

log = logging.getLogger(__name__)

JWT_AUDIENCE = "authenticated"

# Kept as separate tuples, never concatenated -- see the module docstring.
SYMMETRIC_ALGS = ("HS256",)
ASYMMETRIC_ALGS = ("RS256", "ES256")

# Supabase publishes the project's signing keys here.
JWKS_PATH = "/auth/v1/.well-known/jwks.json"

# Cached (url, client). PyJWKClient does its own key caching; this only avoids
# rebuilding the client per request.
_jwks_cache: tuple[str, object] | None = None


# ------------------------------------------------------------------ configuration

def legacy_secret() -> Optional[str]:
    """The legacy shared HS256 secret, or None once it is gone."""
    return os.environ.get("SUPABASE_JWT_SECRET") or None


def project_url() -> Optional[str]:
    """Supabase project URL, trailing slash removed, or None."""
    return (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/") or None


def jwks_url() -> Optional[str]:
    base = project_url()
    return f"{base}{JWKS_PATH}" if base else None


def auth_configured() -> bool:
    """True when at least one verification path is available.

    Either is sufficient: a project mid-migration may have both, a fully migrated
    one has only the URL, and a legacy one only the secret. Requiring both would
    refuse to authenticate a correctly configured server at each end of the move.
    """
    return bool(legacy_secret() or project_url())


def _jwks_client():
    """Memoized PyJWKClient for the current project URL, or None."""
    global _jwks_cache
    url = jwks_url()
    if not url:
        return None
    if _jwks_cache and _jwks_cache[0] == url:
        return _jwks_cache[1]
    from jwt import PyJWKClient
    client = PyJWKClient(url, cache_keys=True, lifespan=300)
    _jwks_cache = (url, client)
    return client


def reset_jwks_cache() -> None:
    """Drop the cached client. For tests and for a post-rotation restart."""
    global _jwks_cache
    _jwks_cache = None


# ------------------------------------------------------------------ verification

def _decode(token: str) -> dict:
    """Decode + verify a Supabase JWT. Raises jwt.PyJWTError on failure."""
    alg = (jwt.get_unverified_header(token) or {}).get("alg")

    if alg in SYMMETRIC_ALGS:
        secret = legacy_secret()
        if not secret:
            # Correct after the legacy secret is revoked: an HS256 token is no
            # longer something this project issues, so it is not something we
            # should accept.
            raise jwt.InvalidKeyError(
                "HS256 token presented but SUPABASE_JWT_SECRET is not set")
        return jwt.decode(token, secret, algorithms=list(SYMMETRIC_ALGS),
                          audience=JWT_AUDIENCE)

    if alg in ASYMMETRIC_ALGS:
        client = _jwks_client()
        if client is None:
            raise jwt.InvalidKeyError(
                "asymmetric token presented but SUPABASE_URL is not set")
        signing_key = client.get_signing_key_from_jwt(token).key
        return jwt.decode(token, signing_key, algorithms=list(ASYMMETRIC_ALGS),
                          audience=JWT_AUDIENCE)

    raise jwt.InvalidAlgorithmError(f"unsupported JWT algorithm: {alg!r}")


# ------------------------------------------------------------------ dependencies

def get_current_user(request: Request) -> Optional[dict]:
    """Return decoded JWT payload (with sub, email, role) or None when anonymous.

    Use as `user: dict | None = Depends(get_current_user)` on endpoints that
    behave differently for signed-in vs signed-out callers.
    """
    if not auth_configured():
        return None  # auth not configured -> everyone is anonymous

    auth = request.headers.get("authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None  # no token -> anonymous

    token = auth[7:].strip()
    if not token:
        return None

    try:
        return _decode(token)
    except jwt.ExpiredSignatureError:
        # Token expired. We return None rather than 401 so the client can show
        # a "your session expired" prompt to re-auth, rather than the endpoint
        # blanket-refusing. Endpoints that require auth use require_user, which
        # WILL 401 in this case (via the None check there).
        return None
    except jwt.PyJWTError as e:
        # Log the TYPE as well as the message. A verification failure during the
        # migration is most often a configuration fact (wrong branch, JWKS
        # unreachable) rather than a bad token, and the class name is what tells
        # those apart at a glance.
        log.warning("JWT verification failed: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=401, detail="invalid auth token")


def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Return the decoded user payload, or 401 if not signed in.

    Use as `user: dict = Depends(require_user)` on endpoints that must be
    signed-in (private cart upload, save-search, profile updates, etc.).
    """
    if not auth_configured():
        raise HTTPException(
            status_code=503,
            detail=("auth not configured on this server "
                    "(set SUPABASE_URL, or SUPABASE_JWT_SECRET for legacy projects)"),
        )
    if user is None:
        raise HTTPException(status_code=401, detail="sign in required")
    return user


def user_id_or_none(user: Optional[dict]) -> Optional[str]:
    """Convenience: pull the Supabase user UUID (sub claim) from a decoded payload."""
    return user.get("sub") if user else None
