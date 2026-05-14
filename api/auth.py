"""
JWT verification middleware for Supabase Auth.

Pattern:
  - get_current_user(): FastAPI dependency returning dict|None. Endpoints that
    work both signed-in and signed-out (e.g. cart list — anon gets bundled +
    sandbox, authed gets bundled + sandbox + own private) use this.
  - require_user(): FastAPI dependency that 401s if no user. Endpoints that
    require sign-in (private cart upload, save-search, etc.) use this.

JWT_SECRET is read from SUPABASE_JWT_SECRET env var at import time. If unset,
all requests are treated as anonymous (matches pre-Supabase v1.1 behavior),
which makes local dev work without configuring the secret. Production must
set it; require_user() raises 503 if called when unset.

Supabase issues HS256-signed JWTs with aud=authenticated when a user is
signed in. Standard claims include sub (user UUID), email, role.
"""

import logging
import os
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request

log = logging.getLogger(__name__)

JWT_SECRET: Optional[str] = os.environ.get("SUPABASE_JWT_SECRET")
JWT_AUDIENCE = "authenticated"
JWT_ALGORITHMS = ["HS256"]

if not JWT_SECRET:
    log.warning(
        "SUPABASE_JWT_SECRET not set. All requests will be treated as anonymous. "
        "Set this env var in production (see OAUTH-PRE-SCOPE.md)."
    )


def _decode(token: str) -> dict:
    """Decode + verify a Supabase JWT. Raises jwt.PyJWTError on failure."""
    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=JWT_ALGORITHMS,
        audience=JWT_AUDIENCE,
    )


def get_current_user(request: Request) -> Optional[dict]:
    """Return decoded JWT payload (with sub, email, role) or None when anonymous.

    Use as `user: dict | None = Depends(get_current_user)` on endpoints that
    behave differently for signed-in vs signed-out callers.
    """
    if JWT_SECRET is None:
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
        log.warning("JWT verification failed: %s", e)
        raise HTTPException(status_code=401, detail="invalid auth token")


def require_user(user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Return the decoded user payload, or 401 if not signed in.

    Use as `user: dict = Depends(require_user)` on endpoints that must be
    signed-in (private cart upload, save-search, profile updates, etc.).
    """
    if JWT_SECRET is None:
        raise HTTPException(
            status_code=503,
            detail="auth not configured on this server (SUPABASE_JWT_SECRET unset)",
        )
    if user is None:
        raise HTTPException(status_code=401, detail="sign in required")
    return user


def user_id_or_none(user: Optional[dict]) -> Optional[str]:
    """Convenience: pull the Supabase user UUID (sub claim) from a decoded payload."""
    return user.get("sub") if user else None
