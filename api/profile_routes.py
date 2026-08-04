"""FastAPI routes for user profiles, seats, and cart sharing.

Route mapping
-------------
- ``GET    /api/me``                     -> identity + seat + apps touched
- ``GET    /api/me/carts``               -> every cart this user can reach, with level
- ``GET    /api/carts/{cart_id}/grants`` -> who else can reach a cart
- ``POST   /api/carts/{cart_id}/grants`` -> grant/update access (OWNER ONLY)
- ``DELETE /api/carts/{cart_id}/grants/{grantee_id}`` -> revoke (OWNER ONLY)

WHERE ENFORCEMENT ACTUALLY LIVES
--------------------------------
Postgres row-level security, not this file. Every query below runs as the calling user
against `db/004_cart_grants.sql`'s policies, so a bug here can fail a request that should
have succeeded but cannot grant access RLS would refuse. That is the deliberate shape: we
decided on 2026-08-02 that a frontend or route bug must not be able to leak another user's
row, because every new endpoint is a fresh chance to forget a `WHERE user_id =`.

The checks in this module are therefore for MESSAGE QUALITY -- a 403 that says why -- and
never the only thing standing between a caller and someone else's data.

NOT A SECRECY BOUNDARY EITHER
-----------------------------
`access_level` scopes what a user is shown. Secrets are separated by CART: a cart someone
must not read is one they cannot mount, not one they are filtered out of. See
`docs/SPEC-SUBCARTRIDGE-MANIFEST.md` section 2.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Request
from pydantic import BaseModel, Field

from .auth import require_user
from .profiles import (
    GRANTABLE_LEVELS,
    describe_cart_access,
    display_name_for,
    overlay_name,
    seat_id,
    visible_carts,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["profiles"])

def supabase_url() -> str:
    return (os.environ.get("SUPABASE_URL") or "").strip()


def supabase_client_key() -> str:
    """The browser-safe key: publishable (new) or anon (legacy).

    Supabase's asymmetric-key migration renames `anon` to a *publishable* key. Both names
    are accepted so the swap in `docs/RUNBOOK-jwt-migration-and-key-revoke-2026-08-04.md`
    is not order-sensitive against a deploy -- an env file updated before or after the new
    code lands works either way.

    Read at call time, not import. The previous module constants froze whatever the
    environment held when the first import ran, which made the migration untestable and
    hid a live fact behind a startup snapshot.
    """
    return (os.environ.get("SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY") or "").strip()


# --------------------------------------------------------------------- models

class GrantRequest(BaseModel):
    grantee_id: str = Field(..., description="Supabase auth user id (uuid) of the recipient")
    access_level: str = Field(..., description=f"one of {GRANTABLE_LEVELS}")


class GrantResponse(BaseModel):
    cart_id: str
    grantee_id: str
    access_level: str


# --------------------------------------------------------------------- helpers

def _supabase(user_token: str):
    """Per-request Supabase client carrying the CALLER's token, so RLS applies to them.

    A service-role key here would bypass every policy in 004 and make this module the only
    thing preventing cross-user reads -- precisely the arrangement we decided against. If the
    client library is unavailable these endpoints report unavailable rather than falling back
    to something more permissive.
    """
    try:
        from supabase import create_client
    except ImportError as e:                       # pragma: no cover - env dependent
        raise HTTPException(status_code=503,
                            detail=f"supabase client not installed: {e}")
    url, key = supabase_url(), supabase_client_key()
    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail=("SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY (or SUPABASE_ANON_KEY) "
                    "not configured"))
    client = create_client(url, key)
    # Only attach a token when we actually have one. postgrest.auth("") raises
    # ValueError("Neither bearer token or basic authentication scheme is provided"), so an
    # anonymous caller used to crash here -- and the mount gate reported that crash as
    # "could not verify access", refusing every anonymous mount with a 503 that looked like
    # a service outage rather than the missing header it was.
    #
    # Without a token the client acts as `anon`, which is exactly what db/005's
    # `cart_access_for()` expects: auth.uid() is null, so it answers 'unregistered' for an
    # unmanaged cart and NULL (denied) for a registered one. Anonymous is a legitimate
    # caller here, not an error.
    if user_token:
        client.postgrest.auth(user_token)
    return client


def _token(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def _validate_level(level: str) -> str:
    if level not in GRANTABLE_LEVELS:
        # 'owner' lands here on purpose: ownership is proven by holding the cart row and the
        # signing key, never by a string someone can POST.
        raise HTTPException(
            status_code=400,
            detail=f"access_level must be one of {list(GRANTABLE_LEVELS)}; "
                   f"got {level!r}. Ownership is not grantable.",
        )
    return level


# --------------------------------------------------------------------- routes

@router.get("/me")
async def me(request: Request, user: dict = Depends(require_user)):
    """Identity for the User Page: who am I, what is my seat, which apps have I touched."""
    seat = seat_id(user)
    if seat is None:
        # require_user already rejected anonymous callers, so a token with no `sub` is a
        # malformed token rather than a logged-out user -- worth a distinct message.
        raise HTTPException(status_code=401, detail="token carries no subject claim")

    profile = {}
    try:
        client = _supabase(_token(request))
        rows = (client.table("profiles").select("*").eq("id", seat).limit(1)
                .execute().data or [])
        profile = rows[0] if rows else {}
    except HTTPException:
        raise
    except Exception as e:                          # noqa: BLE001
        # A missing profile row is normal for a brand-new user; a failed lookup should
        # degrade to "identity from the token" rather than a broken page.
        log.warning("profile lookup failed for %s: %s", seat, e)

    return {
        "seat": seat,
        "overlay": overlay_name(seat),
        "email": user.get("email") or profile.get("email"),
        "display_name": display_name_for(profile, user),
        "avatar_url": profile.get("avatar_url"),
        "apps_list": profile.get("apps_list") or [],
        "created_at": profile.get("created_at"),
    }


@router.get("/me/carts")
async def my_carts(request: Request, user: dict = Depends(require_user)):
    """Every cart this user can reach -- owned and shared -- each with its access level.

    RLS returns owned rows (existing policy) and granted rows (added in 004), so this is one
    query rather than a union we would have to keep correct by hand.
    """
    client = _supabase(_token(request))
    rows = (client.table("user_carts")
            .select("id, cart_filename, display_name, size_bytes, pattern_count, user_id")
            .execute().data or [])

    seat = seat_id(user)
    grants = {g["cart_id"]: g["access_level"] for g in
              (client.table("cart_grants").select("cart_id, access_level")
               .eq("grantee_id", seat).execute().data or [])}

    for row in rows:
        row["effective_access"] = ("owner" if row.get("user_id") == seat
                                   else grants.get(row.get("id")))
    return {"carts": visible_carts(rows)}


@router.get("/carts/{cart_id}/grants")
async def list_grants(request: Request, cart_id: str = PathParam(...),
                      user: dict = Depends(require_user)):
    """Who can reach this cart. RLS shows an owner everything and a grantee only their own."""
    client = _supabase(_token(request))
    rows = (client.table("cart_grants")
            .select("cart_id, grantee_id, access_level, created_at")
            .eq("cart_id", cart_id).execute().data or [])
    return {
        "cart_id": cart_id,
        "grants": [{**r, **describe_cart_access(r.get("access_level"))} for r in rows],
    }


@router.post("/carts/{cart_id}/grants", response_model=GrantResponse)
async def upsert_grant(request: Request, body: GrantRequest,
                       cart_id: str = PathParam(...),
                       user: dict = Depends(require_user)):
    """Grant or change someone's access. OWNER ONLY -- enforced by RLS, not by this code."""
    level = _validate_level(body.access_level)
    seat = seat_id(user)
    if body.grantee_id == seat:
        raise HTTPException(
            status_code=400,
            detail="cannot grant to yourself; owners already hold every capability",
        )

    client = _supabase(_token(request))
    try:
        client.table("cart_grants").upsert(
            {"cart_id": cart_id, "grantee_id": body.grantee_id,
             "access_level": level, "granted_by": seat},
            on_conflict="cart_id,grantee_id",
        ).execute()
    except Exception as e:                          # noqa: BLE001
        # RLS refusing is the EXPECTED failure for a non-owner. Reported as 403 with a
        # reason rather than a 500, so a caller can tell "not allowed" from "broken".
        log.warning("grant failed on %s by %s: %s", cart_id, seat, e)
        raise HTTPException(status_code=403,
                            detail="not permitted to grant access on this cart")
    return GrantResponse(cart_id=cart_id, grantee_id=body.grantee_id, access_level=level)


@router.delete("/carts/{cart_id}/grants/{grantee_id}")
async def revoke_grant(request: Request, cart_id: str = PathParam(...),
                       grantee_id: str = PathParam(...),
                       user: dict = Depends(require_user)):
    """Revoke access. OWNER ONLY.

    Revocation does NOT erase the grantee's attention history: their overlay keys simply stop
    resolving, which the sub-cart resolver already reports as `unresolved`. Whether revoked
    passages appear as restricted or vanish entirely is the cart's `revocation_policy`, not
    this endpoint's business.
    """
    client = _supabase(_token(request))
    try:
        (client.table("cart_grants").delete()
         .eq("cart_id", cart_id).eq("grantee_id", grantee_id).execute())
    except Exception as e:                          # noqa: BLE001
        log.warning("revoke failed on %s by %s: %s", cart_id, seat_id(user), e)
        raise HTTPException(status_code=403,
                            detail="not permitted to revoke access on this cart")
    return {"cart_id": cart_id, "grantee_id": grantee_id, "revoked": True}
