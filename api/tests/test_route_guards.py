"""EVERY route declares a guard or an explicit, reasoned exemption. No third option.

THIS TEST IS THE POINT OF THE WHOLE EXERCISE.

On 2026-08-05 the mount endpoint was gated and twelve other cart-touching routes were not.
Betty, with no grant on redwood-finance, reached every cart and could delete and tombstone
passages. Fixing those twelve by hand would leave thirteen places to keep in agreement, and
the fourteenth route someone adds next month would silently skip the check. So the guarantee
got enforced here rather than by remembering.

⚠ AND THEN THIS TEST FAILED THE SAME WAY IT WAS BUILT TO PREVENT (2026-08-12).

It filtered routes through a hand-written path allowlist:

    CART_TOUCHING = re.compile(r"^/api/(search|patterns|cart/|cartridges/…|membox/…)")

Three routers added later -- `/api/agents`, `/api/reports`, `/api/llm` -- simply fell outside
it. **The test passed, and passing read as "every cart route is guarded."** Measured that day:
68 routes, 20 guarded. `POST /api/agents/run` was dispatching an agent against a caller-named
cart with no caller identity at all, and `DELETE /api/cartridges/eject` would delete the
mounted cart for anyone who asked.

An allowlist can only ever police the routes someone thought of. So it is inverted: enumerate
EVERYTHING, and require each route to declare a guard or appear in EXEMPT **with a written
reason**. The next new router fails closed instead of silently.

Three properties keep the list honest, and each exists because the alternative rots:
  • `test_no_stale_exemptions`      -- a deleted route must not leave its excuse behind
  • `test_every_exemption_states_a_reason` -- caught "as above." on its first run
  • the gate-themselves tests       -- an exemption nobody verifies is a hole with a comment

Adding a route? Guard it, or exempt it and say why. Both are one line; only one is a decision.
"""



import pytest

from api.main import app
from api.cart_guard import (
    require_cart_read,
    require_cart_write,
    require_named_cart_read,
)

# Every dependency that constitutes a guard. `require_named_cart_read` covers the by-name
# routes that read a cart WITHOUT mounting it -- checking the mounted cart there would guard
# the wrong object entirely, and would pass whenever the caller happened to have access to
# whatever else was open.
GUARDS = {require_cart_read, require_cart_write, require_named_cart_read}

# Cart Builder is a DESKTOP surface: it browses and writes the operator's own filesystem and
# never reads cart passages through the access layer. On a public host every write path is
# refused by `_check_writable` / PUBLIC_HOST (verified 2026-08-12: /api/cartbuilder/browse
# returns 403 there). It is nonetheless DEPLOYED on the droplet, and that control lives in
# the systemd unit rather than in code -- open question for Andy, tracked in the register.
_CARTBUILDER = ("desktop surface; filesystem not cart content; write paths refused by "
                "_check_writable/PUBLIC_HOST. Deployed on the droplet -- see register.")

# EVERY route must appear here or declare a guard. The reason is mandatory: an exemption
# without one is indistinguishable from an omission six weeks later.
EXEMPT: dict[tuple[str, str], str] = {
    # -- infrastructure: says nothing about carts ---------------------------------------
    ("GET", "/health"): "liveness only; deliberately says nothing about the corpus.",
    ("GET", "/docs"): "FastAPI docs UI. Describes the API surface, not any cart's content.",
    ("GET", "/docs/oauth2-redirect"): "FastAPI docs OAuth callback. No cart involvement.",
    ("GET", "/openapi.json"): "schema of the API surface; no cart content.",
    ("GET", "/redoc"): "FastAPI docs UI; no cart content.",
    ("GET", "/api/status"): (
        "PUBLIC BY NECESSITY -- App.tsx polls it before sign-in and engine_ready boots the "
        "UI, so a 401 makes the app look dead to every signed-out visitor. Protected "
        "per-FIELD instead: mounted_cartridge and mounted_path are nulled for callers who "
        "may not read the mounted cart (f63e8c3)."),

    # -- discovery: names, never passages -----------------------------------------------
    ("GET", "/api/cartridges"): (
        "Lists carts by filename and size. No passage content, and the cart list is how a "
        "user discovers what to ask for access TO."),
    ("GET", "/api/membox/carts"): "membox cart list; same discovery reasoning as above.",
    ("GET", "/api/membox/status/{cart_id}"): "mount state of one membox cart; no passages.",
    ("GET", "/api/reports/list"): "registered report metadata; no cart is read.",
    ("GET", "/api/reports/carts"): (
        "Enumerates carts with per-cart report compatibility. Names, not passages; same "
        "discovery reasoning as /api/cartridges."),
    ("GET", "/api/agents/list"): "registered agent metadata + cap config; no cart is read.",

    # -- gate themselves, verified by tests below ---------------------------------------
    ("POST", "/api/cartridges/mount"): (
        "Gates the cart being OPENED, not the one already open -- require_cart_read would "
        "check the wrong object and pass whenever the caller had access to whatever was "
        "previously mounted. Calls _gate_mount. See test_mount_routes_gate_themselves."),
    ("POST", "/api/membox/mount"): "as above; calls _gate_mount.",
    ("POST", "/api/agents/run"): (
        "Cart is named by req.cart_ref in the BODY, which FastAPI has not parsed at "
        "dependency-resolution time. Calls cart_guard.enforce_named_read after resolving "
        "the ref. See test_body_cart_routes_gate_themselves."),
    ("POST", "/api/reports/generate"): "as above; body cart_ref, calls enforce_named_read.",

    # -- unmount: dropping your own handle is not a privileged act -----------------------
    ("POST", "/api/cartridges/unmount"): (
        "Only drops the process's own handle. Refusing could strand a caller holding a cart "
        "they may no longer read -- the opposite of useful."),
    ("POST", "/api/membox/unmount"): (
        "Drops this process's membox handle only; same reasoning as /api/cartridges/unmount."),

    # -- identity and grants: governed by Supabase RLS, not by cart_guard ----------------
    ("GET", "/api/me"): "the caller's own profile; RLS-scoped to auth.uid().",
    ("GET", "/api/me/carts"): "the caller's own cart list; RLS-scoped.",
    ("GET", "/api/carts/{cart_id}/grants"): (
        "Grant administration, governed by SQL: db/004 restricts SELECT to the cart owner. "
        "cart_guard governs CONTENT access and would be the wrong authority here."),
    ("POST", "/api/carts/{cart_id}/grants"): (
        "as above; SQL also enforces grantee_id <> auth.uid() so an admin cannot self-grant."),
    ("DELETE", "/api/carts/{cart_id}/grants/{grantee_id}"): "as above; owner-only in SQL.",

    # -- compute with no cart content ----------------------------------------------------
    ("POST", "/api/embed"): (
        "Embeds a caller-supplied string for browser-side WebGPU Associate. Touches no cart. "
        "NOTE: unauthenticated server compute -- a rate-limit question, not an access one."),
    ("POST", "/api/llm/synthesize"): (
        "Sends a caller-supplied prompt to the configured LLM adapter. Reads no cart. "
        "NOTE: unauthenticated access to a PAID Cloudflare worker -- cost control needed, "
        "tracked in the register; not a cart-access guard."),
    ("GET", "/api/llm/health"): "adapter reachability; no cart, no prompt.",

    # -- creation: new files, no existing cart read or modified --------------------------
    ("POST", "/api/forge"): (
        "Creates NEW cart files from uploads; reads no existing cart. Governed by "
        "_enforce_writable, which refuses under VPS_READ_ONLY."),
    ("POST", "/api/cartridges/upload"): (
        "Writes a NEW file to the upload sandbox; reads no existing cart. "
        "NOTE: reachable unauthenticated and validates only file TYPE -- tracked in the "
        "register as an open question, not a cart-access gap."),

    # -- stub ----------------------------------------------------------------------------
    ("POST", "/api/agents/save_to_cart"): (
        "v1 STUB -- logs the intended save and returns success; writes nothing. MUST take "
        "require_cart_write (or enforce_named_read) when the real Membot write lands, or it "
        "becomes the 'reported success for a refused write' bug we closed on 2026-08-10."),

    # -- desktop-only --------------------------------------------------------------------
    ("GET", "/api/browse"): (
        "Opens a NATIVE OS file dialog on the server. Desktop-only by nature and inert on "
        "the headless droplet. Reads no cart. Should also be refused under PUBLIC_HOST -- "
        "tracked in the register."),
    ("DELETE", "/api/cartbuilder/cart_folders"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/browse"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/build/status"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/cart_folders"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/carts"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/files"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/has_changes"): _CARTBUILDER,
    ("GET", "/api/cartbuilder/pattern0"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/build"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/build-to-folder"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/cart_folders"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/clear_workspace"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/ingest"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/load_cart"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/metadata"): _CARTBUILDER,
    ("POST", "/api/cartbuilder/upload"): _CARTBUILDER,
}


def _routes():
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None) or set()
        if not path or not methods:
            continue
        deps = set()
        for d in getattr(getattr(r, "dependant", None), "dependencies", []) or []:
            if d.call in GUARDS:
                deps.add(d.call)
            for sub in getattr(d, "dependencies", []) or []:
                if sub.call in GUARDS:
                    deps.add(sub.call)
        # The dependency may also sit directly in the signature.
        for d in getattr(getattr(r, "dependant", None), "dependencies", []) or []:
            if d.call in GUARDS:
                deps.add(d.call)
        for m in methods:
            if m in ("HEAD", "OPTIONS"):
                continue
            yield m, path, deps


def test_every_route_is_guarded_or_explicitly_exempt():
    """EVERY route. Not a curated subset -- that was the 2026-08-12 bug.

    The previous version filtered through a hand-written path allowlist, so three routers
    added later (`/api/agents`, `/api/reports`, `/api/llm`) were invisible to it. The test
    passed, and passing read as "every cart route is guarded." `POST /api/agents/run` was
    dispatching an agent against a named cart with no caller identity at all.

    Enumerating everything makes the next new router fail closed instead of silently.
    """
    unaccounted = [
        f"{m} {p}"
        for m, p, deps in _routes()
        if not deps and (m, p) not in EXEMPT
    ]
    assert not unaccounted, (
        "These routes declare no access dependency and are not in EXEMPT:\n  "
        + "\n  ".join(sorted(unaccounted))
        + "\n\nEither add Depends(require_cart_read/require_cart_write), or add an EXEMPT "
          "entry WITH A REASON explaining why this route needs no cart guard."
    )


def test_no_stale_exemptions():
    """An EXEMPT entry for a route that no longer exists is a lie that accumulates.

    Without this, a deleted route leaves its excuse behind and the list slowly stops
    describing the app.
    """
    live = {(m, p) for m, p, _ in _routes()}
    stale = [f"{m} {p}" for (m, p) in EXEMPT if (m, p) not in live]
    assert not stale, (
        "EXEMPT names routes that no longer exist:\n  " + "\n  ".join(sorted(stale)))


def test_every_exemption_states_a_reason():
    """A blank reason is an omission wearing an exemption's clothes."""
    thin = [f"{m} {p}" for (m, p), why in EXEMPT.items() if len((why or "").strip()) < 20]
    assert not thin, ("These EXEMPT entries have no real reason:\n  "
                      + "\n  ".join(sorted(thin)))


def test_destructive_routes_require_write_not_merely_read():
    """viewer and commenter both pass require_cart_read. Neither may delete a passage.

    A destructive route guarded only by read access is the bug wearing a seatbelt.
    """
    destructive = [
        (m, p, deps) for m, p, deps in _routes()
        if p.startswith("/api/patterns") and m in ("DELETE", "PUT", "POST")
    ]
    assert destructive, "expected to find the pattern-mutation routes"
    wrong = [f"{m} {p}" for m, p, deps in destructive if require_cart_write not in deps]
    assert not wrong, (
        "These routes MODIFY the cart but do not require write access:\n  "
        + "\n  ".join(sorted(wrong))
    )


def test_the_guard_list_is_not_silently_empty():
    """If route enumeration breaks, everything above passes vacuously."""
    routes = list(_routes())
    assert len(routes) >= 50, f"expected the full app surface, enumerated {len(routes)}"
    guarded = [f"{m} {p}" for m, p, deps in routes if deps]
    assert len(guarded) >= 15, f"expected many guarded routes, found {len(guarded)}"


def test_mount_routes_gate_themselves():
    """The two mount routes are EXEMPT from the dependency, so prove they gate another way.

    An exemption nobody verifies is just a hole with a comment on it. Both must call
    `_gate_mount`, which resolves access to the cart being OPENED rather than the one
    already open.
    """
    import inspect
    from api import main

    for fn_name in ("mount_cartridge", "membox_mount_endpoint"):
        fn = getattr(main, fn_name)
        src = inspect.getsource(fn)
        assert "_gate_mount(" in src, f"{fn_name} is EXEMPT from the guard but never calls _gate_mount"
        assert "decision.allowed" in src, f"{fn_name} calls _gate_mount but ignores the answer"


def test_body_cart_routes_gate_themselves():
    """Routes naming their cart in the BODY are EXEMPT from the dependency. Prove they gate.

    FastAPI resolves a Depends before parsing the body, so `require_named_cart_read` cannot
    see `req.cart_ref`. These call `cart_guard.enforce_named_read` instead -- and this test
    exists because that technicality is exactly why they were unguarded until 2026-08-12.

    The call must come AFTER `_resolve_cart_ref`: guarding the string the caller sent rather
    than the cart it resolves to would check the wrong object.
    """
    import inspect

    from api import agents_routes, reports_routes

    for mod, fn_name in ((agents_routes, "run_agent_route"),
                         (reports_routes, "generate_report")):
        src = inspect.getsource(getattr(mod, fn_name))
        assert "enforce_named_read(" in src, (
            f"{fn_name} is EXEMPT from the dependency but never calls enforce_named_read")
        guard_at = src.index("enforce_named_read(")
        resolve_at = src.index("_resolve_cart_ref(")
        assert resolve_at < guard_at, (
            f"{fn_name} guards before resolving cart_ref -- it is checking the caller's "
            f"string, not the cart it resolves to")


# --------------------------------------------------------------- behaviour, not declaration

class _Req:
    headers: dict = {}


def _mounted(monkeypatch, name="redwood-finance.cart.npz"):
    from api import cart_guard
    monkeypatch.setattr(cart_guard.engine, "mounted_path", name, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    cart_guard._cache.clear()


def _decision(monkeypatch, **kw):
    from api import cart_access, cart_guard
    d = cart_access.decide(**kw)
    monkeypatch.setattr(cart_guard, "resolve", lambda request, user: d)
    return d


def test_viewer_may_read_but_is_refused_a_write(monkeypatch):
    """The distinction the whole ladder turns on: read access is not write access."""
    from fastapi import HTTPException
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=True, owner_id="owner", grant_level="viewer",
              seat="someone")

    assert cart_guard.require_cart_read(_Req(), {"sub": "someone"}) is not None

    with pytest.raises(HTTPException) as e:
        cart_guard.require_cart_write(_Req(), {"sub": "someone"})
    assert e.value.status_code == 403
    assert "viewer" in str(e.value.detail)


def test_ungranted_seat_is_refused_reads_entirely(monkeypatch):
    """Betty on redwood-finance, 2026-08-05: this is the case that was wide open."""
    from fastapi import HTTPException
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=True, owner_id="susie", grant_level=None, seat="betty")

    with pytest.raises(HTTPException) as e:
        cart_guard.require_cart_read(_Req(), {"sub": "betty"})
    assert e.value.status_code == 403


def test_editor_may_write(monkeypatch):
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=True, owner_id="susie", grant_level="editor",
              seat="andy")
    assert cart_guard.require_cart_write(_Req(), {"sub": "andy"}) is not None


def test_nothing_mounted_is_not_a_refusal(monkeypatch):
    """No cart open means nothing to protect; the route reports 'no cart' in its own words."""
    from api import cart_guard
    monkeypatch.setattr(cart_guard.engine, "mounted_path", None, raising=False)
    assert cart_guard.require_cart_read(_Req(), None) is None


def test_legacy_cart_writes_are_deferred_not_refused(monkeypatch):
    """A cart with no grant row must stay writable per its read-only flag.

    Regression for the first draft of require_cart_write, which refused whenever
    `may_write` was False -- freezing every unregistered cart AND the entire single-user
    local studio. `may_write` answers "did a grant authorise this", never "is the cart
    writable". Andy's rule, 2026-08-03: "if they are editable then anyone can write them and
    if they are read-only then no one can write them."
    """
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=False, owner_id=None, grant_level=None, seat="andy")
    assert cart_guard.require_cart_write(_Req(), {"sub": "andy"}) is not None


def test_unenforced_writes_are_deferred_not_refused(monkeypatch):
    """No auth configured at all: the local studio must keep working exactly as before."""
    from api import cart_guard
    _mounted(monkeypatch)
    _decision(monkeypatch, registered=False, owner_id=None, grant_level=None, seat=None,
              enforced=False)
    assert cart_guard.require_cart_write(_Req(), None) is not None


# ------------------------------------------------ per-passage read control (PERM_R)

def test_pattern_permits_read_matches_is_readable_semantics():
    """The two implementations of "is this passage readable" must agree, forever.

    Read control grew two implementations: CartHandle.is_readable (honoured by the AGENT
    retrieval path) and nothing at all on /api/search, which read the perms, returned them
    in every result, and never filtered. A passage marked unreadable was therefore hidden
    from agents and shown to people -- canon §7.1.2 requires BOTH cart access and PERM_R,
    and we enforced the second on one path of two.

    These cases mirror is_readable exactly. If they ever diverge, one caller starts leaking.
    """
    from api.cartridge_io import pattern_permits_read, PERM_R, PERM_W

    # Legacy / absent -> readable. Anything stricter hides every existing cart.
    assert pattern_permits_read(None) is True
    assert pattern_permits_read({}) is True
    assert pattern_permits_read({"perms": None}) is True
    assert pattern_permits_read({"perms": {}}) is True

    # Explicit grants.
    assert pattern_permits_read({"perms": {"r": True,  "w": False}}) is True
    assert pattern_permits_read({"perms": {"r": False, "w": True}}) is False


def test_read_and_write_helpers_are_independent():
    """A passage can be readable-not-writable, or writable-not-readable.

    Sounds odd until you want an append-only record nobody may re-read, or a reference
    passage nobody may edit. The bits are orthogonal; the helpers must not conflate them.
    """
    from api.cartridge_io import pattern_permits_read, pattern_permits_write

    read_only = {"perms": {"r": True, "w": False}}
    assert pattern_permits_read(read_only) and not pattern_permits_write(read_only)

    write_only = {"perms": {"r": False, "w": True}}
    assert not pattern_permits_read(write_only) and pattern_permits_write(write_only)
