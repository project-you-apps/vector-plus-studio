# AGENTS.md — working on Vector+ Studio

> For an agent (or a new human) **working on this codebase**. If you want to *use* a cart
> through tools rather than change this app, see
> [Membot's AGENTS.md](https://github.com/project-you-apps/membot/blob/main/AGENTS.md).

---

## 1 · ⛔ Read this before your first change

**Four things here will waste an afternoon if you learn them by discovery.**

**1 · `membot` must be a sibling directory with that exact name.**
`api/provenance_link.py` resolves it as `here.parent.parent.parent / "membot"` — one level
*above* this repo. ⇒ Layout:

```text
<parent>/
  vector-plus-studio-repo/   ← this repo
  membot/                    ← must be named exactly this
```

⛔ **The Studio imports every membot provenance module in ONE `try`.** A membot that is missing,
stale, or differently named makes `modules()` return `None` and **the entire provenance layer
reports "unavailable"** — notes, attribution, the lot. ⚠ It fails as a feature flag being off,
not as an import error, which is why it is hard to diagnose. ⓘ `MEMBOT_PATH` overrides the
location if you need a different layout.

**2 · `frontend/.env.local` is gitignored and the app will not start without it.**
`src/lib/supabase.ts` **throws at module load** if `VITE_SUPABASE_URL` or
`VITE_SUPABASE_ANON_KEY` is missing. A throw at import time means React never mounts ⇒ **a blank
page, nothing on screen, and a backend log that looks perfectly healthy.**

```text
VITE_SUPABASE_URL=<your supabase url>
VITE_SUPABASE_ANON_KEY=<your publishable anon key>
```

⚠ **Vite reads `.env` files only at startup.** After creating it, restart the dev server; a
browser refresh will not pick it up.

**3 · Dark is the default. Light is a class, not a Tailwind variant.**
Components are written dark-first (`bg-slate-950/60` + light text). Light mode is
`Header.tsx` putting **`html.light`** on the root, with the overrides in
`frontend/src/index.css`. ⛔ **Do not reach for `dark:` variants** — there are two in the whole
tree and they are outliers. ⇒ If you add an `<input>`, `<textarea>` or `<select>`, check it in
**both** themes; the generic `html.light` rules in `index.css` exist because form controls
stayed near-black and illegible otherwise.

**4 · Never commit a cartridge or its sidecars.**
`cartridges/` is gitignored by construction. ⛔ The `.journal.jsonl` sidecars contain **real seat
identifiers**, and this repository is public. ⚠ Also do not use a blanket `git add` — stage by
name.

---

## 2 · Layout

| Path | What lives there |
| --- | --- |
| `api/` | FastAPI backend. `main.py` wires routers |
| `api/reports/` | The report engine — one module per report type, registered by decorator |
| `api/agents/` | Agent recipes that run against a cart |
| `api/tests/`, `api/reports/tests/`, `api/agents/tests/` | The suite (~590 tests) |
| `frontend/src/components/` | Screens and UI |
| `frontend/src/api/` | Client; `client.ts` decides local-versus-server routing |
| `frontend/src/store/appStore.ts` | Zustand store — mount state, seat, modal state |
| `frontend/src/lib/` | Supabase client, posture detection |
| `cartridges/` | Carts, gitignored |
| `sample_data/` | Sample carts via git-lfs (~490 MB) |
| `bin/` | Optional GPU engine, separate licence |

### The backend modules worth knowing by name

| Module | Responsibility |
| --- | --- |
| `cartridge_io.py` | Finds and loads carts. `DATA_DIR` is `<repo>/cartridges` |
| `cart_access.py` | Does this seat have a level on this cart? |
| `object_access.py` | Per-document withholding inside a shared cart |
| `cart_guard.py` | The enforcement helpers routes call |
| `request_cart.py` | Resolves the seat from the request; display names |
| `provenance_link.py` | The membot bridge — see §1 |
| `provenance_routes.py` | Notes: read, write, supersede |
| `search.py` | Ranking and keyword rerank |

---

## 3 · Running it

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..

python run_api.py --host 127.0.0.1 --port 8000     # backend
cd frontend && npm run dev                          # frontend on :5173
```

```bash
python -m pytest api -q      # the suite
cd frontend && npx tsc -b    # type check
```

⚠ **Do not develop with `--reload` if a cart is mounted.** WatchFiles treats antivirus and
indexer touches as edits; a reload **drops the mounted cart**, and on a freshly written tree it
can fire several times a minute.

ⓘ **First search downloads the embedding model** (~1.1 GB, once) to the default HuggingFace
cache. Nothing in `api/` reads `HF_HOME`, so it goes where HuggingFace puts it.

ⓘ **Windows:** clicking inside a console window enables QuickEdit and **suspends the process**
until you press Escape. A "hung" `pip`, `npm` or `uvicorn` is often this.

---

## 4 · Conventions that are load-bearing

**`content_key` is computed server-side, always.** It is a hash of the normalized passage body
and it is how a note finds its passage again after a cart is rebuilt. ⛔ Never accept one from a
client and never recompute it with a different normalization.

**Access is enforced in the body for body-cart routes.** A route whose target cart is named in
the request body **cannot** be gated by a `Depends`, because the dependency runs before the body
is read. ⇒ Those routes call `cart_guard.enforce_named_read(...)` themselves, and
`test_body_cart_routes_gate_themselves` asserts they do. **If you add such a route, add it to
that test.**

⚠ **Access currently fails OPEN for unregistered carts.** Nothing writes `user_carts` on cart
creation, so a cart nobody registered yields `DECISION_UNREGISTERED` and is allowed. ⇒ The gate
is correctly placed and correctly called; what it consults is often empty. **Do not read a
passing access test as proof the boundary bound.**

**Notes live in sidecars, never inside the cart.** Annotating must not rewrite the data. A
correction **supersedes**; it never deletes.

**Cart directories are read from the environment on every call**, not cached at import — so
changing `VPS_EXTRA_CARTRIDGE_DIRS` does not require hunting down a long-running process.
⛔ A directory that does not exist is **skipped, not created**: a typo must not become an empty
folder that then reports zero carts as though that were an answer.

**`list_cartridges` skips anything under 1024 bytes.** Placeholder and pointer files are not
carts, and treating them as carts produced click-to-mount failures that looked like corruption.

---

## 5 · Working on the frontend

- **State is Zustand** in `appStore.ts`. Mount state, the current seat and modal state live
  there; do not duplicate them into component state.
- **Check both themes.** See §1.3.
- ⚠ **Keyboard handlers must bail while the user is typing.** The passage modal navigates with
  arrow keys, which silently ate note drafts until the handler learned to ignore events whose
  target is an `input`, `textarea`, `select` or `contentEditable`. ⇒ If you add a global key
  handler, do the same.
- **Popovers near the nav rail need a portal.** The rail's parent is `overflow-hidden`, so an
  absolutely-positioned popover is clipped. `PostureBadge.tsx` uses `createPortal`.

---

## 6 · ⛔ Things not to do

- ⛔ **Do not commit carts, sidecars, `.env` files or anything under `cartridges/`.**
- ⛔ **Do not blanket `git add`.** Stage by name.
- ⛔ **Do not add a `dark:` variant.** See §1.3.
- ⛔ **Do not accept a `content_key` from a client.**
- ⚠ **Do not assume a passing access test proves enforcement** — see §4 on failing open.
- ⚠ **Do not add a route whose cart comes from the body without gating it in the body** and
  registering it in the self-gating test.
