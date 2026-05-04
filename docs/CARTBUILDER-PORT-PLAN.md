# Cart Builder → VPS Port Plan

*Filed 2026-05-04. Companion to the sidebar reorg phase that promotes
Cart Builder from a nav-rail stub to a real screen.*

## Source

The Cart Builder is a working Flask + HTML/JS app already shipped at
`project-you-apps/cart-builder` (commit `c2fb03c`, v1.1, 2026-04-03).
Local clone: `cart-builder/cart-builder/`. Branch `master`, in sync with
origin. **Not orphaned, not uncommitted** — unlike the Membox visualizer
recovery, this is real shipped code.

Sizes: `app.py` 692 lines, `templates/index.html` 101 lines (inline JS
likely heavier, see `static/`), supporting modules `builder.py`,
`cartridge_builder.py`, `parsers.py`.

## Endpoints (22 Flask routes)

| Route | Method | VPS port disposition |
|---|---|---|
| `/` | GET | **Replace** with React `CartBuilderScreen` (no port needed) |
| `/upload` | POST | **Port** — file upload, multipart |
| `/files` | GET | **Port** — workspace state |
| `/metadata` | POST | **Port** — set per-file metadata |
| `/ingest` | POST | **Port** — extract text from uploaded file |
| `/remove/<id>` | POST | **Port** — soft-remove (undo path) |
| `/restore/<id>` | POST | **Port** — undo soft-remove |
| `/replace` | POST | **Port** — swap file, preserve metadata |
| `/delete/<id>` | DELETE | **Port** — hard delete |
| `/pattern0` | GET | **Port** — manifest preview |
| `/build` | POST | **Port** — kick off cart build |
| `/build/status` | GET | **Port** — long-poll build progress |
| `/carts` | GET | **Reuse** VPS `/api/cartridges` (mostly compatible — verify shape) |
| `/cart_folders` | GET/POST/DELETE | **Port** — manage cart-search folders |
| `/browse` | GET | **Skip / replace** — server-side directory browsing; same anti-pattern as the existing VPS Open Cartridge bug. Replace with `<input type="file" webkitdirectory>` or path entry. |
| `/search` | POST | **Reuse** VPS `/api/search` after build completes |
| `/load_cart` | POST | **Port** — load existing cart into workspace for editing |
| `/deploy` | POST | **Port** with care — known buggy (filed 2026-04-03) |
| `/clear_workspace` | POST | **Port** — reset workspace |
| `/has_changes` | GET | **Port** — dirty flag for unsaved-changes warning |

Total to port: 14 endpoints. 2 reusable from existing VPS. 1 to skip/replace.

## Frontend port

The Flask UI is a single-page HTML template with inline JS calling the
routes above. React port plan:

```
components/cartbuilder/
├── CartBuilderScreen.tsx    (top-level — orchestrates panels)
├── DropZone.tsx              (drag-drop + file picker — fixes the OS picker bug)
├── FileCard.tsx              (per-file card with metadata edit, replace, remove)
├── FileGrid.tsx              (grid of FileCards, capped at 100 visible)
├── MetadataEditor.tsx        (modal: owner / description / tags)
├── Pattern0Preview.tsx       (manifest preview — TOC-like)
├── BuildPanel.tsx            (name field, Build/Update button, progress bar)
├── MyCartsPanel.tsx          (left-pane browser of existing carts)
├── DeployPanel.tsx           (deploy to droplet — gated on bug fix)
└── store/cartBuilderSlice.ts (Zustand slice or sub-store for workspace state)
```

State machine (workspace):
- `idle` — no cart, no files
- `editing-new` — files dropped, metadata being edited, no build yet
- `editing-existing` — opened existing cart, can add/remove/replace
- `building` — build in progress, status polling active
- `built` — build complete, search now available against new cart

## Backend port

Two paths, in order of preference:

**A. Direct module import (preferred)** — same pattern as Membox visualizer:
sys.path patch in `vector-plus-studio-repo/api/main.py` to import
`cart-builder/cart-builder/builder.py`, `parsers.py`, `cartridge_builder.py`
directly. New FastAPI routes wrap them. No Flask process needed.
Pro: single backend process, single port (8000). Con: tighter coupling.

**B. Flask reverse-proxy** — VPS FastAPI proxies `/api/cartbuilder/*` to a
sidecar Flask process on a separate port. Pro: zero changes to existing
Cart Builder code. Con: two backend processes, more deployment friction.

Default to A unless we hit specific reasons not to (e.g. dependency
conflicts between Flask app's pinned versions and FastAPI's).

## Phasing

**Phase 1 (next session):** Backend port via Option A. All 14 routes
mirrored as `/api/cartbuilder/*` in VPS FastAPI. Smoke-tested with curl
against existing fixtures. ~2-3 hours.

**Phase 2:** Frontend skeleton — DropZone + FileGrid + FileCard wired
to upload/files/ingest/metadata. Can drop files, see them, edit
metadata. No build yet. ~2-3 hours.

**Phase 3:** Build flow + Pattern0 preview + progress bar. Manual
build + reload-and-search round trip working. ~1-2 hours.

**Phase 4:** Edit-existing-cart flow (load_cart + replace + has_changes
+ unsaved-warning banner). Fixes one of the 2026-04-03 filed bugs as
a side effect. ~2 hours.

**Phase 5:** Deploy panel — but only after the deploy-endpoint bug is
fixed in upstream cart-builder repo. Coordinate with that repo's owner
(probably Andy + me).

## Interim solution (this commit)

`CartBuilderScreen.tsx` ships tonight with a real-ish layout (drop zone,
empty file grid, disabled build button) so the screen isn't dead and the
visual shape matches what's coming. Plus a clearly-labeled fallback button
to launch the standalone Flask app at `localhost:5000` so users have an
immediate working path.

## Bugs to clean up during port

- **Open Cartridge file picker hang** (filed in `BUGS.md`) — same root
  cause as Cart Builder's `/browse` route. Fix once, apply both places.
- **Cart Builder deploy endpoint** (filed 2026-04-03) — fix in upstream
  cart-builder before Phase 5.
- **"Unsaved changes" warning banner** (filed 2026-04-03) — already
  scaffolded by `/has_changes` route; just need UI.
