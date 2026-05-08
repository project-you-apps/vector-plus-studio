# Vector+ Studio Devlog

## 2026-05-08 — WebGPU Cart Builder V2 Blocks 1-4 ship + hardening pass + eject-sandbox feature

**One-day burst.** Started the morning checking session memory + Uber TODO (Blocks 1+2 were Friday's planned scope after the angel deck took yesterday). Ended the afternoon with the entire WebGPU Cart Builder V2 pipeline shipped through Block 4, hardening pass on the upload path, an eject-sandbox feature for privacy-control, and 12 npm vulnerabilities cleaned up. The day's pace surprised me — clean specs from the Wed-evening smoke tests + tight Python-reference parsers.py made each block roughly mechanical. Block 5 (parity test) needs a browser at the keyboard; deferred to Saturday.

### Live demo URL

**<https://project-you.app/vps/app/>** — unchanged, still serving the three curated carts. New BrowserCartBuilder lands in this codebase but isn't deployed to the droplet yet (lands as part of the V1 launch deploy mid/end-of-next-week, Portland Startup Week).

### What shipped

**WebGPU Cart Builder V2 (Blocks 1–4)** — pure client-side parse → chunk → embed → write → download:

- **Block 1 (parsers + chunker, ~280 LOC):** `cart-builder-v2/types.ts`, `parsers/{pdf,docx,xlsx,text}.ts`, `parsers/index.ts` (registry-style dispatch with `registerParser()` extension hook for next-week JSON/mbox/HTML adapters), `chunker.ts`. Output shape mirrors `cart-builder/cart-builder/parsers.py` (300-word/50-overlap chunks, sections include `text`/`page`/`source`/optional `part`). Small deviation: CSV included in xlsx parser (free via SheetJS, common format).
- **Block 2 (embedder, ~200 LOC):** `embedder/loader.ts` singleton with WebGPU→WASM auto-fallback + `getActiveBackend()` introspection; `embedder/embed.ts` batch mode (default 16/batch, configurable, progress callback) + `embedQuery()` convenience with `search_query:` prefix. Nomic v1.5 prefix convention baked in. Custom callable interface around transformers.js's `pipeline()` to dodge the union-type narrowing issue. Verified at 6th-decimal parity with Python reference per the Wed smoke test.
- **Block 3 (writer, ~370 LOC):** `writer/{npy,hippocampus,manifest,permissions,npz}.ts`. Hippocampus packer matches `api/cartridge_io.py:HIPPO_FORMAT` exactly (offsets locked, perms_byte at 29). Manifest fingerprint matches `tools/bootstrap_claude_journal.py:compute_membot_fingerprint`. Critical late discovery: **npyjs supports `<U${number}>` unicode arrays**, so passages.npy is real NPY format (no JSON sidecar workaround needed) — browser-built carts mount on the existing membot server without server-side changes.
- **Block 4 (pipeline + UI integration):** `cart-builder-v2/pipeline.ts` orchestrator with PipelineProgress streaming events; `components/BrowserCartBuilder.tsx` self-contained UI (drop zone + cart-name + queue + progress + download). Lazy-loaded into `CartBuilderScreen` so the transformers.js bundle only ships when a user opens that screen. Defensive caps in pipeline (50MB per file, 10K chunks per build) prevent browser OOM on hostile inputs.

**Bundle impact:** lazy-load split dropped main bundle from **2.28MB → 480KB** (gzip 314KB → 142KB). Cart Builder visitors get the +1.8MB transformers.js chunk on first visit; Search-only users never pay that cost. ~23MB ONNX WASM binary loads only when the user actually runs the embedder (and only on browsers without WebGPU).

**Hardening pass on upload path** (`api/uploads.py`):

- **Deep structural NPZ validation** (`_validate_npz_structure`): testzip integrity check + zip-slip defense (no `..`, no absolute paths, no backslashes in entry names) + zip-bomb defense (per-entry compression-ratio cap 200x + total-uncompressed cap 8× upload size) + entry-type allowlist (`.npy` only). Validates in-memory before any disk write — malformed/hostile carts never persist even briefly.
- **`.pkl` upload dropped from public-demo allowlist.** Closes the pickle-deserialize-RCE-on-mount vector. Frontend `SearchToolbar.tsx` accept attribute + tooltip updated to match. Private deploys can re-enable by extending `ALLOWED_EXTS`.
- **`npm audit fix`** — 12 of 13 vulnerabilities fixed (vite, rollup, postcss, picomatch, minimatch, flatted, brace-expansion, ajv). One remaining: `xlsx` (SheetJS) has prototype-pollution + ReDoS with no available fix — npm package abandoned upstream. Browser-side impact only (slow tab on hostile xlsx upload, not server-side). Launch-acceptable; follow-up to evaluate `@e965/xlsx` community fork.

**Eject-sandbox feature** (privacy/control improvement):

- Server: `DELETE /api/cartridges/eject?cart_path=…` endpoint with sandbox-only path validation (relative_to-resolved check), refuses-if-mounted (caller unmounts first), deletes file + `.permissions.json` sidecar.
- API surface: `StatusResponse` extended with `mounted_is_sandboxed: bool` + `mounted_path: str | None` so the UI can detect when the current mount is a user upload.
- Frontend: `Trash2`-icon Eject button in `SearchToolbar` next to the cart picker, only visible when `status.mounted_is_sandboxed === true`. Confirms, unmounts, ejects, refreshes status. Privacy story upgrade — users no longer wait up to 1h for TTL eviction.

**Other audits:**

- VPS API audit — no `.env` files in repo; CORS locked to localhost (OK if same-origin via nginx); all write routes call `_enforce_writable()`; no real credential matches in repo grep.
- Console.log audit — all 30+ `console.error` calls are legitimate error-path logging; no chatty `console.log` clutter.
- Security sweep on `membot`/`membraine`/`nudges` deferred (separate repos, not in this tree — needs `gh` CLI or repo clones).
- nginx per-IP rate limiting deferred (config lives droplet-only, no committed copy in repo) — needs droplet ssh.

### Strategic context

Andy's stated launch target shifted today: **Portland Startup Week, middle/end of next week** (Mon-Wed window). He'll be meeting enterprise prospects at PSW and wants a finished product to demo. Engineering scope additions named for the launch: Obsidian import/export (Sunday) and 9 EnterpriseRAG-Bench source-type adapters (Slack/Gmail/Linear/Drive/HubSpot/Fireflies/GitHub/Jira/Confluence — Sunday/Monday). Droplet capacity bump from $28/mo is cost-blocked (family-finance negotiation, finished product is part of winning that conversation). Cost-of-bump is one of the few acceptable launch-delay triggers.

### What's left for V1 launch

Saturday: Block 4+5 completion (parity test against `attention-is-all-you-need.cart.npz` reference). Sunday/Monday: marketing artifacts (tagline, description, screenshots, demo video, Reddit drafts, ProductHunt page, HN Show-HN draft), operational readiness (privacy/terms, analytics, error monitoring, Discord/contact channel, traffic runbook), documentation (README, quickstart, FAQ, license). Mon-Tue: OAuth (Supabase Auth + Google/GitHub/Apple) as part of finishing VPS — same auth landing unblocks Membot / Membox / Heartbeat / Pattern 0 v2 Ownership Block / Cart Marketplace identity.

### Memory hygiene (today's filings)

Three new feedback memories in `memory/`:

- `feedback_powershell_no_approval.md` — routine PowerShell/Bash commands (typecheck, dev server, builds) just run, no approval gate. Same trust level as memory autonomy. Project-level `.claude/settings.local.json` updated with `"PowerShell(*)"` allow rule via the `update-config` skill.
- `feedback_systems_override_runaway_acceptable.md` — Andy 2026-05-08: don't default to "prevent runaway" framing for autonomous Looper/cron design. Runaway IS the goal. Brake (1985-vintage "Systems Override - User Level") is rarely-fired emergency stop, not routine governor.

One project memory updated: `project_angel_deck_in_flight_2026-05-07.md` retired → `project_angel_deck_shipped_2026-05-07.md`.

One CC filed: `CC_enterprise-rag-bench_2026-05.md` — 500K-doc realistic enterprise RAG benchmark from Onyx, three uses (500K cart benchmark + 9 ingestion adapters + marketplace flagship cart), plan post-VPS-ship.

---

## 2026-05-05 → 2026-05-06 — Cart Builder full-parity port + Edit Carts CRUD screen + three-layer RWX + public droplet deploy + WebGPU pivot smoke-tested

**Two-day burst.** 18 commits, 5 new CCs, 2 feedback memories, public demo live, the entire RWX security stack designed and shipped, Cart Builder ported in full, Edit Carts (CRUD) screen built from scratch, and the WebGPU pivot for V2 Cart Builder de-risked via three smoke tests on Wed evening. The kind of stretch where the architecture you've been talking about for weeks becomes the architecture in production. Walked into Tuesday night thinking "Cart Builder Phase 1 backend port"; walked out Wednesday at 10pm with a public-facing demo, a security-sound multi-layer permissions model, and the foundations for a private-by-architecture in-browser cart-build pipeline.

### Live demo URL

**https://project-you.app/vps/app/** — read-only public demo, 3 curated carts (attention paper, gutenberg-poetry, neuroscience-and-AI-papers), upload-your-own-cart sandboxed flow with TTL eviction, end-to-end mount + search verified.

### Commits this session (chronological)

| # | Commit | What it shipped |
|---|---|---|
| 1 | `fe1d557` | Cart Builder Phase 1: 15 backend routes ported as `/api/cartbuilder/*` (FastAPI APIRouter mirroring the Flask app). CRUD scope explicitly removed per the 2026-05-05 IA decision. |
| 2 | `a3f1d7b` | Markdown rendering in PassageModal — `react-markdown` + `remark-gfm` for headings/lists/tables/blockquotes/code/autolinks. Per-element Tailwind classes, no `@tailwindcss/typography` dep. |
| 3 | `1702cdb` | Cart Builder Phase 2 scaffold — typed API client `frontend/src/api/cartbuilder.ts` wrapping all 15 routes. Tree-shaken until imported. |
| 4 | `4a442e7` | Edit Carts (CRUD) screen mockup — first pass with two modes, three op panels, tombstone list, in-screen activity log, sticky save bar. Wired to existing endpoints. |
| 5 | `78adebb` | **Step 1 of RWX roadmap:** `VPS_READ_ONLY` env-var server-wide lock. Every write endpoint refuses with 403; unlock specifically refuses so per-cart locks can't be cleared. |
| 6 | `eaed301` | **Step 2a of RWX roadmap:** cart-format permissions sidecar (`<cart>.permissions.json` with `default: r/rw/rwx`, optional `owner`, `description`). `_enforce_writable` composes server flag + cart sidecar. CLI tool `bin/set_cart_permissions.py` for retrofitting. |
| 7 | `752e3be` | Security sweep + `.gitignore` hardening — caught nested-archive `claude_sessions.txt` files that the original `cartridges/*.txt` pattern would have missed. New globals `**/claude_sessions*`, `**/session_memory*`, `**/memory_keys*`. |
| 8 | `0512560` | Frontend `VITE_BASE` + `VITE_API_BASE` env vars for hosted deploys. Local dev unchanged; droplet build sets `/vps/app/` + `/vps/api`. |
| 9 | `f29cc71` | `list_cartridges` discovers `.cart.npz` (membot format) too, not just `.pkl`. Caught during droplet deploy when the new carts didn't appear in the list. |
| 10 | `5b1ac66` | **Cart Builder Phase 2: full-parity React port.** New `cartBuilderStore.ts` zustand slice, `CartBuilderScreen.tsx` rewrite, shared `CartBrowser.tsx` component embedded in BOTH Cart Builder and Edit Carts (per Q2 = both), `FolderPickerModal.tsx`. Drop zone + file metadata editor + Pattern 0 preview + sticky build bar. |
| 11 | `6e44935` | Phase 2 polish (1/3 + 2/3): screen-wide drag-drop overlay + `Toaster` system. Window-level dragenter/dragleave with file-only filter. |
| 12 | `dc8cd21` | Phase 2 polish (3/3): `@uiw/react-md-editor` for description fields, lazy-loaded so the ~315KB chunk only downloads when user opens Metadata. Initial bundle stayed at 140KB gzip. |
| 13 | `1c08632` | Light-mode fix — global CSS for `html.light` flips form-field background/text/placeholder/focus. Edit Carts/Cart Builder text inputs no longer dark-on-dark. |
| 14 | `e2e7337` | SearchToolbar: hide native file picker + paste-path on public demo (PowerShell doesn't exist on Linux droplet, users can't see server filesystem anyway). |
| 15 | `fbc2b01` | **Sandboxed upload endpoint** for the public-demo Open Cartridge picker. `POST /api/cartridges/upload` with magic-byte validation, size cap, UUID prefix, forced read-only sidecar, TTL cleanup task running from lifespan. |
| 16 | `4a8e22e` | Frontend Upload Cartridge picker — `<input type="file">` + auto-mount + toasts. Visible in both local and droplet deploys; chains upload → mount → toast in ~3 seconds. |
| 17 | `fbdcf4b` | **Step 2b of RWX roadmap:** hippocampus per-pattern RWX in the reserved-area `perms_byte` (offset 29). Pivoted from the flags byte at offset 28 after discovering membot's cart_builder.py owns it for tombstone/pinned/has_parent/has_child/has_sibling/perish. CLI `bin/set_pattern_permissions.py` with range syntax. |
| 18 | `0ae41cb` | **SECURITY** — gate `/api/cartbuilder/browse` and `/api/cartbuilder/carts?path=X` behind read-only mode. Andy discovered the public droplet was exposing arbitrary directory structure to any visitor. Closed the recon vector. |
| 19 | `9c579dd` | Cart Builder read-only-mode UX + `ConfirmDialog` reusable component + light-mode contrast bumps for accent banners. Hide drop zone, workspace, Pattern 0, build bar, Folders button, subdir drill-down in read-only mode. |
| 20 | `afac582` | Edit Carts activity log: proper `mount`/`unmount`/`save`/`create`/`open` kinds with color-coding. Was previously logging cart-mount as `add` (sloppy). |
| 21 | `3b0f8c3` | Edit Carts: friction-by-design banner + zero-byte placeholder filter in `list_cartridges` + mount-failure toasts (closed a long-standing silent-fail UX hole). |
| 22 | `600938f` | **WebGPU pivot prep** — installed `@huggingface/transformers`, `npyjs`, `jszip`, `pdfjs-dist`, `mammoth`, `xlsx`. Smoke-test artifacts moved to `tools/webgpu-smoketest/` with full README capturing Thursday/Friday plan. |

### RWX three-layer security stack — the architectural piece

Designed and shipped end-to-end in a single day. Composes top-down, fail-fast:

1. **Server-wide flag** — `VPS_READ_ONLY=1` env var refuses ALL write endpoints. Unlock specifically refuses too, so attackers can't bypass via the per-cart lock toggle. Blanket "this server is a public demo" lock.
2. **Cart-format sidecar** — `<cart>.permissions.json` with `default: r/rw/rwx`, optional `owner`, `description`. Loaded at mount; gates writes via `cart_permits_write()`. Backward-compat: absent sidecar → writable. **The cart self-declares its policy.**
3. **Hippocampus per-pattern bits** — `perms_byte` at row offset 29 (NOT the flags byte at offset 28 — that's owned by membot's cart-builder for tombstone/pinned/links/perish). Bits `0x01=R / 0x02=W / 0x04=X-reserved`. `_enforce_writable(idx=…)` composes all three layers.

Two CLI tools for retrofitting existing carts without rebuild: `bin/set_cart_permissions.py` for cart-wide, `bin/set_pattern_permissions.py` for per-pattern with range syntax (`--idx 5,12-15,42 --perms r`).

Frontend surfaces all three: Header shows three lock states in priority order (Public read-only > Cart read-only > interactive Lock/Unlock). Result cards show LOCKED badge when `result.perms.w === false` with tooltip explaining the byte value.

### Cart Builder Phase 1+2 — port complete with full parity

Backend (Phase 1, commit `fe1d557`): 15 routes from `cart-builder/cart-builder/app.py` mirrored as FastAPI APIRouter at `/api/cartbuilder/*`. Late-imports `parsers`, `builder` from cart-builder source via temporary sys.path insertion (no module collisions with VPS-local). Returns 503 if cart-builder modules unavailable (which is the intentional droplet state).

Frontend (Phase 2, commit `5b1ac66` + polish in `6e44935`/`dc8cd21`): zustand store `cartBuilderStore.ts`; rewritten `CartBuilderScreen.tsx` with drag-and-drop, inline file metadata editor, live Pattern 0 preview, sticky bottom build bar (matches Edit Carts save bar per Q4); shared `CartBrowser.tsx` embedded in BOTH Cart Builder and Edit Carts (per Q2); `FolderPickerModal.tsx` server-side path browser; lazy-loaded `@uiw/react-md-editor` for description fields (zero growth on initial bundle); `Toaster` system at App level; screen-wide drag-drop overlay with file-only filter and dragCounter for nested-child correctness.

All four IA decisions Andy made on 2026-05-05 are baked in:
- **Q1**: Cart Builder = canonical "open existing cart" target; Edit Carts is manual nav
- **Q2**: CartBrowser = single shared component, two screens
- **Q3**: Full parity (we're going to Reddit / Product Hunt with this)
- **Q4**: Sticky bottom bar (consistent with Edit Carts save bar)

### Edit Carts (CRUD) screen — friction-by-design

New `CRUDScreen.tsx` (the Pencil-icon nav entry). Two modes (Open Cart / New Cart), three op panels (Add / Update / Delete), tombstoned-passages list with one-click Restore, in-screen activity log with kind-coloring (mount/unmount/save/add/update/delete/restore/create/open), sticky save bar when dirty.

Andy's explicit design call (2026-05-05): **Edit Carts SHOULD be friction.** The destructive screen — passages get tombstoned, updated, deleted here, this is the rm-rf screen of VPS. Asymmetric to Search Screen (permissive mount, read-only is safe). No casual mount affordances — users either mount via Search Screen first, or use the deliberate My Carts panel embedded at the bottom.

ConfirmDialog reusable component shipped with this screen — title + rich React body + destructive flag + async onConfirm. Esc-cancel, Enter-confirm. Wired into tombstone + update; future destructive actions across the app adopt the same pattern.

### Public droplet deploy — went live, then immediately patched a security leak

Cloned VPS into `/opt/vector-plus-studio/`, Python 3.11 venv, copied 3 carts as a curated subset (no symlink to `/opt/membot/cartridges/` because Andy correctly worried it would inherit any future personal cart that landed there), set `default: r` permissions sidecars on each, frontend built with `VITE_BASE=/vps/app/ VITE_API_BASE=/vps/api`, scp'd to droplet, systemd unit `vector-plus-studio.service` with `Environment=VPS_READ_ONLY=1`, nginx subroute added. End-to-end test mounted attention paper, ran "transformer attention" search at 593ms, verified unlock → 403.

**Then Andy found the leak.** The Cart Builder folder picker, even on the public droplet, was happily walking arbitrary server paths via `/api/cartbuilder/browse` and `/api/cartbuilder/carts?path=X`. Public visitors could enumerate `/etc/`, `/opt/`, `/root/`. Closed in commit `0ae41cb` — both endpoints 403 in read-only mode (browse always; carts only when `path` query param is non-empty so the safe no-path saved-folders variant still works).

Standing items filed in [`memory/concept-clusters/CC_security-sweep-and-pii-scrubber_2026-05.md`](../../memory/concept-clusters/CC_security-sweep-and-pii-scrubber_2026-05.md): audit `membot` / `membraine` / `nudges` repos with same gitignore + grep, build `cart_audit.py` for binary-cart content scanning, design PII scrubber (`${env:KEY}` pointer pattern with secrets in local `.env`).

### WebGPU pivot for Cart Builder V2 — Wednesday evening smoke-tested

Andy's strategic question (Wed evening): server-side build-and-download as bridge, OR pure client-side WebGPU as the proper architecture? Three counter-arguments to pure WebGPU enumerated; recommendation refined to **WebGPU primary + WASM auto-fallback (no server build path)** — `transformers.js` handles capability detection internally, the privacy pitch stays clean ("100% in your browser, fast where the GPU is, correct everywhere").

Three smoke tests retired the foundational risks before betting Thursday-Friday on the pivot:

| Test | Result |
|---|---|
| **1. WebGPU lattice engine alive** | Existing `vector-benchmark-demo/cuda/webgpu/` works in Chrome. 4096×4096 init OK. Random imprint 151ms, train 30 frames (Hebbian, learn=true) 318ms, signature gen 49ms, L4 state recall, L3 visualization render. **12ms/frame on 16.7M neurons.** |
| **2. Embedder parity (Python vs in-browser transformers.js)** | Same sentence embedded both ways. **Drift at the 6th decimal place** — noise floor. L2 norm 21.072100 (Python) vs 21.072095 (browser). Carts built in browser will be bit-near-identical to carts built server-side. **Cross-compatibility confirmed.** |
| **3. NPZ writer feasibility** | Membraine fetched the actual `npyjs` README — `dump()` is supported (I had a stale assumption that npyjs was read-only). Apache-2.0, maintained, every dtype we need. NPZ via JSZip. ~30-50 lines total instead of the half-day I estimated initially. |

Smoke-test artifacts committed to [`tools/webgpu-smoketest/`](../tools/webgpu-smoketest/) — `embedder-test.html`, `python_reference.py`, `python_reference_full.npy`, README with Thursday/Friday five-block plan + effort estimates + canonical-code-pointers.

### CCs filed today (5 new)

| CC | What it captures |
|---|---|
| `CC_security-sweep-and-pii-scrubber_2026-05.md` | Initial sweep findings (VPS GitHub clean; membot droplet public surface clean; .gitignore hardened); standing items (audit other repos, cart_audit.py, PII scrubber design) |
| `CC_sql-database-ingestion_2026-05.md` | V2 sketch — `.sql` files as a Cart Builder source type; rows-as-passages, schema-as-manifest, FK-as-hippo-edges; table-per-cart agent-swap angle for large enterprise DBs (extends multi_mount infrastructure already shipped) |
| `CC_cloud-storage-for-carts_2026-05.md` | Cloudflare R2 default for cart blob storage (no egress fees, salesperson cold-contacted Andy twice in 10 days = partnership opening when ready); pluggable `CartStorage` interface; 5-phase rollout; aligns with existing Heartbeat Tier 0/1/2/3 model |
| `CC_realtime-l4-visualization-dashboard_2026-05.md` | "When the cognition engine is running, pin the L4 grid at 24fps as a dashboard." Same engine we just smoke-tested makes this newly feasible. Evolution of Hot Stack dashboard — the lattice IS the dashboard. V1.2+ feature. |
| Plus the existing `CC_provenance-as-feature_2026-05` — touched during the activity log + light-mode work |

### Feedback rules filed (2 new)

| Rule | Why |
|---|---|
| `feedback_check_session_cart_before_guessing.md` | When about to guess a value/path/config we've touched before (HuggingFace model paths, install commands, port numbers, API endpoints, model IDs), search session memory FIRST. The asymmetry: 3-second search vs full Andy-roundtrip on a wrong guess. Triggered by Wed-evening Xenova-vs-nomic-ai Nomic ONNX path miss. |
| `reference_droplet_ops.md` | (Filed earlier in the week, used heavily today) — SSH key, host, deploy pattern, target paths so future sessions don't re-derive |

### IA decisions locked

- **Cart Builder vs Edit Carts** (Q1): one button → Cart Builder on cart click; passage editing requires manual nav to Edit Carts
- **CartBrowser placement** (Q2): single component, embedded in both screens
- **Phase 2 scope** (Q3): full parity (this is what gets posted to Reddit/PH)
- **Build progress UX** (Q4): sticky bottom bar matching Edit Carts save bar
- **Edit Carts is destructive-by-design**: friction is the feature. No casual-mount affordances.
- **Open Cartridge on public demo** = client-side `<input type="file">` upload to sandbox + TTL eviction (not server-side path browser, which leaks filesystem)
- **WebGPU pivot architecture**: pure client-side build, transformers.js handles WebGPU/WASM fallback automatically (NO server-side build path; preserves privacy story)

### Outstanding work surfaced today

- **Saturday hardening pass** (~4h): streaming uploads, per-IP rate limiting, deep magic-byte/structural NPZ check, eject-sandbox button, full security punch-list. Pinned, deadline Sat afternoon.
- **Step 2b's natural follow-up — Step 3** (user-auth + per-user ACL, ~10-20h): defer until site-wide login is built; until then, sidecar `owner` is metadata only.
- **Cloud storage Phase 1+** (CartStorage Protocol, R2 backend): no immediate need; pinned for when cart marketplace work spins up.
- **WebGPU Cart Builder V2** (Thu-Fri): five-block plan in [`tools/webgpu-smoketest/README.md`](../tools/webgpu-smoketest/README.md).
- **Cart Builder build-and-download** ABANDONED in favor of pure client-side WebGPU per Wed-evening pivot decision.

### Net effect on the product

Tuesday morning: VPS was a single-user local app with a buggy partial-CRUD UI.

Wednesday night: VPS is a publicly-deployed read-only demo, three-layer-secure cart-format-RWX architecture, full-parity Cart Builder UI, dedicated Edit Carts screen with confirmation dialogs, sandboxed upload-your-own-cart for evaluators, and 80% of the way to a 100%-in-browser private-by-architecture cart-build pipeline that ships Friday.

The product crossed a threshold from "demonstrable in a meeting" to "shareable on Reddit/HN with a URL."

---

## 2026-05-04 (afternoon → past midnight) — Substrate-claim crystallization + RAG+ port + split-cart parity

A long working day that landed two product upgrades, four strategic CCs, the V1 / V1.2 product-versioning frame, the CRUD-as-own-screen architecture decision, and corrected a marketing claim ($12 → $28 droplet cost) across multiple docs. Plus engaged with a NodeMind competitive-landscape discovery via /r/Rag, ran BEIR-vs-physics empirical comparisons, and tracked Dennis's first Kaggle ARC-AGI-3 reconnaissance submission (which surfaced a Kaggle-scored-env network-isolation issue blocking the SDK).

### Commits this session (chronological)

1. **`23d4c92` — Provenance port (partial) from Membot demo to VPS.**
   New `renderTextWithLinks(text, query)` in `ResultCard.tsx` — React-node-tree-safe URL linkification that wraps the existing stem-aware highlightText for non-URL segments. Used in result preview, expanded card, and modal body. Modal also gets a small source line: `source: <cart_name> · pattern #<idx>`. NOT ported tonight: split-cart load-source CTA + paper_id surfacing — those need backend support that VPS didn't have yet (see commit `9cc0790`).

2. **`9cc0790` — Split-cart support for VPS local backend (parity with membot).**
   210 lines across 6 files. Brings VPS into format-alignment with membot — same `.cart.npz` (+ optional `.db` sidecar) mounts in either, with the SQLite providing full passage text on demand.
   - `engine.py` gets `is_split_cart`, `sqlite_conn`, `sqlite_db_path` fields + cleanup in `unmount()`.
   - `api/main.py` gets `_sqlite_fetch_passages()` helper with the load-bearing `int(idx)` cast (sqlite3 binds numpy.int64 silently as no-match — same bug class as membot fixed 2026-05-03).
   - `_mount_membot_npz` detects `has_sqlite=True` NPZ field, opens sidecar at `<cart_dir>/<text_db>`, uses `snippets` field as in-RAM `passages`.
   - `/api/search` populates `source_db` on each result for split carts (paper_id deferred to lazy fetch).
   - `/api/patterns/{idx}` fetches full passage from SQLite when split-cart, returns `source_db` + `paper_id`.
   - `models.py` adds optional `source_db` + `paper_id` to `SearchResult` and `PatternResponse`.
   - Frontend `types.ts`, `appStore.ts` thread the new fields through. New `loadSourceForCurrentPassage` action fires the lazy fetch.
   - `PassageModal.tsx` re-enables the "📂 Load full passage from <db> →" CTA when split-cart, shows full source line (`source: <db> · id: <paper_id>`) after load. Three-state UX matching the public Membot demo.
   - **Cart-format alignment is the structural prerequisite for cart-marketplace endgame** — same artifact moves between Membot and VPS now.

3. **`41bfe39` — Brain / Sigs / SHA badge legend on Overview screen.**
   Cart-row badges had no explanation visible to a new visitor. Added an inline legend strip below the section header + per-badge title-attribute tooltips for hover detail. Brain = trained Hebbian weights (physics modes); Sigs = L2 signatures cached (pure-brain mode); SHA = SHA-256 integrity manifest (mount-time verify).

### Strategic / framing crystallization (in memory, not in repo)

#### V1 / V1.2 product-version reframe

Initial pass tonight introduced "Track A / Track B" framing in the CC. Andy correctly pointed out that **version-numbered roadmap framing is more honest and stronger** — *"V1 today, V1.2 with physics coming soon"* commits to delivery rather than dual-bucketing. The LatticeRunner Declaration already promises physics publicly, so V1.2 isn't optional; it's the delivery commitment. Same answer to *"where's the physics?"* as Postgres or Supabase give about a planned minor: *"1.2, coming soon."* No track-jargon, no apparent bait-and-switch.

CCs updated to use V1/V1.2 throughout: `CC_vps-as-hybrid-dbms_2026-05.md`, `CC_physics-earns-keep-with-teachable-data_2026-05.md`, MEMORY.md index entries.

#### CRUD architecture: own screen (V1.x)

Decision locked tonight. CRUD becomes the 6th nav-rail screen alongside Search / Overview / Cart Builder / SQL / Settings. Two modes: **open-cart** (rwx-gated CRUD on mounted cart) and **new-cart** (blank page, type-to-add only). File ingestion explicitly stays in Cart Builder. Operations: Add (plain text box, future-maybe a tiny markdown editor like react-mde), Update (relocated PassageEditor), Delete (tombstone current pattern). This *reduces Cart Builder Phase 1 scope* — drop `/remove`, `/restore`, `/replace`, `/delete` routes from the port plan; those live in CRUD now.

#### $12 → $28 droplet-cost correction across docs

Andy noticed the `$12/mo` claim was outdated marketing — actual cost is **$28/mo for the 4 GB DigitalOcean droplet** with split-cart architecture. Patched live across:

- `Pitches - 2026/Vector+ Executive Summary-5th-Draft.md` (clean edit, outgoing artifact)
- `memory/cart-roadmap.md` (strikethrough + correction, per the standing strikethrough rule)
- `docs/DEVLOG-PROJECT-YOU.md` (3 instances, strikethrough + correction)

Archive files (`MEMORY_pre-trim.md`, `old-MEMORY.md`) deliberately untouched — those are point-in-time snapshots.

### CCs filed today (5 new)

| CC | What it captures |
|---|---|
| `CC_competitive-landscape-nodemind_2026-05.md` (morning) | NodeMind from /r/Rag — first direct competitor in "binary index replaces vector DB" pitch space; 2 provisional AU patents; substrate-vs-feature framing; patent posture review; BEIR-as-rebuttal action |
| `CC_substrate-validates-across-scales_2026-05.md` (mid-day) | Two independent validation threads converging on the architecture-is-the-moat claim — pathology pilot (4 encoders × 23× scale, encoder-stable signatures) + Dennis's Kaggle (lp85 cross-model 1505/853/2367 px breakthroughs match across e4b vs 26B + T4×2 vs 4090) + vc33 warm-start 54× flywheel demo |
| `CC_vps-as-hybrid-dbms_2026-05.md` (mid-day, updated late-night with V1/V1.2 + materialized-view + WebGPU-second-backend sections) | Product reframe — VPS-on-droplet as world-first free hosted hybrid DBMS (cart-as-table, SQL-as-power-tool); DBMS-concept ↔ VPS-realization mapping; honest pitch script; 7-phase roadmap |
| `CC_physics-earns-keep-with-teachable-data_2026-05.md` (evening) | Honest empirical answer to "what's the enterprise pitch for physics?" — physics doesn't beat cosine on static data (BEIR sweep + dinosaurs case + README audit confirm), earns its keep when data becomes teachable (cognition engine V1.2+); V1/V1.2 product framing originated here |
| `CC_cart-physics-level-metadata_2026-05.md` (late evening) | Operational tracking — carts should record physics processing in Pattern 0 manifest (Q30 / Q300 / B100 / F0). Pairs with above CC to enable correct UI gating + marketplace pricing on physics-trained carts |

### Empirical work — BEIR with sign-zero blend

Built a fresh `tools/benchmarks/run_beir.py` driver in the membot repo (commits `484b02c` through `f964990`). Direct in-process search, no server needed. Ran SciFact + FiQA blend sweeps (0.0 / 0.3 / 1.0):

| Dataset | Pure cosine R@10 | 70/30 blend R@10 | Pure binary R@10 |
|---|---|---|---|
| SciFact (in-distribution) | 0.8396 | 0.8402 | 0.7796 |
| FiQA (real OOD) | 0.4438 | 0.4356 | 0.3691 |

**Headline:** *Membot's 70/30 cosine + sign-zero Hamming blend matches pure float32 cosine recall within 1pp on both in-distribution (SciFact) and OOD (FiQA), at 32× compression. Pure sign-zero binary alone retains 83-93% of cosine's recall, degrading more on OOD as expected, but never collapsing.* Direct rebuttal to NodeMind's asserted-but-unmeasured "binary breaks down on OOD" claim. Plus reaffirms physics-doesn't-help-on-static-data (the V1.2 thesis).

### Dennis context (informational, not blocking us)

Dennis's first Kaggle ARC-AGI-3 reconnaissance submission ran today.

- **Interactive editor success:** lp85 L=4 reached, breakthrough deltas (1505/853/2367) matched Thor's local 26B numbers exactly. vc33 cold→warm 54× flywheel demonstrated in-session. Architecture works on Kaggle T4×2 with Gemma 4 e4b.
- **Scored-env infrastructure block:** `arc_agi.Arcade()` requires network on construction even in `operation_mode="offline"` and the scored Kaggle environment has no network at all (Kaggle Secrets unreachable, `*.arcprize.org` doesn't resolve). Dennis's scored submission lands at zero levels — not our bug, an SDK / Kaggle-network-policy issue. He's drafting feedback to ARC Prize as their #0 priority.
- **Sent today:** the `uplift_over_weights_only` coordination ping to Dennis — split proposed: he writes the value as sidecar against `entry_id` during training, our writer merges at federation time; gracefully degrades when missing because it enters scoring multiplicatively.

### Memory updates

- `feedback_no_droplet_hot_patches.md` already broadened earlier to cover all uncommitted work
- `feedback_session_startup.md` already includes the git-sweep step
- 5 new CCs (above)
- MEMORY.md index updated with all of them

### Tomorrow's queue (decided tonight)

In rough order:

1. **CRUD architecture detail finalize** (15-30 min — already spec'd in chat, just confirm before code)
2. **Cart Builder Phase 1** (backend route port, 2-3h, scope reduced per CRUD decision)
3. **Markdown rendering in PassageModal** (`react-markdown`, ~1-2h, can slot anywhere)
4. **Online VPS demo on droplet** (mirror of `project-you.app/membot/app`, 3-5h, ships everything together)

### Working tree state at EOD

- VPS repo: clean, in sync with origin (4 commits ahead at start of day, all pushed by midnight)
- Membot repo: clean, in sync (multiple BEIR-driver commits pushed)
- Cart-builder repo: 1 unrelated `tools/session_mcp_server.py` modification still pending (not from this session)

---

## 2026-05-04 (late evening) — Sidebar reorg arc + Cart Builder port plan

A 4-hour evening session that took the VPS frontend from a single crowded
sidebar to a Supabase-style nav-rail-plus-screens layout. Companion to the
morning's Membot demo polish (provenance feature). The architectural
reframe and the discipline rule learned this morning carried straight
through into this session.

### What landed (in commit order)

1. **`9d00bfa` — Membox visualizer recovery** — 891 lines across 8 files
   (api/main.py, api/models.py, frontend/src/components/MemboxPanel.tsx,
   plus types/client/store/App/Header). Originally implemented and
   tested 2026-04-11 per spec; sat in working tree for ~3 weeks before
   commit. Same class as the morning's Hamming hot-patch recovery — the
   session-start git-sweep rule (filed in
   `~/.claude/.../feedback_session_startup.md`) is designed to catch
   exactly this drift in future sessions.

2. **`a3230e6` — Phase 1: NavRail + activeScreen state** — 128 lines. New
   `components/NavRail.tsx` (56px icon-only rail, 5 screens). New
   `ActiveScreen` Zustand state. App.tsx renders rail + branches on
   activeScreen. Search screen kept verbatim; other screens stubbed via
   inline `ScreenStub` component.

3. **`dc96820` — Phase 2: cart picker → SearchToolbar** — `+206 / -155`.
   New `components/SearchToolbar.tsx` with cart picker dropdown
   (240→280px Supabase-style trigger; 384px menu with Open/paste/list).
   Sidebar trimmed by removing cart-list block + dead state + dead
   imports. Click-outside dismiss via shared mousedown handler.

4. **`d3354b5` — Phase 3: search mode → SearchToolbar** — `+121 / -95`.
   Lifted MODES const + isDisabled/isTraining/isReady logic from Sidebar
   to a second SearchToolbar dropdown. Blend slider preserved when smart
   mode is active. Sidebar continues to shrink.

5. **`392ce24` — UX polish** — Discord-style flyout tooltips on NavRail,
   "Cartridge:" label on the cart-picker trigger for affordance, new
   `BUGS.md` (severity A/B/C) with "Open Cartridge…" file-picker hang
   filed (later closed in commit 1748591 — was a transient stale-backend
   state, not an architectural bug).

6. **`9a71b7f` — Phase 4: real Overview screen** — 207 lines. 4-card stat
   grid (Engine / Mounted / Lock state / Cartridges), System Health panel
   with checkmark/alert rows, Available Cartridges list with badges. All
   data sources from existing Zustand store + 2s/10s polling. Read-only
   screen by design; mounting still happens from the Search toolbar.

7. **`2c83f5b` — Phase 5: real Settings screen** — 178 lines. Search
   Defaults section (top-K slider, Strict toggle, Exact phrase toggle),
   Appearance section (dark/light segmented control synced with Header),
   Backend section (URL display placeholder), future-settings dashed
   placeholder. Reusable ToggleRow + Section helpers.

8. **`e1f54b2` — Empty-state copy fix** — one-liner: ResultsList no-cart
   message now says "from the picker above" instead of pointing at the
   non-existent sidebar.

9. **`2e2dca0` — Phase 6: Cart Builder skeleton + port plan** — 261 lines
   (135 component + 124 plan doc). Recall-first sequence confirmed Cart
   Builder is fully shipped at project-you-apps/cart-builder commit
   c2fb03c v1.1 (2026-04-03) — different from Membox, working tree was
   clean. CartBuilderScreen.tsx promoted from stub to layout skeleton
   with drop zone, 3-card grid of future-feature placeholders, amber
   info banner pointing at the standalone Flask app at
   `localhost:5000`, "Open standalone Cart Builder" interim button.
   `docs/CARTBUILDER-PORT-PLAN.md` filed: 22 Flask routes mapped to VPS
   port disposition (14 to port, 2 reusable, 1 to skip), 5-phase
   implementation breakdown, 9-component frontend layout sketch, 3 known
   bugs to clean up during the port.

10. **`1748591` — Phase 7: kill Sidebar, widen NavRail** — `-152` net
    lines. Sidebar.tsx deleted. NavRail widened from 56px to 192px with
    icon + label per row. Training progress moved to a global amber
    banner below the Header. Build Cartridge / Add Passage / Tombstoned
    restore — Build Cartridge was redundant with the Cart Builder
    screen; the other two deferred to the CRUD architecture planning
    session (2026-05-04 tomorrow). Closed the file-picker bug as
    not-a-bug (transient stale-backend state).

### State at the end of the night

5-screen nav rail, 4 of them real (Search, Overview, Cart Builder,
Settings), 1 stub (SQL Editor). Membox panel still works as a
slide-out from the Header Activity icon. Search screen is much cleaner
— SearchToolbar at top with cart picker + search-mode dropdowns, no
sidebar. Training progress surfaces globally when active. Theme toggle,
mounted-cart pill, Save, Lock, GPU/CPU indicators all preserved in the
Header.

### Architectural reframe

Started the session with: *"sidebar is doing 9 things and it's
crowded."* Ended with: *"thin nav for screens, per-screen toolbars for
context, modals for one-shot actions, header for global system
state."* Same Supabase pattern Andy named at the start. The fact that
the Membox visualizer was already in this shape (slide-out from Header
toggle) made it a template — we generalized one existing pattern
rather than inventing one.

### Discipline rule applied successfully

The morning's Hamming-hot-patch recovery was the trigger for the new
"git is the only source of truth" rule. Tonight that rule caught the
Membox visualizer drift (pre-emptively, via the recall-first step
before the reorg). It also worked cleanly for Cart Builder — recall
first, found that the work IS committed upstream, no recovery needed.
Two-for-two. The session-start git-sweep step added to
`feedback_session_startup.md` should catch future instances even
earlier.

### What's left (besides SQL editor + cosmetics + CRUD planning)

- **Cart Builder full React port** — 5 phases per the plan doc, ~10
  hours total. Phase 1 (backend route port) is the unlock; phases 2-3
  give a working flow. Highest leverage remaining VPS work.
- **Online VPS demo on droplet** — mirror of Membot's
  `project-you.app/membot/app`. Build static, serve from FastAPI/nginx,
  secure-mode the backend (read-only-ish), pick a path. ~3-5 hours.
  The "shippable end state" cap-off of the project.
- **Provenance feature port from Membot** — `linkify()`, source-DB CTA,
  soft-truncate. Drop-in port to VPS PassageModal. ~30 min.
- **Markdown rendering in passage modal** — planned, ~1-2 hours.
- **CRUD architecture** — Andy's call; planning session 2026-05-04.
  Possibly its own screen, possibly modal-based. Add Passage and
  Tombstoned restore re-emerge here.

### Parallel: Membot demo polish (same session, earlier)

For the broader picture, see `membot/docs/DEVLOG.md` 2026-05-03 entry —
provenance feature shipped on the public demo as the new pitch
differentiator (RAG+, three-tier transparent provenance), CC filed at
`memory/concept-clusters/CC_provenance-as-feature_2026-05.md`.

---
