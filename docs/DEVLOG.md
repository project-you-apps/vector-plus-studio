# Vector+ Studio Devlog

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
