# Vector+ Studio Devlog

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
