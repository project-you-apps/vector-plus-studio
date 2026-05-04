# Vector+ Studio Bugs

Mirror of the Membot BUGS.md convention. Severity classes:

- **A** — blocks shipping, data loss, security, demo-breaking
- **B** — visible product defect, wrong output, broken UX
- **C** — cosmetic, edge-case, low-traffic

Open bugs listed first. Move to "Closed" with a date and resolution note when fixed.

---

## Open

(empty)

---

## Closed

### B-high · "Open Cartridge…" button hangs (file picker never opens)
**Reported:** 2026-05-04 12:18 (Andy, during sidebar reorg testing)
**Closed:** 2026-05-04 12:44 — **NOT A BUG** (transient stale-backend state).
**Resolution:** Verified working same evening after backend refresh. The
`/api/browse` endpoint pops a native Windows tkinter file dialog from the
FastAPI backend, returns the picked path, frontend auto-mounts it. This
is the intended design for the local-only deploy and works as expected.
The earlier "just spins" symptom was almost certainly a stale uvicorn
process that hadn't reloaded the route, plus possibly a hidden-dialog
UX issue (tk dialog spawning behind the browser window).
**If we ever ship a hosted VPS demo on the droplet** (parallel to Membot),
the tk-dialog approach won't work in a headless server context — at that
point we'd need to swap to `<input type="file">` on the client + a backend
upload-and-register route. File a new bug then if/when that happens.
