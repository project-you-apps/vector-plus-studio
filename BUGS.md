# Vector+ Studio Bugs

Mirror of the Membot BUGS.md convention. Severity classes:

- **A** — blocks shipping, data loss, security, demo-breaking
- **B** — visible product defect, wrong output, broken UX
- **C** — cosmetic, edge-case, low-traffic

Open bugs listed first. Move to "Closed" with a date and resolution note when fixed.

---

## Open

### B-high · "Open Cartridge…" button hangs (file picker never opens) — *priority bump 2026-05-04*
**Reported:** 2026-05-04 (Andy, during sidebar reorg testing)
**Priority note (2026-05-04 12:18):** Andy explicitly flagged this for the
near-term sprint queue — *"we should add that file picker for load cartridge
though at some point."* Likely the right fix is replacing the server-side
`tk` dialog with an `<input type="file">` on the client + a backend
upload-and-register route. Investigate at start of next VPS session.
**Where:** SearchToolbar cart picker dropdown (and previously in Sidebar; pre-existing — the migration to SearchToolbar did not introduce it).
**Symptom:** Clicking "Open Cartridge…" sets `pathLoading=true` and shows the spinner, but no native file picker dialog ever appears. The spinner spins indefinitely (or until the operation eventually times out on the backend).
**Root-cause hypothesis:** The `api.browseForCartridge()` backend route presumably uses a desktop file-picker (tkinter/native) that either:
1. Cannot open from the FastAPI server context (no GUI available, or wrong thread)
2. Opens a file picker on the *server* machine instead of the client browser
3. Returns silently when the user cancels but the UI never resets

**Fix scope:** investigate `api/main.py` `/api/browse` (or whatever the route is called) — is it using a server-side tk dialog? If so, this is fundamentally wrong for a web app. The right pattern is either:
- Use `<input type="file">` on the client (works for picking a file the user already has on disk and dragging it in)
- Or accept a path string typed into the existing "paste path" fallback
- Or use the File System Access API (Chrome-only but progressive enhancement)

**Workaround:** "or paste a path…" expandable in the same dropdown still works — paste a full path to a `.pkl` or `.npz` and click Mount.

---

## Closed

(empty)
