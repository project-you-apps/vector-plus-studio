/**
 * Resolve the best-available source path for a picked File.
 *
 * Browser file pickers behave differently depending on the input mode:
 *   - Single-file `<input type="file">` — `file.name` is the basename only;
 *     no folder context (browser security).
 *   - Directory picker (`<input type="file" webkitdirectory>`) — `file.webkitRelativePath`
 *     gives the folder-relative path (e.g. `docs/reports/foo.md`). Preserves
 *     the disambiguating folder hierarchy the user selected.
 *   - File System Access API — a separate flow (not used by the current
 *     Cart Builder); would need its own extractor.
 *
 * Preserving `webkitRelativePath` when the directory picker was used is what
 * lets a hosted-mode cart disambiguate `/archive/foo.md` vs `/drafts/foo.md`
 * — same failure mode Andy flagged 2026-07-20 for the local writer path.
 *
 * Absolute paths (like the Python CLI writer stores) are NOT available in
 * the browser regardless of picker mode — that's a security boundary
 * (same reason the browser hides `C:\Users\...` from JS). For local mode
 * with absolute paths, users go through the desktop Cart Builder (port
 * 7878) which has Python-level disk access.
 */
export function sourcePathForFile(file: File): string {
    const rel = file.webkitRelativePath
    if (rel && rel.length > 0) return rel
    return file.name
}
