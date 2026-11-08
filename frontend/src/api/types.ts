export interface CartridgeInfo {
  name: string
  filename: string
  size_mb: number
  has_brain: boolean
  has_signatures: boolean
  has_manifest: boolean
  pattern_count?: number
}

export interface PatternPerms {
  r: boolean
  w: boolean
  x: boolean
  raw: number
}

export interface SearchResult {
  rank: number
  idx: number
  score: number
  cosine_score: number | null
  physics_score: number | null
  hamming_score: number | null
  keyword_boost: number | null
  title: string
  preview: string
  full_text: string
  from_lattice: boolean
  prev_idx: number | null
  next_idx: number | null
  // Split-cart provenance hints — populated when the mounted cart has a
  // SQLite sidecar. Frontend uses presence of source_db to render the
  // "Load full passage from <db>" CTA in the modal. paper_id arrives later
  // via /api/patterns/{idx} when the user clicks the CTA.
  source_db?: string | null
  paper_id?: string | null
  // Provenance v1 sidecar — per-pattern source filename. Populated for
  // local-mounted carts where the cart .npz contains a source_paths.npy
  // entry. undefined for legacy carts; ResultCard hides the source line
  // when undefined.
  source_path?: string | null
  // v3 provenance — h-row ingestion timestamp (ISO 8601 UTC). PassageModal
  // renders as "Ingested" alongside source_path when the modal opens from
  // a search result.
  ingested_at?: string | null
  // Step 2b: per-pattern RWX from the hippocampus row's perms_byte.
  // null when the cart has no hippocampus data; otherwise an object with
  // r/w/x bools and the raw byte value.
  perms?: PatternPerms | null
  // ⭐⭐ The key the provenance layer addresses this passage by, computed SERVER-SIDE.
  // ⛔ Pass this VERBATIM to /api/provenance/{cart}/for-results. Never derive a key from
  //    `preview` or `full_text` on this side: a key over anything but the exact indexed
  //    body matches nothing, returns zero rows, and raises nothing. That failure is
  //    invisible — an empty panel looks identical to "this passage has no notes".
  // null when the provenance layer is unavailable, which is honest and distinct from "".
  content_key?: string | null
}

// ── Provenance (read-only) ─────────────────────────────────────────────────
// Served by /api/provenance/*. See api/provenance_routes.py — the design rule is that
// provenance travels WITH the result rather than living on a screen of its own, because
// a log on a dashboard is a log nobody reads.

export interface ProvenanceNote {
  id: string
  target_key: string
  // ⚠ `| string` is deliberate: a cart written by a newer membot can carry a kind this
  // build has never heard of, and refusing to render it would hide a real note.
  kind: 'why' | 'context' | 'caution' | 'decision' | 'prediction' | 'recommendation' | string
  body: string
  scope: 'private' | 'shared' | string
  /** ⛔ The seat UUID. This is IDENTITY — render `by_name` instead. */
  by: string
  /** Human label resolved server-side at presentation time. Absent when that seat has not
   *  been seen since the backend booted (the resolver is a cache, not a record). */
  by_name?: string | null
  surface: string
  at: number
  supersedes?: string
  superseded?: boolean
}

export interface AddNoteResponse {
  provenance_available: boolean
  cart: string
  note: ProvenanceNote
  target_key: string
  // ⭐ 'verified' when a real signed-in seat authored it; 'claimed' on a studio with no
  // auth configured. Rendered as-is — a system that shows these identically is asserting
  // something it does not know.
  identity_source: 'verified' | 'claimed' | string
  // ⚠ Non-empty when the body STORED differs from the body TYPED. Must be surfaced: a
  // redaction the author never learns about is content silently eaten.
  redactions: { kind?: string; verdict?: string }[]
  // Whether an identity system was actually governing this write.
  enforced: boolean
}

export interface ProvenanceEntry {
  seq: number
  op: string
  /** ⛔ The seat UUID. This is IDENTITY — render `writer_name` instead. */
  writer: string
  /** Human label resolved server-side. Absent when that seat has not been seen since the
   *  backend booted. */
  writer_name?: string | null
  surface: string
  event_ts: number
  ingest_ts: number
  target: Record<string, unknown>
  payload: Record<string, unknown>
  hash: string
}

export interface ProvenanceIntegrity {
  ok: boolean
  verdict_line: string | null
  entries: number
  first_break: Record<string, unknown> | null
  hwm_present: boolean | null
}

export interface ForResultsResponse {
  provenance_available: boolean
  // Present only when provenance_available is false. ⚠ Render this rather than an empty
  // state: "we cannot read the record" and "there is no record" are different facts.
  reason?: string | null
  cart?: string
  integrity?: ProvenanceIntegrity
  // true when the hash chain is broken and journal entries were deliberately withheld.
  // Notes are still returned — a note is a human statement, not a claim about an
  // unaltered history.
  entries_withheld?: boolean
  keys?: string[]
  by_key?: Record<string, { provenance: ProvenanceEntry[]; notes: ProvenanceNote[] }>
}

export interface PatternResponse {
  idx: number
  title: string
  preview: string
  full_text: string
  prev_idx: number | null
  next_idx: number | null
  // Split-cart provenance — populated when /api/patterns/{idx} fetched the
  // full text from a SQLite sidecar.
  source_db?: string | null
  paper_id?: string | null
  // Step 2b: per-pattern RWX bits from hippocampus row's perms_byte.
  perms?: PatternPerms | null
  // v3 provenance — per-pattern source filename resolved from the mounted
  // cart's source_strings + h-row source_idx (or v1 sidecar fallback).
  // PassageModal renders as a "Source: <filename>" line under the title.
  source_path?: string | null
  // v3 provenance — h-row ingestion timestamp (uint32 Unix epoch) formatted
  // as ISO 8601 UTC. PassageModal renders as "ingested <date>" alongside
  // source_path. undefined/null for legacy carts.
  ingested_at?: string | null
  // ⭐ Server-computed provenance key for THIS passage.
  // ⛔ The modal cannot reuse the search result's key: prev/next refetches by idx, so
  // after one click the hit it was opened from no longer describes what is on screen.
  // The key has to arrive with the text it belongs to.
  content_key?: string | null
}

export interface SearchResponse {
  query: string
  mode: string
  elapsed_ms: number
  result_count: number
  results: SearchResult[]
}

export interface StatusResponse {
  engine_ready: boolean
  gpu_available: boolean
  mounted_cartridge: string | null
  pattern_count: number
  physics_trained: boolean
  training_active: boolean
  training_progress: number
  training_total: number
  multimodal: boolean
  signatures_loaded: boolean
  deleted_count: number
  dirty: boolean
  read_only: boolean
  // True when the server is in global read-only mode (VPS_READ_ONLY env var).
  // Frontend uses this to hide the unlock button — attempting to unlock returns 403.
  read_only_mode?: boolean
  // ⭐⭐ "This backend is reachable by people who do not own the machine."
  // Sent by the server since 2026-08-06 (models.py / main.PUBLIC_HOST) and read by
  // NOTHING on this side until 2026-11-04 — the attestation existed, unsurfaced.
  // ⛔ THE ONLY TRUSTWORTHY SOURCE FOR THIS FACT. The browser sees a relative `/api`
  //    and cannot see through the Vite dev proxy, so it cannot tell a page on
  //    localhost driving a local backend from one driving a public one. Distinct from
  //    read_only_mode: that governs WRITES, this governs whether the host is public.
  // undefined/null means "not answered yet" — see lib/posture.ts, which treats that as
  // UNKNOWN rather than defaulting to the reassuring answer.
  public_host?: boolean | null
  // ⚠ Advisory, NOT a permission — the server decides on every write. False only when
  // the server-wide VPS_READ_ONLY is set, which means this process does not write to disk
  // at all. ⭐ Deliberately NOT tied to `read_only`: the mount lock guards cart CONTENT,
  // and a note is stored beside the cart rather than in it. Requiring an unlock to
  // comment would force the user into a more dangerous state (every passage editable) to
  // do a safer thing.
  notes_writable?: boolean
  // Cart-format permissions sidecar (Step 2a). Present when the cart has a
  // .permissions.json next to it. default: "r" | "rw" | "rwx".
  cart_permissions?: CartPermissions | null
  // True when the currently-mounted cart's file lives in the upload sandbox.
  // UI surfaces an "Eject" button to immediately delete the uploaded file
  // instead of waiting up to 1h for TTL eviction.
  mounted_is_sandboxed?: boolean
  cart_occupants?: string[]
  // No mounted_path. The server stopped publishing the absolute cart path on 2026-08-10 —
  // /api/status is polled before sign-in so it cannot be behind auth, and eject no longer
  // needs the client to name the file. See api/uploads.py.
}

export interface CartPermissions {
  default: string  // "r" | "rw" | "rwx"
  owner?: string
  description?: string
  version?: string
}

// Pattern-0 TOC (2026-07-01, Andy). Response of GET /api/cart/pattern-0.
// One entry per source file (or free-floating passage) in the mounted cart.
export interface Pattern0TocItem {
  name: string
  description?: string | null
  pattern_count: number
}

export interface Pattern0Response {
  mounted: boolean
  name?: string | null
  description?: string | null
  creator?: string | null
  created_at?: string | null   // ISO 8601
  owner?: string | null
  pattern_count?: number
  // v2 agent_briefing block. When present, the TOC panel surfaces a
  // BRIEFING button that opens a modal with the full text. See
  // docs/RFC/pattern-0-v2-spec.md.
  agent_briefing?: string | null
  toc_items: Pattern0TocItem[]
  // True when Pattern-0 was minimal/absent and toc_items came from a
  // hippocampus-source-hash derivation. UI shows the "No metadata available
  // — showing derived stats" banner.
  is_derived: boolean
  // Day 2 — Image Builder integration counts. Zero for text-only carts.
  // Pattern0TocPanel surfaces "N graphics + M tables" alongside the file
  // list when either exceeds zero.
  graphic_count?: number
  table_count?: number
}

// Per-pattern metadata sidecar. Response of
// GET /api/cart/per-pattern-meta. Records parallel `passages` — one entry
// per pattern with content_type + type-specific extras (image_b64 for
// graphics, html for tables). Enables sandbox-mounted carts to reach parity
// with LocalCart-mounted carts for image/table rendering. When mounted is
// false or records is empty, no per_pattern_meta.npy sidecar is present
// (legacy cart) — the UI falls through to the text-only path.
export interface PerPatternMetaRecord {
  v?: number
  content_type?: 'document' | 'graphic' | 'table' | string
  source?: string
  page?: number | null
  chunk?: number
  chunks?: number
  tags?: string[]
  created_at?: number
  tombstone?: boolean
  // Graphic extras
  caption?: string
  image_b64?: string
  bbox?: number[]
  // Table extras
  html?: string
}

export interface PerPatternMetaResponse {
  mounted: boolean
  records: PerPatternMetaRecord[]
}

export interface DeletedPattern {
  idx: number
  title: string
  preview: string
}

export type SearchMode = 'hamming' | 'smart' | 'pure_brain' | 'fast' | 'associate'

// --- Membox visualizer ---

export interface MemboxLockState {
  cart_id: string
  holder: string | null
  held_for_seconds: number | null
  lease_seconds: number
  acquire_count: number
  wait_count: number
  is_locked: boolean
}

export interface MemboxCartInfo {
  cart_id: string
  role: string | null
  n_patterns: number
  lock: MemboxLockState
  recent_writes: number
}

export interface MemboxWriteEntry {
  agent_id: string
  written_at: string
  local_addr: number
  origin: string
  text_preview: string
}

export interface MemboxStatus {
  cart_id: string
  n_patterns: number
  lock: MemboxLockState
  writes_by_agent: Record<string, number>
  recent_writes: MemboxWriteEntry[]
  membox_enabled: boolean
}

export interface MemboxCartListResponse {
  carts: MemboxCartInfo[]
}

export interface MemboxImprintRequest {
  cart_id: string
  text: string
  agent_id: string
  tags?: string
  reasoning?: string
  origin?: string
  timeout_ms?: number
}

export interface MemboxMountRequest {
  cart_path: string
  cart_id?: string | null
  role?: string | null
  lease_seconds?: number
  verify_integrity?: boolean
}
