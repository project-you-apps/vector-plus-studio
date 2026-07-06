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
  // entry (browser-built carts 2026-06-15+). undefined for legacy carts;
  // ResultCard hides the source line when undefined. See
  // CC_cart-provenance-schema_2026-06-15 for v2 schema upgrade plan.
  source_path?: string | null
  // Step 2b: per-pattern RWX from the hippocampus row's perms_byte.
  // null when the cart has no hippocampus data; otherwise an object with
  // r/w/x bools and the raw byte value.
  perms?: PatternPerms | null
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
  // Cart-format permissions sidecar (Step 2a). Present when the cart has a
  // .permissions.json next to it. default: "r" | "rw" | "rwx".
  cart_permissions?: CartPermissions | null
  // True when the currently-mounted cart's file lives in the upload sandbox.
  // UI surfaces an "Eject" button to immediately delete the uploaded file
  // instead of waiting up to 1h for TTL eviction.
  mounted_is_sandboxed?: boolean
  // Absolute path of the mounted cart (used as eject target). Null when nothing mounted.
  mounted_path?: string | null
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

// Per-pattern metadata sidecar (Andy 2026-07-06 AM). Response of
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
