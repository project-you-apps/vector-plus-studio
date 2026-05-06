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
}

export interface CartPermissions {
  default: string  // "r" | "rw" | "rwx"
  owner?: string
  description?: string
  version?: string
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
