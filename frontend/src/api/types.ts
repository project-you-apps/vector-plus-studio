export interface CartridgeInfo {
  name: string
  filename: string
  size_mb: number
  has_brain: boolean
  has_signatures: boolean
  has_manifest: boolean
  pattern_count?: number
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
}

export interface DeletedPattern {
  idx: number
  title: string
  preview: string
}

export type SearchMode = 'hamming' | 'smart' | 'pure_brain' | 'fast'
