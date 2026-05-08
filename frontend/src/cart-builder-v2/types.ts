// Shared types for the in-browser Cart Builder V2 pipeline.
// Output shape matches cart-builder/cart-builder/parsers.py so carts built
// in the browser stay byte-near-identical to server-built ones (verified at
// 6th-decimal embedder drift, smoke-tested 2026-05-06).

export interface Section {
  text: string
  page: number | null
  source: string
  // Populated by the chunker when a section is split into overlapping windows.
  part?: number
}

export interface ParseMeta {
  filename: string
  size: number
  parserUsed: string
  parsedAt: number
}

export interface ParseResult {
  sections: Section[]
  metadata: ParseMeta
}

export interface Parser {
  name: string
  accept(file: File): boolean
  parse(file: File): Promise<Section[]>
}

export class ParseError extends Error {
  filename: string
  constructor(message: string, filename: string, cause?: unknown) {
    super(message, cause !== undefined ? { cause } : undefined)
    this.name = 'ParseError'
    this.filename = filename
  }
}

// ─── Embedder (Block 2) ───────────────────────────────────────────────

// transformers.js progress event shape — fires during model download.
export interface ModelDownloadProgress {
  status: string // 'initiate' | 'download' | 'progress' | 'done' | 'ready' | 'error'
  file?: string
  progress?: number // 0-100
  loaded?: number
  total?: number
}

export type ProgressCallback = (progress: ModelDownloadProgress) => void

export type EmbedderBackend = 'webgpu' | 'wasm' | 'unknown'
