// Shared types for the in-browser Cart Builder V2 pipeline.
// Output shape matches cart-builder/cart-builder/parsers.py so carts built
// in the browser stay byte-near-identical to server-built ones (verified at
// 6th-decimal embedder drift, smoke-tested 2026-05-06).

// content_type on a Section. "document" is the default text-chunk kind used by
// every existing parser; "graphic" and "table" are added Day 2 for Image
// Builder delegation — each Docling-returned graphic + table becomes its own
// pattern rather than being folded back into the text stream. Kept as a plain
// string union rather than a full enum so existing sections (which never set
// this field) implicitly land as "document" without a migration.
export type ContentType = 'document' | 'graphic' | 'table'

// Docling graphic — one per graphic returned by Image Builder's /ocr response.
// Values pass straight through from the Pydantic GraphicItem in
// image-builder/models.py so the Image Builder API stays the frozen boundary.
export interface Graphic {
  page: number            // 1-indexed page in the source doc (1 for standalone images)
  caption: string         // detected caption text; may be ""
  bbox: number[]          // [x0, y0, x1, y1] in page pixels; may be []
  image_b64: string       // base64-encoded PNG; may be ""
}

// Docling table — one per table returned by Image Builder's /ocr response.
// Same passthrough shape as Graphic; TableItem in image-builder/models.py.
export interface Table {
  page: number            // 1-indexed page
  html: string            // HTML-encoded table markup; may be ""
  bbox: number[]          // [x0, y0, x1, y1] in page pixels; may be []
}

// Result of running a file through Image Builder's /ocr endpoint. `markdown`
// gets threaded through the standard chunker like any text extraction;
// `graphics` + `tables` become their own patterns (bypassing the chunker).
export interface ImageBuilderOcrResult {
  markdown: string
  graphics: Graphic[]
  tables: Table[]
  sourceType: 'pdf' | 'image'
  pageCount: number
  elapsedSec: number
}

export interface Section {
  text: string
  page: number | null
  source: string
  // Populated by the chunker when a section is split into overlapping windows.
  part?: number
  // Content kind — omitted (== "document") for existing text sections; set
  // explicitly for graphic + table patterns produced by the Image Builder
  // delegation path (Day 2). The writer reads this to populate
  // per_pattern_meta[i].content_type. When present, the chunker treats the
  // section as opaque (does not split — each graphic / table is one pattern).
  contentType?: ContentType
  // Graphic-only: base64 PNG bytes preserved for future thumbnail + zoom UI.
  imageB64?: string
  // Graphic + table: page-pixel bbox, preserved for future zoom + source-return.
  bbox?: number[]
  // Graphic-only: raw caption text as returned by Docling (may be "").
  caption?: string
  // Table-only: HTML-encoded table markup, preserved for future rich rendering.
  html?: string
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
