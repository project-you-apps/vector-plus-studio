// Image Builder API client (Day 2). Thin wrapper around the local exe's
// `POST /ocr` endpoint on `http://127.0.0.1:7879`. Image Builder shares the
// Cart Builder pairing token — one persisted string in localStorage under
// DESKTOP_HELPER_TOKEN_KEY unlocks both Builders. From the browser we read
// that same key; when a paired Desktop Cart Builder exe delegates OCR, it
// reads `~/.vector-plus/token` from disk (see image-builder/auth.py).
//
// This file is browser-only. Backend delegation (paired Desktop Cart Builder
// exe → Image Builder exe) happens via httpx in api/cartbuilder/builder.py.

import type {
  Graphic,
  ImageBuilderOcrResult,
  Table,
} from '../cart-builder-v2/types'
import { DESKTOP_HELPER_TOKEN_KEY } from '../store/appStore'

// Fixed for MVP; Image Builder falls back to the 7879-7888 port range if 7879
// is busy (see PORT_RANGE in image-builder/main.py). Detection reads the port
// from GET /capabilities before any /ocr call, so this constant is only the
// starting probe; runtime origin is derived from the store's cached caps.
export const IMAGE_BUILDER_DEFAULT_PORT = 7879
export const IMAGE_BUILDER_DEFAULT_ORIGIN = `http://127.0.0.1:${IMAGE_BUILDER_DEFAULT_PORT}`

// Image Builder /capabilities response — matches capabilities() in main.py.
// Kept minimal; only fields Cart Builder actually consumes are typed.
export interface ImageBuilderCapabilities {
  exe: string
  version?: string
  capabilities: string[]
  port: number
  supported_formats: string[]
}

// Detection uses the default port (7879) unconditionally — Image Builder
// binds there first and only falls back on port conflict. If we ever need to
// re-probe the whole PORT_RANGE we can add it, but MVP: one port, one probe.
export async function probeImageBuilder(signal?: AbortSignal): Promise<
  { ok: true; capabilities: ImageBuilderCapabilities }
  | { ok: false; reason: string }
> {
  try {
    const healthResp = await fetch(`${IMAGE_BUILDER_DEFAULT_ORIGIN}/health`, {
      method: 'GET',
      signal,
    })
    if (!healthResp.ok) {
      return { ok: false, reason: `health ${healthResp.status}` }
    }
    const capsResp = await fetch(`${IMAGE_BUILDER_DEFAULT_ORIGIN}/capabilities`, {
      method: 'GET',
      signal,
    })
    if (!capsResp.ok) {
      return { ok: false, reason: `capabilities ${capsResp.status}` }
    }
    const capabilities = (await capsResp.json()) as ImageBuilderCapabilities
    return { ok: true, capabilities }
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : 'probe failed' }
  }
}

/**
 * Attempt to /pair the paste token against Image Builder. Same handshake
 * shape as Desktop Cart Builder — the exe's /pair returns 200 for valid
 * tokens, 401 for anything else. On success, the caller persists the token
 * in localStorage (shared key with Desktop Helper) so subsequent OCR calls
 * pick it up automatically.
 */
export async function pairImageBuilder(
  token: string,
  origin: string = IMAGE_BUILDER_DEFAULT_ORIGIN,
): Promise<{ success: boolean; message: string }> {
  try {
    const resp = await fetch(`${origin}/pair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
    if (resp.ok) return { success: true, message: 'paired' }
    if (resp.status === 401) {
      return { success: false, message: 'Pairing code rejected' }
    }
    return { success: false, message: `pair failed: ${resp.status}` }
  } catch (e) {
    return { success: false, message: e instanceof Error ? e.message : 'pair failed' }
  }
}

// Read the shared bearer token from localStorage. The same key backs both
// Builders — pairing either one persists a token here.
function readSharedToken(): string | null {
  try {
    return localStorage.getItem(DESKTOP_HELPER_TOKEN_KEY)
  } catch {
    return null
  }
}

// Camel-case normalization of the OCR response — the Pydantic model returns
// snake_case (`source_type`, `page_count`, `elapsed_sec`), the frontend
// prefers camelCase throughout. Everything downstream reads via the
// ImageBuilderOcrResult shape defined in cart-builder-v2/types.ts.
export interface OcrOptions {
  includeGraphics?: boolean  // default true (Image Builder's own default)
  includeTables?: boolean    // default true
  maxPages?: number          // default 100 (Image Builder's own default)
}

/**
 * POST a file to Image Builder /ocr. Multipart form: `file` + optional
 * options fields aliased under the `options.*` prefix (see main.py OCR
 * handler). Returns normalized markdown + graphics + tables.
 *
 * Errors surface as thrown Error objects with a status-tagged message so
 * callers can distinguish "Image Builder not running" (connection refused)
 * from "OCR failed for this file" (400/500 with detail). The pipeline uses
 * the shape to decide whether to fall back or emit a placeholder pattern
 * .
 */
export async function ocrFile(
  file: File,
  options: OcrOptions = {},
  origin: string = IMAGE_BUILDER_DEFAULT_ORIGIN,
): Promise<ImageBuilderOcrResult> {
  const token = readSharedToken()
  if (!token) {
    throw new Error('Image Builder not paired (no token)')
  }
  const form = new FormData()
  form.append('file', file, file.name)
  if (options.includeGraphics !== undefined) {
    form.append('options.include_graphics', options.includeGraphics ? '1' : '0')
  }
  if (options.includeTables !== undefined) {
    form.append('options.include_tables', options.includeTables ? '1' : '0')
  }
  if (options.maxPages !== undefined) {
    form.append('options.max_pages', String(options.maxPages))
  }
  const resp = await fetch(`${origin}/ocr`, {
    method: 'POST',
    body: form,
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!resp.ok) {
    // Docling errors surface as {error, detail} JSON per image-builder/main.py.
    // Failed to parse JSON (network 502 in front of a proxied deploy, etc.)
    // — fall back to raw status message.
    let detail = `HTTP ${resp.status}`
    try {
      const body = await resp.json()
      detail = (body?.detail || body?.error || detail) as string
    } catch {
      // keep detail = status
    }
    throw new Error(`Image Builder /ocr ${resp.status}: ${detail}`)
  }
  const raw = (await resp.json()) as {
    markdown: string
    graphics: Graphic[]
    tables: Table[]
    source_type: 'pdf' | 'image'
    page_count: number
    elapsed_sec: number
  }
  return {
    markdown: raw.markdown ?? '',
    graphics: Array.isArray(raw.graphics) ? raw.graphics : [],
    tables: Array.isArray(raw.tables) ? raw.tables : [],
    sourceType: raw.source_type,
    pageCount: raw.page_count,
    elapsedSec: raw.elapsed_sec,
  }
}
