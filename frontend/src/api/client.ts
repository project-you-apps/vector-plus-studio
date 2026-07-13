import type {
  CartridgeInfo, SearchResponse, StatusResponse, DeletedPattern,
  SearchMode, PatternResponse, Pattern0Response, PerPatternMetaResponse,
  MemboxCartInfo, MemboxStatus, MemboxImprintRequest, MemboxMountRequest,
} from './types'
import { useAuthStore } from '../store/authStore'

// Set VITE_API_BASE at build time for hosted deploys (e.g. '/vps/api'). Local
// dev uses '/api' which the Vite proxy routes to localhost:8000.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'

// Reads the current Supabase access token from the auth store (kept in sync via
// onAuthStateChange in authStore.init). Returns an empty object when signed out
// so anonymous endpoints (sandbox uploads, public mounts) still work.
function authHeaders(): Record<string, string> {
  const session = useAuthStore.getState().session
  return session ? { Authorization: `Bearer ${session.access_token}` } : {}
}

async function fetchJSON<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    ...opts,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

export async function getStatus(): Promise<StatusResponse> {
  return fetchJSON('/status')
}

// Pattern-0 TOC for the mounted cart. Returns {mounted:false} when nothing
// is mounted server-side, so the caller can hide the panel without a special
// error branch. See Pattern0TocPanel.tsx (2026-07-01).
export async function getCartPattern0(): Promise<Pattern0Response> {
  return fetchJSON('/cart/pattern-0')
}

// Per-pattern metadata sidecar for the currently-mounted cart. Payload can
// be several MB for image-heavy carts (each graphic carries base64 PNG
// bytes). Andy 2026-07-06 AM: called on mount so sandbox-mounted carts
// can render graphics/tables just like LocalCart mounts do.
export async function getCartPerPatternMeta(): Promise<PerPatternMetaResponse> {
  return fetchJSON('/cart/per-pattern-meta')
}

export async function getCartridges(): Promise<CartridgeInfo[]> {
  const data = await fetchJSON<{ cartridges: CartridgeInfo[] }>('/cartridges')
  return data.cartridges
}

export async function browseForCartridge(): Promise<string> {
  const data = await fetchJSON<{ path: string }>('/browse')
  return data.path
}

export interface UploadResponse {
  success: boolean
  message: string
  cart_path: string
  size_mb: number
  ttl_sec: number
}

export async function uploadCartridge(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/cartridges/upload`, {
    method: 'POST',
    body: form,
    headers: authHeaders(),
  })
  if (!res.ok) {
    let detail = `${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* keep status */ }
    throw new Error(detail)
  }
  return res.json()
}

export async function ejectCartridge(cartPath: string): Promise<{ success: boolean; ejected: string }> {
  const res = await fetch(
    `${BASE}/cartridges/eject?cart_path=${encodeURIComponent(cartPath)}`,
    { method: 'DELETE', headers: authHeaders() }
  )
  if (!res.ok) {
    let detail = `${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* keep status */ }
    throw new Error(detail)
  }
  return res.json()
}

export async function mountCartridge(filename: string) {
  return fetchJSON<{ success: boolean; message: string; name: string; pattern_count: number }>(
    '/cartridges/mount',
    { method: 'POST', body: JSON.stringify({ filename }) }
  )
}

export async function unmountCartridge() {
  return fetchJSON<{ success: boolean; message: string }>(
    '/cartridges/unmount',
    { method: 'POST' }
  )
}

export async function saveCartridge() {
  return fetchJSON<{ success: boolean; message: string }>(
    '/cartridges/save',
    { method: 'POST' }
  )
}

export async function lockCartridge() {
  return fetchJSON<{ success: boolean; message: string }>(
    '/cartridges/lock',
    { method: 'POST' }
  )
}

export async function unlockCartridge() {
  return fetchJSON<{ success: boolean; message: string }>(
    '/cartridges/unlock',
    { method: 'POST' }
  )
}

export async function search(
  query: string,
  mode: SearchMode,
  alpha: number,
  top_k: number
): Promise<SearchResponse> {
  return fetchJSON('/search', {
    method: 'POST',
    body: JSON.stringify({ query, mode, alpha, top_k }),
  })
}

export async function getPattern(idx: number): Promise<PatternResponse> {
  return fetchJSON(`/patterns/${idx}`)
}

export async function deletePattern(idx: number) {
  return fetchJSON<{ success: boolean; message: string }>(
    `/patterns/${idx}`,
    { method: 'DELETE' }
  )
}

export async function restorePattern(idx: number) {
  return fetchJSON<{ success: boolean; message: string }>(
    `/patterns/${idx}/restore`,
    { method: 'POST' }
  )
}

export async function getDeletedPatterns(): Promise<DeletedPattern[]> {
  const data = await fetchJSON<{ deleted: DeletedPattern[] }>('/patterns/deleted')
  return data.deleted
}

export interface PatternListItem {
  idx: number
  title: string
  preview: string
  word_count: number
}

export interface PatternListResponse {
  passages: PatternListItem[]
  total: number
  offset: number
  limit: number
  filter: string | null
}

export async function listPatterns(
  offset: number = 0,
  limit: number = 25,
  q?: string,
  source?: string,
): Promise<PatternListResponse> {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) })
  if (q && q.trim()) params.append('q', q.trim())
  if (source && source.trim()) params.append('source', source.trim())
  return fetchJSON<PatternListResponse>(`/patterns?${params.toString()}`)
}

export async function addPassage(text: string) {
  return fetchJSON<{ success: boolean; message: string }>(
    '/patterns',
    { method: 'POST', body: JSON.stringify({ text }) }
  )
}

// --- Membox visualizer ---

export async function fetchMemboxCarts(): Promise<MemboxCartInfo[]> {
  const data = await fetchJSON<{ carts: MemboxCartInfo[] }>('/membox/carts')
  return data.carts
}

export async function fetchMemboxStatus(cartId: string): Promise<MemboxStatus> {
  return fetchJSON(`/membox/status/${encodeURIComponent(cartId)}`)
}

export async function memboxImprint(req: MemboxImprintRequest) {
  return fetchJSON<{ success: boolean; message: string }>(
    '/membox/imprint',
    { method: 'POST', body: JSON.stringify(req) }
  )
}

export async function memboxMount(req: MemboxMountRequest) {
  return fetchJSON<{ success: boolean; message: string }>(
    '/membox/mount',
    { method: 'POST', body: JSON.stringify(req) }
  )
}

export async function memboxUnmount(cartId: string) {
  return fetchJSON<{ success: boolean; message: string }>(
    '/membox/unmount',
    { method: 'POST', body: JSON.stringify({ cart_id: cartId }) }
  )
}

// --- Reports engine (Wave-2 dispatch, 2026-07-13) ---
//
// generateReport POSTs form values plus the server cart identifier at
// /api/reports/generate. The route returns a fully-rendered markdown
// string plus metadata / warnings; the pane renders markdown via
// react-markdown + remark-gfm. Wave-2 slugs (timeline / trend /
// financial_rollup / tldr) come back as 501 — the caller surfaces that
// as a friendly "future release" message, not an error.

export interface GenerateReportRequest {
  report_slug: string
  cart_ref: string
  cart_ref_b?: string | null
  inputs: Record<string, unknown>
}

export interface GenerateReportResponse {
  markdown: string
  metadata: Record<string, unknown>
  warnings: string[]
  csv_data: string | null
  html_extra: string | null
  report_slug: string
  generated_at: string
}

export interface GenerateReportErrorDetail {
  error: string
  message: string
  report_slug?: string
  cart_ref?: string
}

export class GenerateReportError extends Error {
  status: number
  detail: GenerateReportErrorDetail
  constructor(status: number, detail: GenerateReportErrorDetail) {
    super(detail.message || `Report generation failed (${status})`)
    this.status = status
    this.detail = detail
  }
}

// GET /api/reports/carts — enumerates server carts with per-cart report
// compatibility. Used by ReportsScreen's cart selector to grey out legacy
// .pkl carts (report engine only reads .cart.npz) with a helpful tooltip.
// Ordering is server-side: compatible first alphabetical, then incompatible
// alphabetical. Failures fall back to an empty list so the selector at
// least still renders — the input pane's cart_not_found branch is the
// final safety net.

export interface ReportCartEntry {
  id: string
  display_name: string
  report_compatible: boolean
  format: 'npz' | 'pkl' | string
}

export async function fetchReportCarts(): Promise<ReportCartEntry[]> {
  const data = await fetchJSON<{ carts: ReportCartEntry[] }>('/reports/carts')
  return data.carts
}

export async function generateReport(
  req: GenerateReportRequest,
): Promise<GenerateReportResponse> {
  const res = await fetch(`${BASE}/reports/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    // FastAPI wraps our HTTPException detail dict in {detail: {...}}.
    // Pydantic 422s wrap it in {detail: [{...}]}. Handle both shapes so
    // the caller always gets a normalized error object.
    let raw: unknown
    try { raw = await res.json() } catch { raw = null }
    let detail: GenerateReportErrorDetail
    if (raw && typeof raw === 'object' && 'detail' in raw) {
      const d = (raw as { detail: unknown }).detail
      if (Array.isArray(d)) {
        // Pydantic validation error list — turn the first one into
        // something we can render.
        const first = d[0] as { msg?: string; loc?: string[] } | undefined
        detail = {
          error: 'validation_error',
          message: first?.msg
            ? `${first.msg}${first.loc ? ` (${first.loc.join('.')})` : ''}`
            : 'Request validation failed',
        }
      } else if (typeof d === 'object' && d !== null) {
        detail = d as GenerateReportErrorDetail
      } else {
        detail = { error: 'unknown_error', message: String(d ?? `HTTP ${res.status}`) }
      }
    } else {
      detail = { error: 'unknown_error', message: `HTTP ${res.status}` }
    }
    throw new GenerateReportError(res.status, detail)
  }
  return res.json()
}

export async function forgeCartridge(name: string, files: File[]) {
  const form = new FormData()
  form.append('name', name)
  for (const f of files) {
    form.append('files', f)
  }
  const res = await fetch(`${BASE}/forge`, {
    method: 'POST',
    body: form,
    headers: authHeaders(),
  })
  if (!res.ok) throw new Error(`Forge failed: ${res.status}`)
  return res.json() as Promise<{ success: boolean; message: string }>
}
