import type {
  CartridgeInfo, SearchResponse, StatusResponse, DeletedPattern,
  SearchMode, PatternResponse, Pattern0Response, PerPatternMetaResponse,
  MemboxCartInfo, MemboxStatus, MemboxImprintRequest, MemboxMountRequest,
} from './types'
import { useAuthStore } from '../store/authStore'
import {
  REPORT_BUILDER_DEFAULT_ORIGIN,
  readReportBuilderToken,
} from './reportBuilder'

// Set VITE_API_BASE at build time for hosted deploys (e.g. '/vps/api'). Local
// dev uses '/api' which the Vite proxy routes to localhost:8000.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'

// ---------------------------------------------------------------------------
// Report Builder routing
// ---------------------------------------------------------------------------
//
// When the user has a paired Report Builder AND the active cart_ref
// carries the browser-local prefix (`local:`), Reports + Agents calls
// route to the local exe on 127.0.0.1:7880 instead of the droplet.
// Data (the cart's contents) stays on the user's machine; only the
// anonymized prompt + retrieved context leave for LLM synthesis.
//
// Detection lives in the app store (reportBuilderState). We read the
// store lazily via a factory so the client doesn't take a hard runtime
// dep on appStore (avoids circular-import issues; tests can inject).

let _reportBuilderStateReader: () => 'unknown' | 'detecting' | 'not-found' |
  'detected-unpaired' | 'detected-paired' = () => 'unknown'

// Called once from appStore module scope at import time — same trick as
// authStore.getState(). Kept as a setter so tests can override.
export function _registerReportBuilderStateReader(
  fn: () => 'unknown' | 'detecting' | 'not-found' |
    'detected-unpaired' | 'detected-paired',
) {
  _reportBuilderStateReader = fn
}

interface RouterTarget {
  base: string
  headers: Record<string, string>
}

// Decide whether to route a Reports / Agents call to Report Builder.
// Returns null when the droplet path should be used (default). Returns
// {base, headers} when the local exe should handle it.
function reportBuilderTarget(cartRef: string | null | undefined): RouterTarget | null {
  if (!cartRef || !cartRef.startsWith('local:')) return null
  if (_reportBuilderStateReader() !== 'detected-paired') return null
  const token = readReportBuilderToken()
  if (!token) return null
  return {
    base: REPORT_BUILDER_DEFAULT_ORIGIN,
    headers: { Authorization: `Bearer ${token}` },
  }
}

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
  // 'canonical' — lives under cartridges/ or sample_data/ (curated demo cart).
  // 'sandbox'   — lives under cartridges/_session_uploads/ (short-TTL user
  //               upload via POST /api/cartridges/upload). Selector renders a
  //               distinct badge so users know the cart is temporary.
  location?: 'canonical' | 'sandbox' | string
}

// Fetches the droplet's cart list. When Report Builder is paired, the
// caller (ReportsScreen / AgentsScreen) also calls fetchLocalReportCarts
// and merges the two lists — local: entries carry a `location: 'local'`
// badge so the selector renders them distinctly.
export async function fetchReportCarts(): Promise<ReportCartEntry[]> {
  const data = await fetchJSON<{ carts: ReportCartEntry[] }>('/reports/carts')
  return data.carts
}

// Enumerate the paired Report Builder's local cart folder. Returns null
// when Report Builder isn't paired (nothing to enumerate). All entries
// come back report_compatible + format='npz' + location='local' — the
// exe only opens NPZ.
export async function fetchLocalReportCarts(): Promise<ReportCartEntry[] | null> {
  if (_reportBuilderStateReader() !== 'detected-paired') return null
  const token = readReportBuilderToken()
  if (!token) return null
  try {
    const res = await fetch(`${REPORT_BUILDER_DEFAULT_ORIGIN}/reports/carts`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!res.ok) return []
    const body = (await res.json()) as { carts: ReportCartEntry[] }
    return body.carts.map((c) => ({ ...c, location: 'local' as const }))
  } catch {
    return []
  }
}

export async function generateReport(
  req: GenerateReportRequest,
): Promise<GenerateReportResponse> {
  // Route to Report Builder when the cart_ref is browser-local AND the
  // exe is paired. Falls through to the droplet path otherwise so
  // canonical + sandbox carts continue to run server-side.
  const target = reportBuilderTarget(req.cart_ref)
  const url = target ? `${target.base}/reports/generate` : `${BASE}/reports/generate`
  const headers = target
    ? { 'Content-Type': 'application/json', ...target.headers }
    : { 'Content-Type': 'application/json', ...authHeaders() }
  const res = await fetch(url, {
    method: 'POST',
    headers,
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

// --- Agents engine (2026-07-13 MVP) ---
//
// Mirrors the Reports client shape — POST /api/agents/run for dispatch,
// GET /api/agents/list for enumeration, POST /api/agents/save_to_cart
// for the v1 stub save action. Wire shape documented in
// api/agents_routes.py; keep the two in sync (they share the
// AgentRun / SaveToCart pydantic models).

export interface RunAgentRequest {
  agent_slug: string
  cart_ref: string
  inputs: Record<string, unknown>
  // Opaque per-browser identifier; used by the server-side neuron-cap
  // counter to key per-session buckets. Generated once in appStore and
  // persisted in localStorage so a refresh doesn't rekey the bucket.
  session_id?: string
}

export interface RunAgentResponse {
  run_id: string
  markdown: string
  metadata: Record<string, unknown>
  cited_patterns: number[]
  warnings: string[]
  llm_usage: Record<string, unknown>
  agent_slug: string
  generated_at: string
  elapsed_ms: number
}

export interface RunAgentErrorDetail {
  error: string
  message: string
  agent_slug?: string
  cart_ref?: string
  reset_at?: string
  cap_hit?: string
}

export class RunAgentError extends Error {
  status: number
  detail: RunAgentErrorDetail
  constructor(status: number, detail: RunAgentErrorDetail) {
    super(detail.message || `Agent run failed (${status})`)
    this.status = status
    this.detail = detail
  }
}

export interface AgentListEntry {
  name: string
  display_name: string
  description: string
  input_schema: Array<Record<string, unknown>>
  llm_dependency: boolean
}

export interface AgentListResponse {
  agents: AgentListEntry[]
  caps: {
    max_requests_per_day: number
    max_neurons_per_day: number
    warn_threshold: number
  }
}

export async function fetchAgentList(): Promise<AgentListResponse> {
  return fetchJSON<AgentListResponse>('/agents/list')
}

export async function runAgent(
  req: RunAgentRequest,
): Promise<RunAgentResponse> {
  // Same routing rule as generateReport — local: cart on a paired
  // Report Builder runs against the local exe. Neuron-cap headers are
  // meaningless in that path (the exe returns caps=0 in /agents/list)
  // so the response shape stays the same; callers just don't see the
  // X-Agent-* warning headers.
  const target = reportBuilderTarget(req.cart_ref)
  const url = target ? `${target.base}/agents/run` : `${BASE}/agents/run`
  const headers = target
    ? { 'Content-Type': 'application/json', ...target.headers }
    : { 'Content-Type': 'application/json', ...authHeaders() }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    // Same detail-unwrapping dance as generateReport — FastAPI's
    // HTTPException detail is a dict; Pydantic 422 is a list.
    let raw: unknown
    try { raw = await res.json() } catch { raw = null }
    let detail: RunAgentErrorDetail
    if (raw && typeof raw === 'object' && 'detail' in raw) {
      const d = (raw as { detail: unknown }).detail
      if (Array.isArray(d)) {
        const first = d[0] as { msg?: string; loc?: string[] } | undefined
        detail = {
          error: 'validation_error',
          message: first?.msg
            ? `${first.msg}${first.loc ? ` (${first.loc.join('.')})` : ''}`
            : 'Request validation failed',
        }
      } else if (typeof d === 'object' && d !== null) {
        detail = d as RunAgentErrorDetail
      } else {
        detail = { error: 'unknown_error', message: String(d ?? `HTTP ${res.status}`) }
      }
    } else {
      detail = { error: 'unknown_error', message: `HTTP ${res.status}` }
    }
    throw new RunAgentError(res.status, detail)
  }
  return res.json()
}

export interface SaveAgentToCartResponse {
  success: boolean
  saved_at: string
  message: string
  run_id: string
}

export async function saveAgentToCart(
  runId: string,
  opts?: { cartRef?: string; sessionId?: string },
): Promise<SaveAgentToCartResponse> {
  // Save-to-cart is anchored to the run's source cart. When that cart
  // was `local:`, the run object lives in the Report Builder's cache,
  // so save must hit the same exe or the run_id 404s.
  const target = reportBuilderTarget(opts?.cartRef)
  const body = JSON.stringify({
    run_id: runId,
    cart_ref: opts?.cartRef ?? null,
    session_id: opts?.sessionId ?? null,
  })
  if (target) {
    const res = await fetch(`${target.base}/agents/save_to_cart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...target.headers },
      body,
    })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(`Report Builder save_to_cart ${res.status}: ${text}`)
    }
    return res.json()
  }
  return fetchJSON<SaveAgentToCartResponse>('/agents/save_to_cart', {
    method: 'POST',
    body,
  })
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
