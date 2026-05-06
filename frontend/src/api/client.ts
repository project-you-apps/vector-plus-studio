import type {
  CartridgeInfo, SearchResponse, StatusResponse, DeletedPattern,
  SearchMode, PatternResponse,
  MemboxCartInfo, MemboxStatus, MemboxImprintRequest, MemboxMountRequest,
} from './types'

// Set VITE_API_BASE at build time for hosted deploys (e.g. '/vps/api'). Local
// dev uses '/api' which the Vite proxy routes to localhost:8000.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'

async function fetchJSON<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
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

export async function getCartridges(): Promise<CartridgeInfo[]> {
  const data = await fetchJSON<{ cartridges: CartridgeInfo[] }>('/cartridges')
  return data.cartridges
}

export async function browseForCartridge(): Promise<string> {
  const data = await fetchJSON<{ path: string }>('/browse')
  return data.path
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

export async function forgeCartridge(name: string, files: File[]) {
  const form = new FormData()
  form.append('name', name)
  for (const f of files) {
    form.append('files', f)
  }
  const res = await fetch(`${BASE}/forge`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Forge failed: ${res.status}`)
  return res.json() as Promise<{ success: boolean; message: string }>
}
