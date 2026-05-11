// Cart Builder API client — wraps /api/cartbuilder/* (Phase 1 backend).
// Mirrors the 15 backend routes from api/cartbuilder.py with typed
// request / response shapes. Used by the future Cart Builder screen
// (Phase 2 frontend port).

// Set VITE_API_BASE at build time for hosted deploys (e.g. '/vps/api'). Local
// dev uses '/api' which the Vite proxy routes to localhost:8000.
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'
const BASE = `${API_BASE}/cartbuilder`

async function fetchJSON<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Cart Builder API error ${res.status}: ${text}`)
  }
  return res.json()
}

// ---------------------------------------------------------------------------
// Types — public-facing shapes from the FastAPI router.
// Keep aligned with api/cartbuilder.py response objects.
// ---------------------------------------------------------------------------

export interface CartBuilderFile {
  id: string
  name: string
  type: string
  size: number
  chunks: number
  chars: number
  preview: string
  owner: string
  description: string
  tags: string[]
  from_cart?: boolean
}

export interface CartBuilderListedCart {
  name: string
  filename: string
  size_mb: number
  passages: number | string
  modified: string
  path: string
  folder?: string
}

export interface CartBuilderSubdir {
  name: string
  path: string
}

export interface CartBuilderDoc {
  name: string
  path: string
  size: number
  type: string
}

export interface CartBuilderCartsResponse {
  carts: CartBuilderListedCart[]
  subdirs: CartBuilderSubdir[]
  docs?: CartBuilderDoc[]
  folders: string[]
  current_path: string
}

export interface CartBuilderBrowseResponse {
  path: string
  dirs: string[]
  parent?: string | null
  is_root: boolean
}

export interface CartBuilderPattern0 {
  cart_name: string
  file_count: number
  total_chunks: number
  files: { name: string; chunks: number; owner: string }[]
  created: string
}

export interface CartBuilderBuildState {
  status: 'idle' | 'building' | 'done' | 'error' | string
  progress: number
  chunks_done?: number
  chunks_total?: number
  cart_path?: string | null
  error?: string | null
  message?: string
}

export interface CartBuilderHasChanges {
  has_files: boolean
  has_built: boolean
  file_count: number
  message: string
}

export interface CartBuilderLoadCartResponse {
  ok: boolean
  cart_path: string
  cart_name: string
  files: CartBuilderFile[]
  total_passages: number
  total_sources: number
  truncated: boolean
  showing: number
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

export async function uploadFiles(files: File[]): Promise<{ files: CartBuilderFile[] }> {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Upload failed ${res.status}: ${text}`)
  }
  return res.json()
}

export async function listFiles(): Promise<{ files: CartBuilderFile[] }> {
  return fetchJSON('/files')
}

export async function setMetadata(payload: {
  file_id: string
  owner?: string
  description?: string
  tags?: string[]
}): Promise<{ ok: boolean }> {
  return fetchJSON('/metadata', { method: 'POST', body: JSON.stringify(payload) })
}

export async function ingestPath(path: string): Promise<{ file: CartBuilderFile }> {
  return fetchJSON('/ingest', { method: 'POST', body: JSON.stringify({ path }) })
}

export async function getPattern0(name = 'hackathon-cart'): Promise<CartBuilderPattern0> {
  return fetchJSON(`/pattern0?name=${encodeURIComponent(name)}`)
}

export async function buildCart(cart_name: string): Promise<{
  status: string
  cart_name: string
  chunks: number
}> {
  return fetchJSON('/build', { method: 'POST', body: JSON.stringify({ cart_name }) })
}

export async function getBuildStatus(): Promise<CartBuilderBuildState> {
  return fetchJSON('/build/status')
}

export async function listCarts(path = ''): Promise<CartBuilderCartsResponse> {
  const q = path ? `?path=${encodeURIComponent(path)}` : ''
  return fetchJSON(`/carts${q}`)
}

export async function getCartFolders(): Promise<{ folders: string[] }> {
  return fetchJSON('/cart_folders')
}

export async function addCartFolder(folder: string): Promise<{ folders: string[] }> {
  return fetchJSON('/cart_folders', { method: 'POST', body: JSON.stringify({ folder }) })
}

export async function removeCartFolder(folder: string): Promise<{ folders: string[] }> {
  return fetchJSON('/cart_folders', { method: 'DELETE', body: JSON.stringify({ folder }) })
}

export async function browseFolders(path = ''): Promise<CartBuilderBrowseResponse> {
  const q = path ? `?path=${encodeURIComponent(path)}` : ''
  return fetchJSON(`/browse${q}`)
}

export async function loadCart(cart_path: string): Promise<CartBuilderLoadCartResponse> {
  return fetchJSON('/load_cart', { method: 'POST', body: JSON.stringify({ cart_path }) })
}

export async function clearWorkspace(): Promise<{ ok: boolean }> {
  return fetchJSON('/clear_workspace', { method: 'POST' })
}

export async function getHasChanges(): Promise<CartBuilderHasChanges> {
  return fetchJSON('/has_changes')
}

export interface BuildToFolderResponse {
  cart_path: string
  mounted_filename: string
  folder: string
}

// Write a browser-built cart bundle (cart blob + manifest blob + permissions
// blob) to a server-side folder. Used by Edit Carts "New Cart" flow where
// the user picks the destination folder via the server-side FolderPickerModal.
// Server honors the user's permissions sidecar (no forced read-only). Gated
// on the server by VPS_READ_ONLY — writable instances only.
// Sentinel error subclass so the UI can disambiguate "cart already exists"
// from other failures (network, validation, permissions) without parsing
// error message strings. Carries the server's detail so the prompt can
// include the path the user picked.
export class CartExistsError extends Error {
  readonly detail: string
  constructor(detail: string) {
    super(detail)
    this.name = 'CartExistsError'
    this.detail = detail
  }
}

export async function buildToFolder(payload: {
  cartBlob: Blob
  manifestBlob: Blob
  permissionsBlob: Blob
  folder: string
  cartName: string
  replace?: boolean
}): Promise<BuildToFolderResponse> {
  const form = new FormData()
  form.append('cart', payload.cartBlob, `${payload.cartName}.cart.npz`)
  form.append('manifest', payload.manifestBlob, `${payload.cartName}.cart_manifest.json`)
  form.append('permissions', payload.permissionsBlob, `${payload.cartName}.permissions.json`)
  form.append('folder', payload.folder)
  form.append('cart_name', payload.cartName)
  form.append('replace', payload.replace ? 'true' : 'false')
  const res = await fetch(`${BASE}/build-to-folder`, { method: 'POST', body: form })
  if (!res.ok) {
    let detail = `${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* keep */ }
    if (res.status === 409) throw new CartExistsError(detail)
    throw new Error(detail)
  }
  return res.json()
}
