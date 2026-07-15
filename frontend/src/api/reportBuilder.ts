// Report Builder API client — thin wrapper around the local exe on
// http://127.0.0.1:7880. Runs the Wave 1 report engine + Agents engine
// against local .cart.npz files, so `local:` cart_refs stay on the
// user's machine end-to-end (only the anonymized prompt + retrieved
// context leave for LLM synthesis).
//
// The pill / pair modal / token storage share shape with Image Builder
// (see api/imageBuilder.ts). Same DESKTOP_HELPER_TOKEN_KEY backs all
// three Builders — one pairing unlocks Cart Builder + Image Builder +
// Report Builder.

import { DESKTOP_HELPER_TOKEN_KEY } from '../store/appStore'

// Fixed for MVP; Report Builder scans 7880-7889 for a free slot and
// records the actual bound port in its /capabilities response. Runtime
// origin is derived from the store's cached caps after detect.
export const REPORT_BUILDER_DEFAULT_PORT = 7880
export const REPORT_BUILDER_DEFAULT_ORIGIN = `http://127.0.0.1:${REPORT_BUILDER_DEFAULT_PORT}`

// Report Builder /capabilities response — matches main.py.
export interface ReportBuilderCapabilities {
  exe: string
  version?: string
  port: number
  capabilities: string[]
  reports: string[]
  agents: string[]
  wave2_pending: string[]
  vision: boolean
  cart_dir: string
}

export async function probeReportBuilder(signal?: AbortSignal): Promise<
  { ok: true; capabilities: ReportBuilderCapabilities }
  | { ok: false; reason: string }
> {
  try {
    const healthResp = await fetch(`${REPORT_BUILDER_DEFAULT_ORIGIN}/health`, {
      method: 'GET',
      signal,
    })
    if (!healthResp.ok) {
      return { ok: false, reason: `health ${healthResp.status}` }
    }
    const capsResp = await fetch(`${REPORT_BUILDER_DEFAULT_ORIGIN}/capabilities`, {
      method: 'GET',
      signal,
    })
    if (!capsResp.ok) {
      return { ok: false, reason: `capabilities ${capsResp.status}` }
    }
    const capabilities = (await capsResp.json()) as ReportBuilderCapabilities
    return { ok: true, capabilities }
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : 'probe failed' }
  }
}

// Attempt to /pair the paste token against Report Builder. Same
// handshake shape as Image Builder / Desktop Cart Builder. On success,
// the caller persists the token in localStorage under the shared key
// so subsequent report + agent runs pick it up automatically.
export async function pairReportBuilder(
  token: string,
  origin: string = REPORT_BUILDER_DEFAULT_ORIGIN,
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

// Read the shared bearer token from localStorage. Same key as Image /
// Desktop Builders — pairing any one persists a token here.
export function readReportBuilderToken(): string | null {
  try {
    return localStorage.getItem(DESKTOP_HELPER_TOKEN_KEY)
  } catch {
    return null
  }
}

// A cart entry as enumerated by GET /reports/carts on the local exe.
// Fields match the droplet's shape so the frontend selector can splice
// them together without a schema divergence.
export interface LocalReportCartEntry {
  id: string
  display_name: string
  report_compatible: boolean
  format: 'npz' | string
  location: 'local'
}
