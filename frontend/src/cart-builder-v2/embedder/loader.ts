import { pipeline } from '@huggingface/transformers'
import type { Tensor } from '@huggingface/transformers'
import type { EmbedderBackend, ProgressCallback } from '../types'

// Singleton embedder. transformers.js model download is ~30-100MB; we want
// exactly one load per session, shared across every cart-build pipeline run.
//
// Verified at 6th-decimal parity with the Python reference (server-side
// build) on 2026-05-06 via webgpu-smoketest. WebGPU first, WASM fallback
// for browsers without GPU compute.

export interface FeatureExtractionOptions {
  pooling?: 'none' | 'mean' | 'cls'
  normalize?: boolean
}

// transformers.js's `pipeline()` returns a union of every task pipeline,
// which TS can't narrow on the literal task name in v4.x. We assert to a
// callable shape that captures only what we use at runtime.
export interface EmbedderInstance {
  (
    texts: string | string[],
    options?: FeatureExtractionOptions
  ): Promise<Tensor>
}

export interface LoaderOptions {
  device?: 'webgpu' | 'wasm' | 'auto'
  dtype?: 'fp32' | 'fp16'
  modelId?: string
  onProgress?: ProgressCallback
}

const DEFAULT_MODEL_ID = 'nomic-ai/nomic-embed-text-v1.5'

let cached: EmbedderInstance | null = null
let loading: Promise<EmbedderInstance> | null = null
let activeBackend: EmbedderBackend = 'unknown'

function hasWebGPU(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

export async function getEmbedder(
  options: LoaderOptions = {}
): Promise<EmbedderInstance> {
  if (cached) return cached
  if (loading) return loading

  const modelId = options.modelId ?? DEFAULT_MODEL_ID
  const dtype = options.dtype ?? 'fp32'
  const requested = options.device ?? 'auto'

  // Build attempt order. 'auto' tries WebGPU then WASM. Explicit 'webgpu'
  // still falls back to WASM on failure (gives the user the cart anyway).
  // Explicit 'wasm' skips WebGPU entirely.
  const attempts: Array<'webgpu' | 'wasm'> =
    requested === 'wasm'
      ? ['wasm']
      : hasWebGPU()
        ? ['webgpu', 'wasm']
        : ['wasm']

  loading = (async (): Promise<EmbedderInstance> => {
    let lastError: unknown
    for (const device of attempts) {
      try {
        const instance = (await pipeline('feature-extraction', modelId, {
          device,
          dtype,
          progress_callback: options.onProgress
            ? (data: unknown) => {
                const d = data as Record<string, unknown>
                options.onProgress!({
                  status: String(d.status ?? 'unknown'),
                  file: d.file as string | undefined,
                  progress: d.progress as number | undefined,
                  loaded: d.loaded as number | undefined,
                  total: d.total as number | undefined,
                })
              }
            : undefined,
        })) as EmbedderInstance
        cached = instance
        activeBackend = device
        return instance
      } catch (err) {
        lastError = err
        // Try next device
      }
    }
    loading = null
    throw new Error(
      `Failed to load Nomic embedder on any backend (tried: ${attempts.join(
        ', '
      )}). Last error: ${
        lastError instanceof Error ? lastError.message : String(lastError)
      }`
    )
  })()

  return loading
}

export function getActiveBackend(): EmbedderBackend {
  return activeBackend
}

/** Drop the cached embedder instance. Mainly useful for tests / re-init. */
export function clearEmbedderCache(): void {
  cached = null
  loading = null
  activeBackend = 'unknown'
}

/**
 * Tear down the current (presumed dead) embedder and reload on WASM. Used
 * when WebGPU device-loss is detected mid-build — the WebGPU pipeline can't
 * be recovered by re-trying, so we drop it and continue on CPU. Returns the
 * fresh WASM embedder, ready to embed the batch that just failed.
 */
export async function forceWasmFallback(
  options: LoaderOptions = {},
): Promise<EmbedderInstance> {
  clearEmbedderCache()
  return getEmbedder({ ...options, device: 'wasm' })
}

// Minimal type shapes for adapter.info / adapter.limits we read. The DOM
// lib types for WebGPU are in flux across browsers; rather than fight
// lib.dom.d.ts version skew, we cast through these structural types at the
// inspection site.
interface AdapterInfo {
  vendor?: string
  architecture?: string
  device?: string
  description?: string
}

interface AdapterShape {
  limits: { maxBufferSize: number }
  info?: AdapterInfo
}

/**
 * Get the WebGL2 `WEBGL_debug_renderer_info` output. This API predates WebGPU
 * and (unlike WebGPU's `adapter.info`) typically exposes the FULL GPU name
 * — e.g. "ANGLE (NVIDIA, NVIDIA GeForce RTX 4080 SUPER (0x00002702) ...)"
 * or "ANGLE (AMD, AMD Radeon(TM) 780M Graphics, D3D11)" — even on browsers
 * that mask WebGPU's adapter.info.description.
 *
 * Why this exists: tonight's WebGPU adapter.info diagnostic showed empty
 * description on both Andy's home machine RTX 4080 Super AND the laptop's
 * Ryzen 7000 integrated Radeon. WebGL2 fills the gap. The renderer string
 * is what Stage 2 of Patch 6 will pattern-match on to distinguish integrated
 * from discrete GPUs (the only signal that survives Chrome's anti-fingerprint
 * choices in WebGPU). Verified working on Edge + Chrome 149 on Windows.
 *
 * Returns null if WebGL2 is unavailable or the extension is masked
 * (some configs return generic strings like "WebGL" or "Google SwiftShader").
 */
function getWebGlRendererDiag(): { renderer: string; vendor: string } | null {
  try {
    if (typeof document === 'undefined') return null
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null
    if (!gl) return null
    const ext = gl.getExtension('WEBGL_debug_renderer_info')
    if (!ext) return null
    return {
      renderer: String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) ?? ''),
      vendor: String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) ?? ''),
    }
  } catch {
    return null
  }
}

/**
 * Heuristic: is this WebGPU adapter likely to sustain the Nomic embed model
 * under real workload (300+ token chunks, batch=8, hundreds of chunks)?
 *
 * Returns false for adapters that report:
 *   1. `maxBufferSize` at or below the WebGPU spec default of 256 MiB — strong
 *      signal of constrained hardware. Discrete GPUs typically expose 1 GiB+.
 *      Nomic intermediate tensors during inference can need >256 MiB of
 *      contiguous buffer during peak, so devices stuck at the default hang
 *      under sustained workload (DXGI_ERROR_DEVICE_HUNG on Windows D3D12).
 *   2. `adapter.info` description matching known integrated-GPU patterns —
 *      AMD APU "Radeon(TM) Graphics" (no discrete model id), Intel
 *      Iris/HD/UHD prefixes. Apple Silicon and discrete NVIDIA/AMD are
 *      treated as capable.
 *
 * False negative cost: an under-spec'd discrete GPU runs on WASM unnecessarily.
 * False positive cost: a weak GPU attempts WebGPU and hangs mid-build,
 * triggering the mid-stream WASM fallback (`embed.ts`'s WebGpuDeviceLostError
 * path) — defended in depth.
 *
 * Origin: 2026-06-29 — Andy's laptop (AMD Ryzen 7000 + integrated Radeon,
 * 489 MB dedicated VRAM per dxdiag) crawled at 1%/min then hung. The first
 * version of this probe ran a 1-token embed which succeeded on the integrated
 * GPU (trivial workload) and incorrectly marked WebGPU as adequate.
 */
function isWebGpuLikelyAdequate(adapter: AdapterShape): boolean {
  // Signal 1: maxBufferSize at the WebGPU default suggests a constrained
  // device. Keep the comparison <= to also catch the off-by-one case.
  if (adapter.limits.maxBufferSize <= 256 * 1024 * 1024) return false

  // Signal 2: adapter.info description patterns for known-integrated GPUs.
  // Chrome 124+ exposes adapter.info synchronously; older browsers don't
  // populate it, in which case we trust Signal 1 alone.
  const info = adapter.info
  if (info) {
    const vendor = (info.vendor ?? '').toLowerCase()
    const desc = (info.description ?? '').toLowerCase()
    // AMD APU integrated graphics: typically described as "AMD Radeon(TM)
    // Graphics" with NO discrete model number (RX, PRO, FirePro etc.).
    if (vendor === 'amd' && /^amd radeon\(tm\) graphics\s*$/i.test(desc)) {
      return false
    }
    // Intel iGPUs: Iris, HD, UHD prefixes are characteristic of integrated
    // graphics. Intel Arc discrete cards use different names ("Arc A380",
    // "Arc A770") and aren't matched here.
    if (vendor === 'intel' && /\b(iris|hd|uhd)\b/i.test(desc)) {
      return false
    }
  }

  return true
}

/**
 * Decide whether to use WebGPU or WASM for this device, then PRE-LOAD the
 * model on the chosen backend so the first Build click skips model-load
 * cost. Replaces the 2026-06-29 morning version of this function which ran
 * a 1-token embed — that approach proved inadequate (1-token workload
 * succeeds on integrated GPUs that crawl-then-hang on real chunks). The
 * new approach inspects `adapter.limits` + `adapter.info` instead — instant
 * decision, no model load on the wrong backend, no false positives from
 * trivially-small probe inputs.
 *
 * Mid-stream WASM fallback (`embed.ts`'s WebGpuDeviceLostError handler) is
 * still in place as a defense-in-depth — if this heuristic misjudges and
 * the device hangs anyway, the build will recover and complete on WASM.
 *
 * `timeoutMs` retained for backward compat with the old API surface but no
 * longer used in the body — adapter inspection is sub-100ms by construction.
 */
export async function probeWebGpuCapability(
  options: { timeoutMs?: number; modelId?: string } = {},
): Promise<EmbedderBackend> {
  void options.timeoutMs // retained for caller compat; not used in inspection path
  if (!hasWebGPU()) {
    // No adapter at all — pre-load WASM eagerly so the badge is honest and
    // the first Build click skips model-download cost.
    await getEmbedder({ device: 'wasm', modelId: options.modelId })
    return 'wasm'
  }

  let adapter: AdapterShape | null = null
  try {
    const nav = navigator as Navigator & {
      gpu: { requestAdapter: () => Promise<AdapterShape | null> }
    }
    adapter = await nav.gpu.requestAdapter()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.warn(
      `[cart-builder probe] requestAdapter threw — falling back to WASM. ` +
      `(${msg.slice(0, 200)})`,
    )
    await getEmbedder({ device: 'wasm', modelId: options.modelId })
    return 'wasm'
  }

  if (!adapter) {
    console.warn(
      '[cart-builder probe] No WebGPU adapter returned — falling back to WASM.',
    )
    await getEmbedder({ device: 'wasm', modelId: options.modelId })
    return 'wasm'
  }

  // 2026-06-30 Patch 6 Stage 1: diagnostic logging at probe decision time.
  // The 06-29 heuristic (Patch 5) missed Andy's Ryzen 7000 + integrated
  // Radeon — adapter inspection returned 'webgpu' but the device couldn't
  // sustain real workload. Without seeing what `adapter.info` and
  // `adapter.limits` actually report on the offending device, refining the
  // heuristic is guesswork. Logging every probe decision lets us refine
  // from real data on every machine that hits this code path.
  const diagInfo = adapter.info ?? {}
  const diagLimits = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxBufferSize_MiB:
      Math.round(adapter.limits.maxBufferSize / (1024 * 1024)),
  }
  const adequateVerdict = isWebGpuLikelyAdequate(adapter)
  // Serialize the fields explicitly into the log message rather than passing
  // raw objects — Chrome renders raw GPUAdapterInfo / Limits objects as
  // type-name placeholders ("GPUAdapterInfo", "Object") that aren't
  // copy-pasteable without clicking to expand. The explicit string form
  // makes the log line self-contained for sharing back during diagnostic.
  const infoStr = JSON.stringify({
    vendor: diagInfo.vendor ?? '',
    architecture: diagInfo.architecture ?? '',
    device: diagInfo.device ?? '',
    description: diagInfo.description ?? '',
  })
  console.log(
    `[cart-builder probe] adapter.info=${infoStr} ` +
    `maxBufferSize=${diagLimits.maxBufferSize_MiB} MiB (${diagLimits.maxBufferSize} bytes) ` +
    `heuristic verdict=${adequateVerdict ? 'webgpu' : 'wasm'}`,
  )

  // 2026-06-30 PM: WebGL2 renderer diagnostic. WebGPU's adapter.info comes
  // back with empty description on Chromium/Windows (anti-fingerprint
  // choice); WebGL2's WEBGL_debug_renderer_info still exposes the full GPU
  // name and is what Stage 2 of Patch 6 will pattern-match on. Logging here
  // alongside the WebGPU diagnostic lets us see both signals side-by-side
  // on every machine that hits this code, no manual console snippet needed.
  // Also logs navigator.deviceMemory + hardwareConcurrency as secondary
  // signals (laptop vs workstation rough discriminator when GPU name is
  // ambiguous or masked).
  const webglDiag = getWebGlRendererDiag()
  const nav = navigator as Navigator & { deviceMemory?: number }
  console.log(
    `[cart-builder probe] webgl_renderer=${
      JSON.stringify(webglDiag?.renderer ?? '(unavailable)')
    } webgl_vendor=${
      JSON.stringify(webglDiag?.vendor ?? '(unavailable)')
    } deviceMemory=${nav.deviceMemory ?? 'unknown'} ` +
    `hardwareConcurrency=${navigator.hardwareConcurrency ?? 'unknown'}`,
  )

  if (!adequateVerdict) {
    const desc = diagInfo.description ?? '(no description)'
    const maxBuf = diagLimits.maxBufferSize_MiB
    console.warn(
      `[cart-builder probe] WebGPU adapter inadequate for sustained embed ` +
      `workload (device="${desc}", maxBufferSize=${maxBuf} MiB). Pre-loading ` +
      `WASM instead — build will be slower but will complete reliably.`,
    )
    await getEmbedder({ device: 'wasm', modelId: options.modelId })
    return 'wasm'
  }

  // Adapter looks adequate. Pre-load on WebGPU.
  try {
    await getEmbedder({ device: 'webgpu', modelId: options.modelId })
    return 'webgpu'
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    console.warn(
      `[cart-builder probe] WebGPU model load failed despite adequate-looking ` +
      `adapter — falling back to WASM. (${msg.slice(0, 200)})`,
    )
    await forceWasmFallback({ modelId: options.modelId })
    return 'wasm'
  }
}

/** Exposed so the UI can show "no GPU adapter at all" upfront vs. probe-failure. */
export function webGpuAdapterAvailable(): boolean {
  return hasWebGPU()
}
