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

/**
 * Probe whether WebGPU can actually sustain the embed workload (not just
 * "does the adapter exist"). Performs a tiny real embed of a single token
 * sequence on WebGPU; if it succeeds, the embedder is cached and returned
 * as 'webgpu' active backend so the first real Build button click reuses it
 * instead of paying the load cost twice. If it fails, the embedder falls
 * through to WASM, gets cached as WASM, and the badge can downgrade upfront.
 *
 * **Why a real embed and not just `requestAdapter()`?** On constrained
 * devices (integrated GPUs sharing system RAM, older drivers, Chromium
 * Windows quirks), `requestAdapter()` returns a valid adapter but the actual
 * Nomic inference hangs the D3D12 device mid-stream (`DXGI_ERROR_DEVICE_HUNG`).
 * The only reliable test is "run the real workload." This costs ~10-30 sec
 * one-time (model download + tiny embed) but eliminates mid-build surprises.
 *
 * Time-bounded: if the probe doesn't complete in `timeoutMs`, treat as
 * unusable. Default 60s covers slow connections + cold model download.
 */
export async function probeWebGpuCapability(
  options: { timeoutMs?: number; modelId?: string } = {},
): Promise<EmbedderBackend> {
  const timeoutMs = options.timeoutMs ?? 60_000
  if (!hasWebGPU()) {
    // No adapter at all — load WASM eagerly so the badge is honest and the
    // first build doesn't pay an additional load cost.
    await getEmbedder({ device: 'wasm', modelId: options.modelId })
    return 'wasm'
  }
  try {
    const probeRun = (async () => {
      const e = await getEmbedder({ device: 'webgpu', modelId: options.modelId })
      // Real workload: embed one short string. If WebGPU is going to hang on
      // this device, it'll hang here — before the user has invested any time.
      await e(['probe'], { pooling: 'mean', normalize: false })
      return getActiveBackend()
    })()
    const timeout = new Promise<EmbedderBackend>((_, reject) =>
      setTimeout(
        () => reject(new Error(`WebGPU probe timed out after ${timeoutMs}ms`)),
        timeoutMs,
      ),
    )
    return await Promise.race([probeRun, timeout])
  } catch (err) {
    // Probe failed — either device-lost, OOM on tiny input (driver bug),
    // timeout, or any other surprise. Tear down the WebGPU pipeline and
    // load WASM so the cart-builder is ready when the user clicks Build.
    const msg = err instanceof Error ? err.message : String(err)
    console.warn(
      `[cart-builder probe] WebGPU capability probe failed — falling back ` +
      `to WASM. (${msg.slice(0, 200)})`,
    )
    await forceWasmFallback({ modelId: options.modelId })
    return 'wasm'
  }
}

/** Exposed so the UI can show "no GPU adapter at all" upfront vs. probe-failure. */
export function webGpuAdapterAvailable(): boolean {
  return hasWebGPU()
}
