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
