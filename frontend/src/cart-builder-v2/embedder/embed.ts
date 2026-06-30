import type { Tensor } from '@huggingface/transformers'
import type { EmbedderBackend } from '../types'
import { forceWasmFallback, getActiveBackend, getEmbedder, type LoaderOptions } from './loader'

// Nomic v1.5 prefix convention. Matches Python reference build pipeline.
const DOC_PREFIX = 'search_document: '
const QUERY_PREFIX = 'search_query: '

const NOMIC_DIM = 768
const DEFAULT_BATCH_SIZE = 8

// Thrown when the WebGPU device is lost / hung / removed mid-build. Distinct
// from buffer-OOM (which is recoverable by halving batch size). On device-lost
// the embedder must be torn down and reloaded on WASM — no batch size helps a
// corpse. See [[CC_webgpu-device-lost-fallback-2026-06-29]] for the diagnosis.
//
// Parameter-property shorthand (`public readonly underlying: Error` in the
// constructor signature) is forbidden by the project's `erasableSyntaxOnly`
// TS option, so the field is declared explicitly and assigned in the body.
export class WebGpuDeviceLostError extends Error {
  public readonly underlying: Error
  constructor(underlying: Error) {
    super(`WebGPU device lost: ${underlying.message}`)
    this.name = 'WebGpuDeviceLostError'
    this.underlying = underlying
  }
}

export type PrefixMode = 'document' | 'query' | 'raw'

export interface EmbedOptions extends LoaderOptions {
  /** 'document' applies `search_document:` (default), 'query' applies `search_query:`, 'raw' skips prefixing. */
  prefixMode?: PrefixMode
  /** How many texts to push through the pipeline per call. Larger = faster but more GPU memory. */
  batchSize?: number
  /** Fires after each batch with (completedCount, totalCount). */
  onBatch?: (completed: number, total: number) => void
  /** Fires when the active embedder backend changes mid-stream (e.g. WebGPU
   *  device lost → silent failover to WASM). Lets the UI flip its backend
   *  badge so the user sees the graceful degradation rather than wondering
   *  why their build kept going after the console exploded. */
  onBackendChange?: (backend: EmbedderBackend) => void
}

export interface EmbedResult {
  embeddings: Float32Array // shape [count, dim], row-major
  dim: number
  count: number
  backend: EmbedderBackend
}

function applyPrefix(text: string, mode: PrefixMode): string {
  switch (mode) {
    case 'query':
      return QUERY_PREFIX + text
    case 'raw':
      return text
    case 'document':
    default:
      return DOC_PREFIX + text
  }
}

/**
 * Batch-embed a list of texts with the cached Nomic v1.5 model.
 * Returns a single contiguous Float32Array of shape [count, 768].
 *
 * Uses pooling='mean' and normalize=false to match the Python reference.
 * If your downstream needs L2-normalized vectors, normalize after the fact.
 */
export async function embedTexts(
  texts: string[],
  options: EmbedOptions = {}
): Promise<EmbedResult> {
  const {
    prefixMode = 'document',
    batchSize = DEFAULT_BATCH_SIZE,
    onBatch,
    onBackendChange,
    ...loaderOpts
  } = options

  if (texts.length === 0) {
    return {
      embeddings: new Float32Array(0),
      dim: NOMIC_DIM,
      count: 0,
      backend: getActiveBackend(),
    }
  }

  let embedder = await getEmbedder(loaderOpts)
  const prefixed = texts.map((t) => applyPrefix(t, prefixMode))
  const result = new Float32Array(texts.length * NOMIC_DIM)
  let completed = 0

  for (let i = 0; i < prefixed.length; i += batchSize) {
    const batch = prefixed.slice(i, i + batchSize)
    try {
      await embedBatchWithFallback(embedder, batch, result, i * NOMIC_DIM)
    } catch (err) {
      if (!(err instanceof WebGpuDeviceLostError)) throw err
      // WebGPU device died mid-build. Tear down the dead pipeline, reload on
      // WASM, swap the embedder reference, retry this batch, and continue
      // with WASM for every subsequent batch. The user gets their cart
      // instead of a half-built failure + page-reload requirement.
      console.warn(
        `[cart-builder embed] WebGPU device lost mid-build — reloading on ` +
        `WASM and continuing. (${err.underlying.message.slice(0, 120)})`,
      )
      embedder = await forceWasmFallback(loaderOpts)
      onBackendChange?.('wasm')
      await embedBatchWithFallback(embedder, batch, result, i * NOMIC_DIM)
    }
    completed += batch.length
    onBatch?.(completed, prefixed.length)
  }

  return {
    embeddings: result,
    dim: NOMIC_DIM,
    count: texts.length,
    backend: getActiveBackend(),
  }
}

/**
 * Embed a single batch with WebGPU-OOM-adaptive halving fallback.
 *
 * Two distinct failure modes get demuxed here:
 *
 * 1. **Device-lost / hung** (D3D12 DXGI_ERROR_DEVICE_HUNG, "Device is lost",
 *    "Device removed"): the WebGPU adapter is dead and no batch size will
 *    save us. Throws `WebGpuDeviceLostError` so the outer `embedTexts` loop
 *    can tear down the dead pipeline and reload on WASM. Before 2026-06-29
 *    this case got misclassified as buffer-OOM via the `/failed to call
 *    OrtRun/i` match and went through 4 useless halving retries before
 *    surfacing a confusing error to the user.
 *
 * 2. **Buffer-OOM** (WebGPU 2 GiB single-allocation cap; transformer
 *    intermediate tensors scale with batch × seq_len × hidden_dim × N_layers
 *    and can blow past this when chunks happen to be long or batch size is
 *    high): halve the batch and recurse. Worst case (batch of 1 still fails)
 *    the underlying error is re-thrown — a single input that doesn't fit is
 *    a genuine model-side problem we shouldn't silently mask.
 *
 * Order matters: check device-lost FIRST because the OrtRun regex below would
 * otherwise match its error string and trigger the halving loop.
 */
async function embedBatchWithFallback(
  embedder: (texts: string[], opts: Record<string, unknown>) => Promise<Tensor>,
  batch: string[],
  result: Float32Array,
  outFloatOffset: number
): Promise<void> {
  let out: Tensor
  try {
    out = (await embedder(batch, {
      pooling: 'mean',
      normalize: false,
    })) as Tensor
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err)
    const isDeviceLost =
      /Device is lost/i.test(msg) ||
      /DXGI_ERROR_DEVICE_HUNG/i.test(msg) ||
      /Device removed/i.test(msg) ||
      /GPU(?:Device)?(?: was)? destroyed/i.test(msg)
    if (isDeviceLost) {
      throw new WebGpuDeviceLostError(
        err instanceof Error ? err : new Error(String(err)),
      )
    }
    const isWebGpuOom =
      /max buffer size limit/i.test(msg) ||
      /WebGPU validation failed/i.test(msg) ||
      /failed to call OrtRun/i.test(msg)
    if (!isWebGpuOom || batch.length === 1) {
      throw err
    }
    const half = Math.ceil(batch.length / 2)
    console.warn(
      `[cart-builder embed] WebGPU buffer-cap hit at batchSize=${batch.length}; ` +
      `halving to ${half} and retrying. (${msg.slice(0, 120)})`
    )
    await embedBatchWithFallback(embedder, batch.slice(0, half), result, outFloatOffset)
    await embedBatchWithFallback(
      embedder,
      batch.slice(half),
      result,
      outFloatOffset + half * NOMIC_DIM
    )
    return
  }

  if (out.dims.length !== 2) {
    throw new Error(
      `Unexpected embedding shape: [${out.dims.join(', ')}] (expected [N, ${NOMIC_DIM}])`
    )
  }
  if (out.dims[1] !== NOMIC_DIM) {
    throw new Error(
      `Unexpected embedding dim: ${out.dims[1]} (expected ${NOMIC_DIM})`
    )
  }
  if (out.dims[0] !== batch.length) {
    throw new Error(
      `Embedding batch size mismatch: returned ${out.dims[0]} vectors for ${batch.length} inputs`
    )
  }
  result.set(out.data as Float32Array, outFloatOffset)
}

/**
 * Convenience wrapper for single-query embedding. Applies the `search_query:`
 * prefix per Nomic v1.5 spec. Returns a flat Float32Array of length 768.
 */
export async function embedQuery(
  text: string,
  options: Omit<EmbedOptions, 'prefixMode' | 'batchSize' | 'onBatch'> = {}
): Promise<Float32Array> {
  const result = await embedTexts([text], { ...options, prefixMode: 'query' })
  // subarray returns a view — fine for one-off use; consumers needing
  // ownership should slice() instead.
  return result.embeddings.subarray(0, result.dim)
}

export { NOMIC_DIM }
