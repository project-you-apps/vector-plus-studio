import type { Tensor } from '@huggingface/transformers'
import type { EmbedderBackend } from '../types'
import { getActiveBackend, getEmbedder, type LoaderOptions } from './loader'

// Nomic v1.5 prefix convention. Matches Python reference build pipeline.
const DOC_PREFIX = 'search_document: '
const QUERY_PREFIX = 'search_query: '

const NOMIC_DIM = 768
const DEFAULT_BATCH_SIZE = 16

export type PrefixMode = 'document' | 'query' | 'raw'

export interface EmbedOptions extends LoaderOptions {
  /** 'document' applies `search_document:` (default), 'query' applies `search_query:`, 'raw' skips prefixing. */
  prefixMode?: PrefixMode
  /** How many texts to push through the pipeline per call. Larger = faster but more GPU memory. */
  batchSize?: number
  /** Fires after each batch with (completedCount, totalCount). */
  onBatch?: (completed: number, total: number) => void
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

  const embedder = await getEmbedder(loaderOpts)
  const prefixed = texts.map((t) => applyPrefix(t, prefixMode))
  const result = new Float32Array(texts.length * NOMIC_DIM)
  let completed = 0

  for (let i = 0; i < prefixed.length; i += batchSize) {
    const batch = prefixed.slice(i, i + batchSize)
    // transformers.js accepts string[] natively. Output is a Tensor with
    // dims [N, dim] when pooling is set (otherwise [N, seq_len, dim]).
    const out = (await embedder(batch, {
      pooling: 'mean',
      normalize: false,
    })) as Tensor

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
    result.set(out.data as Float32Array, i * NOMIC_DIM)
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
