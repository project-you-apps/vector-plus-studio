// Cart Builder V2 pipeline orchestrator.
// File[] → parse → chunk → embed → write → download-ready BuiltCart.
//
// Streams progress events for the UI to render per-stage status.

import type { ChunkOptions } from './chunker'
import { chunkSections } from './chunker'
import type { EmbedOptions } from './embedder/embed'
import { embedTexts } from './embedder/embed'
import { parseFile } from './parsers'
import type { Section } from './types'
import type { BuildCartOptions, BuiltCart } from './writer/npz'
import { buildCart } from './writer/npz'

export type PipelineStage =
  | 'idle'
  | 'parsing'
  | 'chunking'
  | 'embedding'
  | 'writing'
  | 'done'
  | 'error'

export interface PipelineProgress {
  stage: PipelineStage

  // Parsing-stage detail
  filesParsed?: number
  filesTotal?: number
  currentFile?: string

  // Chunking-stage detail
  sectionsTotal?: number

  // Embedding-stage detail
  embeddingsCompleted?: number
  embeddingsTotal?: number

  // Embedder model download (fires during loader.getEmbedder on first run)
  modelStatus?: string
  modelDownloadProgress?: number

  // Free-form status text for the UI
  message?: string

  // Populated when stage === 'error'
  errorMessage?: string
}

export interface PipelineOptions {
  cartName: string
  chunkOptions?: ChunkOptions
  embedOptions?: Omit<EmbedOptions, 'onBatch' | 'onProgress'>
  buildOptions?: Omit<BuildCartOptions, 'cartName'>
  onProgress?: (progress: PipelineProgress) => void
  /** Reject files larger than this. Default 50 MB per file. */
  maxFileSizeBytes?: number
  /** Reject builds that exceed this chunk count. Default 10,000 chunks. */
  maxChunks?: number
}

// Defensive caps to prevent browser OOM on hostile/oversized inputs.
// These are deliberately generous for legitimate use — most documents
// are <10MB; most carts are <2K chunks. The caps catch outliers, not
// reasonable workflows.
export const DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 // 50 MB per file
export const DEFAULT_MAX_CHUNKS_PER_BUILD = 10_000

export interface PipelineResult {
  cart: BuiltCart
  fileCount: number
  sectionCount: number
  chunkCount: number
}

/**
 * Orchestrate the full browser-side build pipeline.
 *
 * Stages (each emitted via onProgress):
 *   1. parsing  — sequentially extract sections from each file
 *   2. chunking — apply 300-word/50-overlap chunking
 *   3. embedding — batch embed chunks (model download on first call)
 *   4. writing  — pack NPZ + manifest + permissions
 *   5. done     — return BuiltCart for caller to download or mount
 *
 * Errors get one final progress event with stage='error' before throwing.
 */
export async function buildCartFromFiles(
  files: File[],
  options: PipelineOptions
): Promise<PipelineResult> {
  const onProgress = options.onProgress ?? (() => {})

  if (files.length === 0) {
    const err = new Error('No files provided to pipeline')
    onProgress({ stage: 'error', errorMessage: err.message })
    throw err
  }

  const maxFileSize = options.maxFileSizeBytes ?? DEFAULT_MAX_FILE_SIZE_BYTES
  const maxChunks = options.maxChunks ?? DEFAULT_MAX_CHUNKS_PER_BUILD

  try {
    // Pre-flight size check — fail fast on oversized files before any
    // parser/embedder work begins. Saves the user 30+ seconds of model
    // download on hopeless inputs.
    for (const file of files) {
      if (file.size > maxFileSize) {
        throw new Error(
          `File "${file.name}" is ${(file.size / (1024 * 1024)).toFixed(1)} MB, ` +
          `exceeds ${(maxFileSize / (1024 * 1024)).toFixed(0)} MB per-file cap. ` +
          `Split the document or raise maxFileSizeBytes.`
        )
      }
    }

    // ── Stage 1: parse ─────────────────────────────────────────────────
    onProgress({
      stage: 'parsing',
      filesParsed: 0,
      filesTotal: files.length,
      message: `Parsing ${files.length} file${files.length === 1 ? '' : 's'}…`,
    })

    const allSections: Section[] = []
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      onProgress({
        stage: 'parsing',
        filesParsed: i,
        filesTotal: files.length,
        currentFile: file.name,
        message: `Parsing ${file.name}…`,
      })
      const result = await parseFile(file)
      allSections.push(...result.sections)
    }
    onProgress({
      stage: 'parsing',
      filesParsed: files.length,
      filesTotal: files.length,
      sectionsTotal: allSections.length,
      message: `Parsed ${allSections.length} section${allSections.length === 1 ? '' : 's'}.`,
    })

    if (allSections.length === 0) {
      throw new Error(
        'No text content extracted. Are the files empty or in an unsupported format?'
      )
    }

    // ── Stage 2: chunk ─────────────────────────────────────────────────
    onProgress({
      stage: 'chunking',
      sectionsTotal: allSections.length,
      message: 'Splitting into overlapping chunks…',
    })
    const chunks = chunkSections(allSections, options.chunkOptions)

    if (chunks.length > maxChunks) {
      throw new Error(
        `Build would produce ${chunks.length.toLocaleString()} chunks, ` +
        `exceeds ${maxChunks.toLocaleString()}-chunk cap. ` +
        `Try fewer files, a larger chunkSize, or raise maxChunks.`
      )
    }

    // ── Stage 3: embed ─────────────────────────────────────────────────
    onProgress({
      stage: 'embedding',
      embeddingsCompleted: 0,
      embeddingsTotal: chunks.length,
      message: `Loading embedding model (~80MB on first run)…`,
    })

    const texts = chunks.map((c) => c.text)
    const embedResult = await embedTexts(texts, {
      ...options.embedOptions,
      onProgress: (modelProgress) => {
        onProgress({
          stage: 'embedding',
          embeddingsCompleted: 0,
          embeddingsTotal: chunks.length,
          modelStatus: modelProgress.status,
          modelDownloadProgress: modelProgress.progress,
          message: modelProgress.file
            ? `Downloading ${modelProgress.file}…`
            : `Loading model (${modelProgress.status})…`,
        })
      },
      onBatch: (completed, total) => {
        onProgress({
          stage: 'embedding',
          embeddingsCompleted: completed,
          embeddingsTotal: total,
          message: `Embedding chunks… ${completed}/${total}`,
        })
      },
    })

    // ── Stage 4: write ─────────────────────────────────────────────────
    onProgress({
      stage: 'writing',
      message: 'Building cart bundle…',
    })
    const cart = await buildCart(embedResult.embeddings, chunks, {
      cartName: options.cartName,
      ...options.buildOptions,
    })

    // ── Done ───────────────────────────────────────────────────────────
    onProgress({
      stage: 'done',
      message: `Built ${cart.cartFilename}: ${chunks.length} chunks, ${embedResult.dim} dims, backend ${embedResult.backend}.`,
    })

    return {
      cart,
      fileCount: files.length,
      sectionCount: allSections.length,
      chunkCount: chunks.length,
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    onProgress({ stage: 'error', errorMessage: message, message })
    throw err
  }
}
