// Cart Builder V2 pipeline orchestrator.
// File[] → parse → chunk → embed → write → download-ready BuiltCart.
//
// Streams progress events for the UI to render per-stage status.

import type { ChunkOptions } from './chunker'
import { chunkSections } from './chunker'
import type { EmbedOptions } from './embedder/embed'
import { embedTexts } from './embedder/embed'
import {
  classifyPdf,
  isImageFile,
  parseFile,
  parseViaImageBuilder,
} from './parsers'
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
  embedOptions?: Omit<EmbedOptions, 'onBatch' | 'onProgress' | 'onBackendChange'>
  buildOptions?: Omit<BuildCartOptions, 'cartName'>
  onProgress?: (progress: PipelineProgress) => void
  /** Fires if the embedder backend changes mid-build (e.g. WebGPU → WASM
   *  failover on device loss). Lets the UI flip its badge so the user sees
   *  the graceful degradation rather than wondering why the console exploded
   *  but the build kept going. */
  onBackendChange?: (backend: import('./types').EmbedderBackend) => void
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

// Per-file route decision (Day 2). Mirrors the classify Option C heuristic:
// images always go to Image Builder, PDFs are text-checked, everything else
// stays on the existing chunker fast path. The router is intentionally
// separate from the parsers so UI code can call it too (per-file badges).
//
// 'text' = existing parseFile → chunker path
// 'image' = POST to Image Builder → markdown + graphics + tables
// 'scanned' = same as 'image' route, but reached via PDF text-check
// 'pending' = PDF classification hasn't run yet (browser-side, async pdfjs)
export type FileRoute = 'text' | 'image' | 'scanned' | 'pending'

/**
 * Synchronous per-file route decision based on MIME/extension alone.
 *
 * Returns 'image' for any image MIME/extension, 'text' for everything else
 * INCLUDING PDFs — PDFs need an async classifyPdf() call to decide between
 * 'text' and 'scanned', so this sync helper returns 'pending' for PDFs and
 * the caller drives the async follow-up. Used by the UI to paint badges
 * without triggering the pdfjs classification eagerly.
 */
export function syncRouteForFile(file: File): FileRoute {
  if (isImageFile(file)) return 'image'
  const name = file.name.toLowerCase()
  if (name.endsWith('.pdf') || file.type === 'application/pdf') return 'pending'
  return 'text'
}

/**
 * Full async route decision. For PDFs, runs classifyPdf(); everything else
 * matches syncRouteForFile. Used by the pipeline to decide where to send
 * each file at build time, and by the badge renderer to upgrade a PDF's
 * label from "PDF (checking…)" to "TEXT PDF" or "SCANNED PDF".
 */
export async function routeForFile(file: File): Promise<'text' | 'image' | 'scanned'> {
  if (isImageFile(file)) return 'image'
  const name = file.name.toLowerCase()
  if (name.endsWith('.pdf') || file.type === 'application/pdf') {
    const kind = await classifyPdf(file)
    return kind === 'text' ? 'text' : 'scanned'
  }
  return 'text'
}

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

    // Split incoming files into text-only sections (feed the chunker) and
    // graphic + table sections (Day 2 — bypass the chunker; each is its own
    // pattern). Text sections come from either the existing parser fast
    // path or from Image Builder's markdown output (via parseViaImageBuilder).
    const textSections: Section[] = []
    const graphicSections: Section[] = []
    const tableSections: Section[] = []
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      onProgress({
        stage: 'parsing',
        filesParsed: i,
        filesTotal: files.length,
        currentFile: file.name,
        message: `Parsing ${file.name}…`,
      })
      // Route decision — synchronous MIME check first; PDFs get an async
      // pdfjs text-check to differentiate text-bearing PDFs (fast path) from
      // scans (delegate to Image Builder). Non-image / non-PDF files skip
      // the round-trip entirely.
      const route = await routeForFile(file)
      if (route === 'text') {
        const result = await parseFile(file)
        textSections.push(...result.sections)
      } else {
        // 'image' or 'scanned' — POST to Image Builder /ocr. If Image
        // Builder is unreachable or the file fails, parseViaImageBuilder
        // throws; the surrounding try/catch bubbles it up as a build error
        // so the UI shows a helpful message rather than silently dropping
        // the file. Fallback (skip vs abort) is handled UPSTREAM in
        // BrowserCartBuilder before Build fires.
        onProgress({
          stage: 'parsing',
          filesParsed: i,
          filesTotal: files.length,
          currentFile: file.name,
          message: `OCR-ing ${file.name} via Image Builder…`,
        })
        const parsed = await parseViaImageBuilder(file)
        textSections.push(...parsed.textSections)
        graphicSections.push(...parsed.graphicSections)
        tableSections.push(...parsed.tableSections)
      }
    }
    onProgress({
      stage: 'parsing',
      filesParsed: files.length,
      filesTotal: files.length,
      sectionsTotal: textSections.length + graphicSections.length + tableSections.length,
      message: `Parsed ${textSections.length} text section${textSections.length === 1 ? '' : 's'}`
        + (graphicSections.length ? `, ${graphicSections.length} graphic${graphicSections.length === 1 ? '' : 's'}` : '')
        + (tableSections.length ? `, ${tableSections.length} table${tableSections.length === 1 ? '' : 's'}` : '')
        + '.',
    })

    if (textSections.length + graphicSections.length + tableSections.length === 0) {
      throw new Error(
        'No text content extracted. Are the files empty or in an unsupported format?'
      )
    }

    // ── Stage 2: chunk ─────────────────────────────────────────────────
    onProgress({
      stage: 'chunking',
      sectionsTotal: textSections.length,
      message: 'Splitting into overlapping chunks…',
    })
    // Only text sections go through the chunker's overlapping-window logic.
    // Graphic + table sections carry contentType metadata and bypass the
    // overlap step — each is a single pattern. Chunker checks contentType
    // internally and short-circuits for the non-'document' cases.
    const textChunks = chunkSections(textSections, options.chunkOptions)
    // Final section list: text chunks first (preserves the historical
    // ordering that carts have always had), then graphics, then tables.
    // Search doesn't care about order, but the Pattern-0 TOC + Edit Carts
    // drill-down surface groups by source, so keeping content types together
    // makes the built cart's per-file passage sequence more scannable in
    // both the drill-down UI and raw index dumps.
    const chunks: Section[] = [...textChunks, ...graphicSections, ...tableSections]

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
      onBackendChange: options.onBackendChange,
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
      sectionCount: textSections.length + graphicSections.length + tableSections.length,
      chunkCount: chunks.length,
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    onProgress({ stage: 'error', errorMessage: message, message })
    throw err
  }
}

/**
 * Build a cart from a list of typed passages — no file parsing involved.
 *
 * Used by the Edit Carts "New Cart" mode, where the user composes passages
 * directly in a text box instead of importing documents. Each passage is
 * pre-formed at chunk granularity, but we still run them through the
 * chunker so any oversized paste (>300 words) gets split rather than
 * landing as one giant chunk that retrieval can't grip.
 *
 * Same downstream stages as buildCartFromFiles — chunker, embedder,
 * writer. Skips parsing entirely. Resulting BuiltCart is interchangeable.
 */
export async function buildCartFromPassages(
  passages: string[],
  options: PipelineOptions,
): Promise<PipelineResult> {
  const onProgress = options.onProgress ?? (() => {})

  const cleaned = passages.map((p) => p.trim()).filter((p) => p.length > 0)
  if (cleaned.length === 0) {
    const err = new Error('No passages provided to pipeline')
    onProgress({ stage: 'error', errorMessage: err.message })
    throw err
  }

  const maxChunks = options.maxChunks ?? DEFAULT_MAX_CHUNKS_PER_BUILD

  try {
    // Wrap typed passages as Sections so the chunker + writer can consume
    // them like any other source. `source` carries provenance back to the
    // typed-in origin; `page` is the 1-indexed passage number.
    const sections: Section[] = cleaned.map((text, i) => ({
      text,
      page: i + 1,
      source: `typed-passage-${i + 1}`,
    }))

    onProgress({
      stage: 'parsing',
      filesParsed: 0,
      filesTotal: 0,
      sectionsTotal: sections.length,
      message: `Composing cart from ${sections.length} typed passage${sections.length === 1 ? '' : 's'}…`,
    })

    // ── Stage 2: chunk ─────────────────────────────────────────────────
    onProgress({
      stage: 'chunking',
      sectionsTotal: sections.length,
      message: 'Splitting oversized passages…',
    })
    const chunks = chunkSections(sections, options.chunkOptions)

    if (chunks.length > maxChunks) {
      throw new Error(
        `Build would produce ${chunks.length.toLocaleString()} chunks, ` +
        `exceeds ${maxChunks.toLocaleString()}-chunk cap. ` +
        `Trim some passages or raise maxChunks.`,
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
          message: `Embedding passages… ${completed}/${total}`,
        })
      },
      onBackendChange: options.onBackendChange,
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

    onProgress({
      stage: 'done',
      message: `Built ${cart.cartFilename}: ${chunks.length} chunks, ${embedResult.dim} dims, backend ${embedResult.backend}.`,
    })

    return {
      cart,
      fileCount: 0,
      sectionCount: sections.length,
      chunkCount: chunks.length,
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    onProgress({ stage: 'error', errorMessage: message, message })
    throw err
  }
}
