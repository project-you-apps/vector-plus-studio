import JSZip from 'jszip'
import type { Section } from '../types'
import { NOMIC_DIM } from '../embedder/embed'
import { type HippocampusOptions, packHippocampus } from './hippocampus'
import { type CartManifest, buildManifest } from './manifest'
import { dumpFloat32, dumpUint8, dumpUnicode } from './npy'
import {
  type CartPermissionsPayload,
  type CartPermissionsSpec,
  buildPermissions,
} from './permissions'

// Rich Pattern-0 metadata baked into the NPZ at build time. Mirrors the
// cart-level fields set server-side by api/cartbuilder/builder.py so the
// Pattern-0 TOC panel (Andy 2026-07-02) surfaces the same shape regardless
// of whether the cart was built via the browser pipeline or the VPS /build
// endpoint. Empty strings become generic-fallback text; empty tags stay
// empty. `owner` propagates to the top-level pattern0 record.
export interface Pattern0Meta {
  description?: string
  agent_briefing?: string
  owner?: string
  tags?: string[]
  creator?: string
}

export interface BuildCartOptions {
  cartName: string
  hippocampus?: HippocampusOptions
  permissions?: CartPermissionsSpec
  pattern0Meta?: Pattern0Meta
  // Day 2 — Image Builder integration. Extend the cart with per-pattern
  // metadata + optional graphic + table pattern data. All three are optional
  // and default-empty; carts built from purely text sources (Day 1 shape)
  // continue to write the same NPZ layout they always did, just with a new
  // per_pattern_meta.npy sidecar attached.
  //
  // NOTE: graphics + tables are already threaded through the sections array
  // by the pipeline (they arrive as Sections with contentType='graphic' or
  // 'table' and were embedded like any other pattern). The writer just needs
  // to read section.contentType + optional per-item metadata (image_b64,
  // bbox, caption, html) and emit the right per_pattern_meta shape.
}

// Kept in sync with GENERIC_DESCRIPTION / GENERIC_AGENT_BRIEFING in
// api/cartbuilder/__init__.py so browser-built and server-built carts show
// identical fallback text in the Pattern-0 TOC panel.
const GENERIC_DESCRIPTION = 'A data or information cartridge for easy information access.'
const GENERIC_AGENT_BRIEFING =
  'This cart contains reference material. When answering questions, ' +
  'search it for relevant passages and cite them by source. Do not ' +
  'invent content that isn\'t present in the cart.'

export interface BuiltCart {
  /** ZIP container with embeddings.npy + hippocampus.npy + passages.json + compressed_texts.json. */
  cartBlob: Blob
  /** Standalone .cart_manifest.json (mcp-v3 fingerprint). */
  manifestBlob: Blob
  /** Standalone .permissions.json (Step 2a). */
  permissionsBlob: Blob
  manifest: CartManifest
  permissions: CartPermissionsPayload
  cartFilename: string
  manifestFilename: string
  permissionsFilename: string
}

/**
 * Bundle embeddings + sections into a deployable cart trio.
 *
 * The .cart.npz is a ZIP container holding:
 *   embeddings.npy        — Float32Array shape [N, 768], NPY float32
 *   hippocampus.npy       — Uint8Array shape [N, 64], NPY uint8
 *   passages.npy          — string[] shape [N], NPY unicode (<U<maxlen>)
 *   compressed_texts.npy  — string[] shape [N], NPY unicode (no compression in v1)
 *
 * passages and compressed_texts use NPY unicode (`<U<n>`) rather than the
 * pickled object-array Python defaults to. numpy loads `<U` arrays without
 * `allow_pickle=True` and behaves identically for read purposes (trailing
 * nulls stripped on access). Net result: browser-built carts mount on the
 * existing membot server with no server-side changes needed.
 */
export async function buildCart(
  embeddings: Float32Array,
  sections: Section[],
  options: BuildCartOptions
): Promise<BuiltCart> {
  const count = sections.length
  if (count === 0) {
    throw new Error('Cannot build cart from zero sections')
  }
  if (embeddings.length !== count * NOMIC_DIM) {
    throw new Error(
      `Embeddings size mismatch: ${embeddings.length} (expected ${count * NOMIC_DIM} for ${count} sections × ${NOMIC_DIM} dims)`
    )
  }

  const passages = sections.map((s) => s.text)
  const compressed_texts = passages // v1: no compression
  // Provenance v1 sidecar — per-pattern source filename so result cards can
  // display "this came from foo.py" without needing to reverse the FNV-1a
  // source_hash in the hippocampus row. v2 will replace this with a proper
  // strings table + source_idx field in h-row; tracked in
  // CC_cart-provenance-schema_2026-06-15 as a v2 pilot blocker. See ALSO
  // the "Provenance: v1 sidecar (alpha)" badge in the UI.
  const source_paths = sections.map((s) => s.source)
  const hippocampus = packHippocampus(sections, options.hippocampus)
  const manifest = await buildManifest(embeddings, count, NOMIC_DIM)
  const permissions = buildPermissions(options.permissions)

  // Day 2 — per_pattern_meta sidecar. Matches the shape backend builder.py
  // writes for server-built carts so a mounted cart's per-pattern metadata
  // looks identical regardless of build path. Fields:
  //   v          — schema version (1)
  //   content_type — "document" (default text), "graphic", or "table"
  //   source     — original filename
  //   page       — 1-indexed page, or null for whole-doc extractions
  //   chunk / chunks — position within section (0-indexed; total sections)
  //   tags       — reserved for future per-pattern tag propagation; empty [] today
  //   created_at — unix time (seconds, float) for consistency w/ backend
  //   tombstone  — always false at build time; edited via Edit Carts later
  //   ...content-type specific extras: bbox, caption, image_b64 (graphic);
  //     bbox + html (table). Preserved for future thumbnail + zoom UI.
  const buildTs = Date.now() / 1000
  // Per-source chunk counters — advance across the section list so each
  // pattern's `chunk` reflects its position within its parent file. Text
  // patterns typically get 0..N sequences; graphic/table patterns advance
  // the same source's counter so cross-content indexing stays coherent.
  const sourceCursor = new Map<string, { idx: number; total: number }>()
  for (const s of sections) {
    const entry = sourceCursor.get(s.source) ?? { idx: 0, total: 0 }
    entry.total++
    sourceCursor.set(s.source, entry)
  }
  const sourceIdx = new Map<string, number>()
  const per_pattern_meta = sections.map((s) => {
    const idx = sourceIdx.get(s.source) ?? 0
    const total = sourceCursor.get(s.source)?.total ?? 1
    sourceIdx.set(s.source, idx + 1)
    const ct = s.contentType ?? 'document'
    // Base record — matches backend api/cartbuilder/builder.py per_pattern_meta.
    const record: Record<string, unknown> = {
      v: 1,
      content_type: ct,
      source: s.source,
      page: s.page ?? null,
      chunk: idx,
      chunks: total,
      tags: ct === 'graphic' ? ['graphic'] : ct === 'table' ? ['table'] : [],
      created_at: buildTs,
      tombstone: false,
      // Legacy fields preserved for backend-parity — backend builder writes
      // these even when empty so downstream readers can rely on them existing.
      owner: '',
      description: '',
    }
    if (ct === 'graphic') {
      record.caption = s.caption ?? ''
      record.image_b64 = s.imageB64 ?? ''
      record.bbox = Array.isArray(s.bbox) ? s.bbox : []
    } else if (ct === 'table') {
      record.html = s.html ?? ''
      record.bbox = Array.isArray(s.bbox) ? s.bbox : []
    }
    return record
  })
  // Total counts for the Pattern-0 header — surfaced in Pattern0TocPanel.
  const graphicCount = per_pattern_meta.reduce((n, r) => n + (r.content_type === 'graphic' ? 1 : 0), 0)
  const tableCount = per_pattern_meta.reduce((n, r) => n + (r.content_type === 'table' ? 1 : 0), 0)

  // Build the pattern0_data payload with the same shape as
  // api/cartbuilder/builder.py: cart-level metadata + per-file list. The
  // Pattern-0 TOC panel reader (api/main.py:_parse_pattern0_from_npz)
  // consumes this JSON directly when the cart is mounted on the VPS.
  const meta = options.pattern0Meta ?? {}
  const uniqueSources = new Map<string, number>()
  for (const s of sections) {
    uniqueSources.set(s.source, (uniqueSources.get(s.source) ?? 0) + 1)
  }
  const pattern0_data = {
    cart_name: options.cartName,
    creator: meta.creator || 'Cart Builder (browser)',
    created_at: new Date().toISOString(),
    file_count: uniqueSources.size,
    total_chunks: count,
    embedding_model: 'nomic-ai/nomic-embed-text-v1.5',
    embedding_dim: NOMIC_DIM,
    files: Array.from(uniqueSources.entries()).map(([name, chunks]) => ({
      name,
      owner: meta.owner ?? '',
      description: '',
      tags: [],
      chunks,
    })),
    description: (meta.description ?? '').trim() || GENERIC_DESCRIPTION,
    agent_briefing: (meta.agent_briefing ?? '').trim() || GENERIC_AGENT_BRIEFING,
    owner: meta.owner ?? '',
    tags: meta.tags ?? [],
    // Day 2 — Image Builder integration surfaces these in the Pattern-0 TOC
    // panel header ("N graphics + M tables"). Zero when no Image-Builder-routed
    // files landed in this build; the panel hides the badge on 0s.
    graphic_count: graphicCount,
    table_count: tableCount,
  }
  const pattern0Json = JSON.stringify(pattern0_data)
  // per_pattern_meta.npy — Day 2 sidecar. Single-element unicode NPY holding
  // a JSON payload with a per-pattern record for every pattern in `passages`.
  // Consumed by Day 3+ UI (small-PNG thumbnails on graphic patterns, "table"
  // badge on table patterns). See backend builder.py for the byte-parity
  // reference; server-built carts write the same shape.
  const perPatternMetaJson = JSON.stringify(per_pattern_meta)

  const zip = new JSZip()
  zip.file('embeddings.npy', dumpFloat32(embeddings, [count, NOMIC_DIM]))
  zip.file('hippocampus.npy', dumpUint8(hippocampus, [count, 64]))
  zip.file('passages.npy', dumpUnicode(passages, [count]))
  zip.file('compressed_texts.npy', dumpUnicode(compressed_texts, [count]))
  zip.file('source_paths.npy', dumpUnicode(source_paths, [count]))
  // pattern0.npy — single-element unicode array of the JSON payload. Matches
  // the Cart Builder GUI schema so api/cart/pattern-0 reads it uniformly.
  zip.file('pattern0.npy', dumpUnicode([pattern0Json], [1]))
  // per_pattern_meta.npy — Day 2. Same JSON-in-unicode-array shape as
  // pattern0.npy for consistency. Backend variant uses np.savez_compressed
  // which stores it as an object array; both formats round-trip through
  // np.load(...) without allow_pickle=True (unicode NPY) or with it (object).
  zip.file('per_pattern_meta.npy', dumpUnicode([perPatternMetaJson], [1]))

  const cartBlob = await zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  })
  const manifestBlob = new Blob([JSON.stringify(manifest, null, 2)], {
    type: 'application/json',
  })
  const permissionsBlob = new Blob([JSON.stringify(permissions, null, 2)], {
    type: 'application/json',
  })

  return {
    cartBlob,
    manifestBlob,
    permissionsBlob,
    manifest,
    permissions,
    cartFilename: `${options.cartName}.cart.npz`,
    manifestFilename: `${options.cartName}.cart_manifest.json`,
    permissionsFilename: `${options.cartName}.permissions.json`,
  }
}

/**
 * Save the three cart artifacts. Uses the File System Access API
 * (`showDirectoryPicker`) when available so the user can pick a real
 * destination folder instead of being forced into the browser's
 * Downloads/ default — Chrome/Edge/Opera 86+ support this. Firefox and
 * Safari fall back to the legacy three-<a download> approach.
 *
 * Resolves on completion (or user cancel). Rejects only on unexpected
 * filesystem-API failure; caller should still treat a rejection as
 * non-fatal since the user can always re-trigger the save.
 */
export async function downloadBuiltCart(cart: BuiltCart): Promise<void> {
  type DirPickerOpts = { mode?: 'read' | 'readwrite' }
  type DirPicker = (opts?: DirPickerOpts) => Promise<FileSystemDirectoryHandle>
  const pickerFn = (window as unknown as { showDirectoryPicker?: DirPicker }).showDirectoryPicker
  if (typeof pickerFn === 'function') {
    try {
      const dir = await pickerFn({ mode: 'readwrite' })
      await writeBlobToDir(dir, cart.cartFilename, cart.cartBlob)
      await writeBlobToDir(dir, cart.manifestFilename, cart.manifestBlob)
      await writeBlobToDir(dir, cart.permissionsFilename, cart.permissionsBlob)
      return
    } catch (err) {
      // User cancelled the picker — bail silently, no fallback (legacy
      // download would land files they didn't ask for).
      if ((err as DOMException)?.name === 'AbortError') return
      // Permission denied / quota / other API failure — fall through to
      // the legacy path so the user still gets the cart somewhere.
      console.warn('showDirectoryPicker failed, falling back to <a download>', err)
    }
  }
  triggerDownload(cart.cartBlob, cart.cartFilename)
  triggerDownload(cart.manifestBlob, cart.manifestFilename)
  triggerDownload(cart.permissionsBlob, cart.permissionsFilename)
}

async function writeBlobToDir(
  dir: FileSystemDirectoryHandle,
  name: string,
  blob: Blob,
): Promise<void> {
  const fileHandle = await dir.getFileHandle(name, { create: true })
  const writable = await fileHandle.createWritable()
  await writable.write(blob)
  await writable.close()
}

/**
 * Save a built cart directly to a previously-picked directory handle.
 * Used by the New Cart flow in Edit Carts, where the user picks the
 * destination folder BEFORE composing passages — so there's no need to
 * re-prompt with a picker at save time.
 */
export async function saveBuiltCartToDirectory(
  cart: BuiltCart,
  dir: FileSystemDirectoryHandle,
): Promise<void> {
  await writeBlobToDir(dir, cart.cartFilename, cart.cartBlob)
  await writeBlobToDir(dir, cart.manifestFilename, cart.manifestBlob)
  await writeBlobToDir(dir, cart.permissionsFilename, cart.permissionsBlob)
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // Revoke after a tick so the browser has time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
