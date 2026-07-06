import type { Section } from './types'

export interface ChunkOptions {
  chunkSize?: number
  overlap?: number
  /**
   * Hard cap on a chunk's character length. Any chunk longer than this after
   * word-based chunking is split further at whitespace boundaries (see
   * `splitByChars`). Default 6000.
   *
   * ── Why 6000? ─────────────────────────────────────────────────────────
   * Nomic-embed-v1.5 (the embedder we ship in-browser) uses a self-attention
   * layer whose intermediate buffer is roughly `seq_len² × hidden_dim × 4`
   * bytes in float32. WebGPU imposes a hard 2 GiB per-buffer cap
   * (`maxBufferSize = 2_147_483_648`). Empirically a ~7000-token input
   * produces the attention buffer at ~2.45 GiB, which trips:
   *
   *   "WebGPU validation failed. Buffer size (2448240000) exceeds the max
   *    buffer size limit (2147483648)"
   *
   * transformers.js caches its InferenceSession, so we can't mid-stream
   * failover to WASM once the WebGPU session is bound (Patch 7b). The only
   * reliable fix is to keep chunks small enough that the attention buffer
   * never blows the cap. 6000 chars → ~1500 tokens on prose (chars/token
   * ratio ≈ 4 for English), well below the ~4500-token ceiling where the
   * attention buffer starts crowding the 2 GiB limit, and with margin for
   * dense/technical text where the ratio drops toward 3.
   */
  maxChars?: number
}

const DEFAULT_CHUNK_SIZE = 300
const DEFAULT_OVERLAP = 50
const DEFAULT_MAX_CHARS = 6000

/**
 * Split a single string into pieces of at most `maxChars` characters.
 *
 * Strategy: walk left-to-right, cutting at the last whitespace at or before
 * the cap. If no whitespace lives in the second half of the current window
 * (i.e. we'd throw away most of the window hunting for a boundary), cut
 * hard at the cap position and accept that we've split mid-word — that only
 * happens on pathological inputs (giant URLs, one-line minified blobs).
 *
 * Preserves nothing else — the caller is responsible for stitching metadata
 * back onto each piece.
 */
function splitByChars(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) return [text]
  const pieces: string[] = []
  let cursor = 0
  while (cursor < text.length) {
    const remaining = text.length - cursor
    if (remaining <= maxChars) {
      const tail = text.slice(cursor).trim()
      if (tail) pieces.push(tail)
      break
    }
    const windowEnd = cursor + maxChars
    // Search for the last whitespace at or before windowEnd, but only within
    // the second half of the window — otherwise we'd shrink the chunk too
    // aggressively chasing a boundary that isn't there.
    const halfway = cursor + Math.floor(maxChars / 2)
    let splitAt = -1
    for (let i = windowEnd; i > halfway; i--) {
      if (/\s/.test(text[i])) {
        splitAt = i
        break
      }
    }
    if (splitAt === -1) splitAt = windowEnd
    const piece = text.slice(cursor, splitAt).trim()
    if (piece) pieces.push(piece)
    // Skip the whitespace we cut on so it doesn't lead the next piece.
    cursor = splitAt
    while (cursor < text.length && /\s/.test(text[cursor])) cursor++
  }
  return pieces
}

/**
 * Split sections into overlapping word-based chunks. Defaults match
 * cart-builder/cart-builder/parsers.py:chunk_texts (300 words, 50 overlap)
 * so server-built and browser-built carts stay output-compatible.
 *
 * Sections shorter than chunkSize pass through unchanged (no `part` field
 * added), matching Python behavior — UNLESS their character length exceeds
 * `maxChars`, in which case they get char-split and each piece receives a
 * `part` index.
 *
 * After word-based chunking, every produced chunk is measured against
 * `maxChars` (default 6000) and further split at whitespace boundaries if
 * it exceeds the cap. This guards the WebGPU embedder against the 2 GiB
 * per-buffer limit that Nomic v1.5's attention matrix hits on ~7000-token
 * inputs — see the `maxChars` docstring on `ChunkOptions` for the math.
 * `part` is renumbered across the char-split so downstream consumers see a
 * clean 0, 1, 2, … sequence.
 */
export function chunkSections(
  sections: Section[],
  options: ChunkOptions = {}
): Section[] {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE
  const overlap = options.overlap ?? DEFAULT_OVERLAP
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS
  if (chunkSize <= overlap) {
    throw new Error(
      `chunkSize (${chunkSize}) must exceed overlap (${overlap})`
    )
  }
  if (maxChars <= 0) {
    throw new Error(`maxChars (${maxChars}) must be positive`)
  }
  const stride = chunkSize - overlap
  const chunks: Section[] = []

  for (const section of sections) {
    // Graphic + table sections (Day 2: Image Builder delegation) are opaque
    // one-pattern-per-item units — never chunked. Their `text` is the caption
    // (graphic) or plain-text extraction of the HTML (table); we still cap
    // char length below via splitByChars so the embedder doesn't get an
    // oversized input, but each item passes through as a single chunk
    // (no overlapping windows) so it maps 1-1 to a pattern in the cart.
    if (section.contentType === 'graphic' || section.contentType === 'table') {
      if (section.text.length <= maxChars) {
        chunks.push(section)
      } else {
        // Extremely rare — Docling captions and even table plaintext very
        // rarely blow 6000 chars. If they do, split-and-keep the metadata on
        // every piece so the writer still emits one per_pattern_meta row per
        // pattern (the graphic/table gets duplicated across sub-patterns,
        // which is preferable to losing the tail).
        const pieces = section.text.split(/\s+/).length > 1
          ? section.text.length > maxChars
            ? [section.text.slice(0, maxChars)]  // simple truncation for now
            : [section.text]
          : [section.text.slice(0, maxChars)]
        for (const piece of pieces) {
          chunks.push({ ...section, text: piece.trim() })
        }
      }
      continue
    }

    // Collect the word-chunked pieces first, without the final `part`
    // renumber, so we can apply the char-cap in a second pass and renumber
    // `part` in one consistent sequence across the whole section.
    //
    // Line-aware chunker: chunk boundaries fall between lines, never
    // mid-line, so markdown structure (tables, lists, headings, paragraph
    // breaks) survives intact for the passage viewer's markdown renderer.
    // Andy 2026-07-05: previous split(/\s+/) + join(' ') implementation
    // destroyed all newlines, turning Docling OCR into a single wall of
    // piped text — mirror of the backend chunk_texts fix.
    const preCharSplit: Section[] = []
    let wasWordSplit = false
    const words = section.text.split(/\s+/).filter((w) => w.length > 0)
    if (words.length <= chunkSize) {
      // Pass-through path — no `part` yet. If the char-cap forces a split
      // below, we'll add `part` at that point.
      preCharSplit.push(section)
    } else {
      wasWordSplit = true
      const lines = section.text.split('\n')
      const lineWc = lines.map((l) => l.split(/\s+/).filter((w) => w.length > 0).length)
      const n = lines.length
      let i = 0
      let part = 0
      while (i < n) {
        let j = i
        let budget = 0
        while (j < n && (budget === 0 || budget + lineWc[j] <= chunkSize)) {
          budget += lineWc[j]
          j += 1
        }
        const chunkText = lines.slice(i, j).join('\n').trim()
        if (chunkText) {
          preCharSplit.push({
            text: chunkText,
            page: section.page,
            source: section.source,
            part,
          })
          part += 1
        }
        if (j >= n) break
        // Walk backward from j to leave `overlap` words as prefix for next
        // chunk. Guard: always advance at least one line.
        let back = j
        let overlapWc = 0
        while (back > i + 1 && overlapWc < overlap) {
          back -= 1
          overlapWc += lineWc[back]
        }
        i = back
      }
    }
    // Reference stride to keep param-consumption stable (line-aware pass
    // computes its own overlap window rather than a fixed word stride).
    void stride

    // Second pass: enforce maxChars. Any chunk that exceeds the cap gets
    // split; `part` is renumbered across the whole section so the final
    // sequence is 0, 1, 2, … regardless of which sub-splits fired.
    const finalPieces: Section[] = []
    let anyCharSplit = false
    let biggestOffender = 0
    let subChunkCount = 0
    for (const piece of preCharSplit) {
      if (piece.text.length <= maxChars) {
        finalPieces.push(piece)
        continue
      }
      anyCharSplit = true
      if (piece.text.length > biggestOffender) biggestOffender = piece.text.length
      const sub = splitByChars(piece.text, maxChars)
      subChunkCount += sub.length
      for (const subText of sub) {
        finalPieces.push({
          text: subText,
          page: piece.page,
          source: piece.source,
          // `part` is a placeholder here — renumbered below.
          part: 0,
        })
      }
    }

    if (anyCharSplit) {
      // Diagnostic only — one warning per section, not per sub-chunk, so
      // large builds don't flood the console.
      // eslint-disable-next-line no-console
      console.warn(
        `[chunker] section chunk (${biggestOffender} chars) exceeded ` +
        `maxChars=${maxChars}; split into ${subChunkCount} sub-chunks`
      )
    }

    if (anyCharSplit || wasWordSplit) {
      // Renumber `part` cleanly across the section's final chunk list so
      // downstream consumers always see 0, 1, 2, … with no gaps.
      for (let i = 0; i < finalPieces.length; i++) {
        finalPieces[i] = { ...finalPieces[i], part: i }
      }
      chunks.push(...finalPieces)
    } else {
      // Pure pass-through: single chunk, no `part`, unchanged from today's
      // behavior. Preserves cart byte-parity for sections under both caps.
      chunks.push(...finalPieces)
    }
  }
  return chunks
}
