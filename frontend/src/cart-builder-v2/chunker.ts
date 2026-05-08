import type { Section } from './types'

export interface ChunkOptions {
  chunkSize?: number
  overlap?: number
}

const DEFAULT_CHUNK_SIZE = 300
const DEFAULT_OVERLAP = 50

/**
 * Split sections into overlapping word-based chunks. Defaults match
 * cart-builder/cart-builder/parsers.py:chunk_texts (300 words, 50 overlap)
 * so server-built and browser-built carts stay output-compatible.
 *
 * Sections shorter than chunkSize pass through unchanged (no `part` field
 * added), matching Python behavior.
 */
export function chunkSections(
  sections: Section[],
  options: ChunkOptions = {}
): Section[] {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE
  const overlap = options.overlap ?? DEFAULT_OVERLAP
  if (chunkSize <= overlap) {
    throw new Error(
      `chunkSize (${chunkSize}) must exceed overlap (${overlap})`
    )
  }
  const stride = chunkSize - overlap
  const chunks: Section[] = []

  for (const section of sections) {
    const words = section.text.split(/\s+/).filter((w) => w.length > 0)
    if (words.length <= chunkSize) {
      chunks.push(section)
      continue
    }
    let start = 0
    let part = 0
    while (start < words.length) {
      const end = start + chunkSize
      const chunkText = words.slice(start, end).join(' ').trim()
      if (chunkText) {
        chunks.push({
          text: chunkText,
          page: section.page,
          source: section.source,
          part,
        })
        part += 1
      }
      start += stride
    }
  }
  return chunks
}
