import * as pdfjsLib from 'pdfjs-dist'
// Vite resolves ?url to a hashed asset URL at build time. If the worker file
// path changes in a future pdfjs-dist version, update here.
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import type { Parser, Section } from '../types'
import { ParseError } from '../types'

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl

export const pdfParser: Parser = {
  name: 'pdf',
  accept(file: File): boolean {
    return (
      file.name.toLowerCase().endsWith('.pdf') ||
      file.type === 'application/pdf'
    )
  },
  async parse(file: File): Promise<Section[]> {
    const buffer = await file.arrayBuffer()
    let pdf
    try {
      pdf = await pdfjsLib.getDocument({ data: buffer }).promise
    } catch (err) {
      throw new ParseError('Failed to open PDF', file.name, err)
    }
    const sections: Section[] = []
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i)
      const content = await page.getTextContent()
      const text = content.items
        .map((it) => ('str' in it ? it.str : ''))
        .join(' ')
        .trim()
      if (text) {
        sections.push({ text, page: i, source: file.name })
      }
    }
    return sections
  },
}

// PDF classification threshold — sum of extractable text length across the
// first N pages. 500 chars is the Day 2 spec default: text PDFs almost always
// exceed this on page 1 alone; scanned image PDFs return near-zero unless
// there's a stray extracted logo/watermark. Threshold applies AFTER trimming
// whitespace so pages with only page numbers / running headers don't tip
// borderline scans into the text path.
export const PDF_CLASSIFY_TEXT_THRESHOLD = 500
// Andy 2026-07-05 PM (Grant's pitch deck): expanded from 3 → 15 pages
// because the deck was clean on pages 1-3 and only had broken ToUnicode
// fonts on pages 4-7 — sampling only the head missed the corruption.
// 15 caps classify cost on very long documents.
export const PDF_CLASSIFY_MAX_PAGES = 15
// Andy 2026-07-05: PDFs with broken font ToUnicode maps return LOTS of
// characters but most are Private Use Area / replacement / non-Latin
// garbage that produces unreadable ingest. Two-level check:
//   - Per-page: if ANY sampled page has substantial content (>50 chars)
//     but < PAGE_READABLE_THRESHOLD readable, route to Image Builder.
//     One bad page = broken font used elsewhere in the doc.
//   - Aggregate fallback: overall readable fraction < threshold, same.
// Mirrors backend api/cartbuilder/parsers.py.
export const PDF_CLASSIFY_READABLE_THRESHOLD = 0.6
export const PDF_CLASSIFY_PAGE_READABLE_THRESHOLD = 0.6
export const PDF_CLASSIFY_PAGE_MIN_CHARS = 50

function isReadableChar(c: string): boolean {
  const cp = c.charCodeAt(0)
  if (cp >= 0x20 && cp <= 0x7E) return true
  if (c === '\n' || c === '\r' || c === '\t') return true
  if (cp >= 0xA0 && cp <= 0x24F) return true
  return false
}

function readableCharCount(s: string): number {
  let n = 0
  for (const c of s) if (isReadableChar(c)) n++
  return n
}

/**
 * Classify a PDF as text-bearing or scanned-image.
 *
 * Runs pdfjs' `getTextContent()` across the first PDF_CLASSIFY_MAX_PAGES
 * pages and sums both the extractable text length and the count of readable
 * ASCII/Latin characters. Returns "text" only when total >
 * PDF_CLASSIFY_TEXT_THRESHOLD AND readable-fraction >=
 * PDF_CLASSIFY_READABLE_THRESHOLD. Otherwise → "scanned" (route to Image
 * Builder /ocr). Mirrors the server-side classify_pdf() in
 * api/cartbuilder/parsers.py so both build paths produce the same routing
 * decision on the same file (Day 2 golden-path invariant).
 *
 * Failure mode: if pdfjs can't open the file at all, we return "scanned"
 * on the theory that Image Builder / Docling is more likely to salvage a
 * malformed PDF than the browser pipeline is. The Image Builder call may
 * still fail, at which point the user sees the standard fallback.
 */
export async function classifyPdf(file: File): Promise<'text' | 'scanned'> {
  const buffer = await file.arrayBuffer()
  let pdf
  try {
    pdf = await pdfjsLib.getDocument({ data: buffer }).promise
  } catch {
    // Pdfjs couldn't parse — treat as scanned so we try OCR rather than
    // failing outright. Image Builder may still bounce it, but that's a
    // Docling-side decision, not ours.
    return 'scanned'
  }
  const pagesToCheck = Math.min(pdf.numPages, PDF_CLASSIFY_MAX_PAGES)
  let totalChars = 0
  let readableChars = 0
  let corruptPageFound = false
  let corruptPageIdx = -1
  let corruptPageFraction = 0
  for (let i = 1; i <= pagesToCheck; i++) {
    let pageChars = 0
    let pageReadable = 0
    try {
      const page = await pdf.getPage(i)
      const content = await page.getTextContent()
      for (const item of content.items) {
        if ('str' in item) {
          const s = item.str.trim()
          pageChars += s.length
          pageReadable += readableCharCount(s)
        }
      }
    } catch {
      // Skip page on error, continue counting; a partial score is still
      // useful — a truly scanned PDF will still land under threshold.
      continue
    }
    totalChars += pageChars
    readableChars += pageReadable
    // Per-page corruption check — substantial content but low readable
    // fraction signals a broken ToUnicode font. One bad page taints the
    // whole ingest because the same font likely appears elsewhere.
    if (pageChars >= PDF_CLASSIFY_PAGE_MIN_CHARS && !corruptPageFound) {
      const pageFraction = pageReadable / pageChars
      if (pageFraction < PDF_CLASSIFY_PAGE_READABLE_THRESHOLD) {
        corruptPageFound = true
        corruptPageIdx = i
        corruptPageFraction = pageFraction
      }
    }
  }
  const readableFraction = totalChars > 0 ? readableChars / totalChars : 0
  // Diagnostic (Andy 2026-07-05) — remove after Grant's deck routing tunes correctly.
  // eslint-disable-next-line no-console
  console.log(
    `[classifyPdf] ${file.name}: totalChars=${totalChars}, readableChars=${readableChars}, ` +
    `readableFraction=${readableFraction.toFixed(3)}, pagesSampled=${pagesToCheck}, ` +
    `corruptPage=${corruptPageFound ? `${corruptPageIdx} (fraction ${corruptPageFraction.toFixed(3)})` : 'none'}`
  )
  if (totalChars <= PDF_CLASSIFY_TEXT_THRESHOLD) return 'scanned'
  if (corruptPageFound) return 'scanned'
  if (readableFraction < PDF_CLASSIFY_READABLE_THRESHOLD) return 'scanned'
  return 'text'
}
