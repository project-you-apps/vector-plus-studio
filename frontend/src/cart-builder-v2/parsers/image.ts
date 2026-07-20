// Image (and scanned-PDF) parser — delegates OCR to Image Builder's `/ocr`
// endpoint (Day 2). Unlike the other parsers in this folder, this one calls
// out to a local exe over loopback and returns a mix of text sections
// (chunked normally) + graphic + table sections (each their own pattern).
//
// Not registered in the default parser registry — the pipeline explicitly
// routes here via MIME check / PDF classifier, because kicking off a
// multi-second OCR round-trip is a big enough hammer that we want the
// routing decision to be visible upstream (badges, fallback dialog) rather
// than buried in a `accept()` predicate.

import { ocrFile, type OcrOptions } from '../../api/imageBuilder'
import { ParseError, type Graphic, type ImageBuilderOcrResult, type Section, type Table } from '../types'
import { sourcePathForFile } from '../sourcePath'

// Convert a Docling-returned HTML table into a GFM markdown table so the
// passage viewer's react-markdown + remark-gfm renders it as an actual
// table (rows, columns, cell borders) — not a wall of piped characters.
// Also keeps cell values searchable via cosine similarity (e.g. `$1,463.40`
// from a receipt).
//
// We parse tags with regex — good enough for Docling's typically-clean
// `<table>` output. previous impl produced pipe-delimited
// flat text with no separator row, so remark-gfm parsed it as a paragraph.
// Mirror of the backend `_table_html_to_text` in api/cartbuilder/__init__.py.
const _ENT_MAP: Record<string, string> = {
  '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
  '&quot;': '"', '&#39;': "'",
}

function decodeEntities(s: string): string {
  for (const [k, v] of Object.entries(_ENT_MAP)) {
    s = s.replace(new RegExp(k, 'gi'), v)
  }
  return s
}

function cleanCell(raw: string): string {
  // Strip inner tags, decode entities, collapse whitespace, escape pipes.
  return decodeEntities(raw.replace(/<[^>]+>/g, ''))
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\|/g, '\\|')
}

export function tableHtmlToText(html: string): string {
  if (!html) return ''
  const rowRe = /<tr[^>]*>([\s\S]*?)<\/tr>/gi
  const cellRe = /<(th|td)[^>]*>([\s\S]*?)<\/\1>/gi

  const rows: string[][] = []
  let rowMatch: RegExpExecArray | null
  while ((rowMatch = rowRe.exec(html)) !== null) {
    const rowHtml = rowMatch[1]
    const cells: string[] = []
    let cellMatch: RegExpExecArray | null
    while ((cellMatch = cellRe.exec(rowHtml)) !== null) {
      cells.push(cleanCell(cellMatch[2]))
    }
    if (cells.length > 0) rows.push(cells)
  }

  if (rows.length === 0) return ''

  const ncols = rows.reduce((m, r) => Math.max(m, r.length), 0)
  const padded = rows.map(r => r.concat(Array(ncols - r.length).fill('')))
  const fmt = (cells: string[]) => '| ' + cells.join(' | ') + ' |'
  const separator = '| ' + Array(ncols).fill('---').join(' | ') + ' |'

  return [fmt(padded[0]), separator, ...padded.slice(1).map(fmt)].join('\n')
}

// Cart Builder prefixes uploads with an 8-char hex hash + underscore for
// disk-collision avoidance. Strip it from user-facing caption text.
const HASH_PREFIX_RE = /^[0-9a-f]{8}_/
function displayName(filename: string): string {
  return filename.replace(HASH_PREFIX_RE, '')
}

// Convert Image Builder graphics into Sections. Each graphic becomes ONE
// pattern in the cart; the caption is the passage text (searchable). If the
// caption is empty, we fall back to a placeholder so the passage isn't
// literally empty (and can still be embedded — the embedder chokes on empty
// strings). Real image thumbnails + zoom UI land Day 3+.
export function graphicsToSections(
  graphics: Graphic[],
  sourceName: string,
): Section[] {
  const displaySource = displayName(sourceName)
  return graphics.map((g, i) => {
    const caption = (g.caption ?? '').trim()
    // Placeholder text ensures cosine similarity has something to embed
    // even when Docling didn't detect a caption. Keeping it explicit +
    // scannable makes debug-search-by-eyeball work — "graphic 3 of
    // jfc-invoice.jpg" is a legitimate query on a bespoke tag.
    const passageText = caption
      || `Graphic ${i + 1} of ${displaySource} Page ${g.page || 1}`
    return {
      text: passageText,
      page: g.page || 1,
      source: sourceName,
      contentType: 'graphic',
      imageB64: g.image_b64 || '',
      bbox: Array.isArray(g.bbox) ? g.bbox.slice() : [],
      caption,
    }
  })
}

// Convert Image Builder tables into Sections. Each table becomes ONE pattern;
// the passage text is the plaintext extraction of the HTML so search finds
// cell values. Original HTML preserved for future rich rendering.
export function tablesToSections(
  tables: Table[],
  sourceName: string,
): Section[] {
  const displaySource = displayName(sourceName)
  return tables.map((t, i) => {
    const plainText = tableHtmlToText(t.html || '').trim()
    const passageText = plainText || `Table ${i + 1} of ${displaySource} Page ${t.page || 1}`
    return {
      text: passageText,
      page: t.page || 1,
      source: sourceName,
      contentType: 'table',
      html: t.html || '',
      bbox: Array.isArray(t.bbox) ? t.bbox.slice() : [],
    }
  })
}

// Full pipeline for a single image or scanned-PDF file. POSTs the file to
// Image Builder /ocr, then normalizes the response into a `{textSections,
// graphicSections, tableSections}` triple that the caller can thread into
// the chunker (for the text) and the writer (for the graphic + table
// patterns).
export interface ImageParseOutput {
  textSections: Section[]      // markdown chunked by the caller
  graphicSections: Section[]   // one section per Docling graphic
  tableSections: Section[]     // one section per Docling table
  raw: ImageBuilderOcrResult   // pass-through for optional telemetry / logging
}

export async function parseViaImageBuilder(
  file: File,
  options: OcrOptions = {},
): Promise<ImageParseOutput> {
  let result: ImageBuilderOcrResult
  try {
    result = await ocrFile(file, options)
  } catch (err) {
    throw new ParseError(
      `Image Builder OCR failed: ${err instanceof Error ? err.message : String(err)}`,
      file.name,
      err,
    )
  }

  const markdown = (result.markdown || '').trim()
  const textSections: Section[] = markdown
    ? [{ text: markdown, page: 1, source: sourcePathForFile(file) }]
    : []

  return {
    textSections,
    graphicSections: graphicsToSections(result.graphics, file.name),
    tableSections: tablesToSections(result.tables, file.name),
    raw: result,
  }
}

// MIME types Image Builder can OCR. Kept aligned with SUPPORTED_FORMATS in
// image-builder/main.py (pdf, png, jpg, jpeg, tiff, webp, bmp, heic). PDFs
// aren't included here — the pipeline decides via classifyPdf() whether a
// given PDF goes through the text parser or Image Builder.
const IMAGE_MIMES = new Set([
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/heic',
  'image/heif',
  'image/tiff',
  'image/webp',
  'image/bmp',
])

const IMAGE_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.webp', '.bmp',
])

// True when the file is an image type that always routes to Image Builder,
// regardless of PDF text-check. Used by both the routing pipeline and the
// per-file badge renderer so the two stay in sync (spec Q4: images are
// always amber).
export function isImageFile(file: File): boolean {
  if (file.type && IMAGE_MIMES.has(file.type.toLowerCase())) return true
  const name = file.name.toLowerCase()
  const dot = name.lastIndexOf('.')
  if (dot === -1) return false
  return IMAGE_EXTENSIONS.has(name.slice(dot))
}
