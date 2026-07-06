// Registry-style parser dispatch. Plug-and-play extensible — `registerParser`
// lets next-week JSON / mbox / HTML / source-schema adapters slot in without
// touching the chunker or downstream pipeline.

import type { Parser, ParseResult } from '../types'
import { docxParser } from './docx'
import { htmlParser } from './html'
import { pdfParser } from './pdf'
import { markdownParser, rtfParser, textParser } from './text'
import { xlsxParser } from './xlsx'

// Order matters for accept() conflicts — more specific parsers go first.
const parsers: Parser[] = [
  pdfParser,
  docxParser,
  xlsxParser,
  markdownParser,
  htmlParser,  // strips <script>/<style>/tags; runs before textParser's catch-all
  rtfParser,
  textParser, // .txt and the catch-all fallback path
]

/**
 * Add a parser to the front of the registry. Newer parsers are tried before
 * the built-ins, so a custom JSON-Slack adapter can override the generic
 * text parser for .json files.
 */
export function registerParser(parser: Parser): void {
  parsers.unshift(parser)
}

/** Read-only view of the current parser registry. Useful for debugging. */
export function listParsers(): readonly Parser[] {
  return parsers
}

export async function parseFile(file: File): Promise<ParseResult> {
  const parser = parsers.find((p) => p.accept(file))
  if (parser) {
    const sections = await parser.parse(file)
    return {
      sections,
      metadata: {
        filename: file.name,
        size: file.size,
        parserUsed: parser.name,
        parsedAt: Date.now(),
      },
    }
  }
  // Fallback: treat unknown extensions as plain text. Mirrors parsers.py
  // behavior, which falls through to `filepath.read_text(errors="replace")`.
  const sections = await textParser.parse(file)
  return {
    sections,
    metadata: {
      filename: file.name,
      size: file.size,
      parserUsed: `${textParser.name} (fallback)`,
      parsedAt: Date.now(),
    },
  }
}

export {
  docxParser,
  markdownParser,
  pdfParser,
  rtfParser,
  textParser,
  xlsxParser,
}

// Day 2 additions — PDF classifier (browser twin of api/cartbuilder/parsers
// classify_pdf) + Image Builder delegation path. Not part of the registry;
// the pipeline routes to these explicitly via MIME check + PDF classify.
export {
  classifyPdf,
  PDF_CLASSIFY_MAX_PAGES,
  PDF_CLASSIFY_TEXT_THRESHOLD,
} from './pdf'
export {
  graphicsToSections,
  isImageFile,
  parseViaImageBuilder,
  tableHtmlToText,
  tablesToSections,
  type ImageParseOutput,
} from './image'
