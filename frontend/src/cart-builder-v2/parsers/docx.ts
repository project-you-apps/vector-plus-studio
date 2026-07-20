import mammoth from 'mammoth'
import type { Parser, Section } from '../types'
import { ParseError } from '../types'
import { sourcePathForFile } from '../sourcePath'

// mammoth handles .docx (zipped XML). Legacy .doc (binary) is NOT supported —
// the accept() lets the extension through but parse() will throw. Mirrors
// Python: python-docx has the same limitation.
export const docxParser: Parser = {
  name: 'docx',
  accept(file: File): boolean {
    const lower = file.name.toLowerCase()
    return (
      lower.endsWith('.docx') ||
      lower.endsWith('.doc') ||
      file.type ===
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )
  },
  async parse(file: File): Promise<Section[]> {
    const buffer = await file.arrayBuffer()
    let result
    try {
      result = await mammoth.extractRawText({ arrayBuffer: buffer })
    } catch (err) {
      throw new ParseError('Failed to extract DOCX text', file.name, err)
    }
    const text = result.value.trim()
    if (!text) return []
    return [{ text, page: null, source: sourcePathForFile(file) }]
  },
}
