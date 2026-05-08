import type { Parser, Section } from '../types'

export const textParser: Parser = {
  name: 'text',
  accept(file: File): boolean {
    const lower = file.name.toLowerCase()
    return lower.endsWith('.txt') || file.type === 'text/plain'
  },
  async parse(file: File): Promise<Section[]> {
    const text = (await file.text()).trim()
    if (!text) return []
    return [{ text, page: null, source: file.name }]
  },
}

export const markdownParser: Parser = {
  name: 'markdown',
  accept(file: File): boolean {
    const lower = file.name.toLowerCase()
    return (
      lower.endsWith('.md') ||
      lower.endsWith('.markdown') ||
      file.type === 'text/markdown'
    )
  },
  // Splits at "## " H2 headers, mirroring parsers.py:parse_markdown.
  async parse(file: File): Promise<Section[]> {
    const text = await file.text()
    const lines = text.split('\n')
    const rawSections: string[] = []
    let current: string[] = []
    for (const line of lines) {
      if (line.startsWith('## ') && current.length > 0) {
        rawSections.push(current.join('\n'))
        current = [line]
      } else {
        current.push(line)
      }
    }
    if (current.length > 0) {
      rawSections.push(current.join('\n'))
    }

    const sections: Section[] = []
    rawSections.forEach((section, i) => {
      const trimmed = section.trim()
      if (trimmed) {
        sections.push({
          text: trimmed,
          page: i + 1,
          source: file.name,
        })
      }
    })
    return sections
  },
}

// Lightweight RTF stripper. Handles standard RTF control words, hex-escaped
// chars, and groups. Complex RTF (tables, embedded objects, font-mapped
// non-ASCII) may degrade. Sufficient for Word-saved plain-text RTFs; upgrade
// to a proper library if quality becomes an issue.
function stripRtf(rtf: string): string {
  let s = rtf
  // \\'XX hex-escaped char → real char
  s = s.replace(/\\'([0-9a-fA-F]{2})/g, (_, hex) =>
    String.fromCharCode(parseInt(hex, 16))
  )
  // Unicode \\uNNNN — keep the codepoint, drop the optional fallback char that follows
  s = s.replace(/\\u(-?\d+)\??/g, (_, n) =>
    String.fromCharCode(((parseInt(n, 10) % 65536) + 65536) % 65536)
  )
  // Control words: \\word optional-arg optional-trailing-space
  s = s.replace(/\\[a-zA-Z]+-?\d* ?/g, ' ')
  // Control symbols (\\X for non-letter X like \\* \\~ \\-)
  s = s.replace(/\\[^a-zA-Z]/g, '')
  // Group braces
  s = s.replace(/[{}]/g, '')
  // Collapse whitespace
  s = s.replace(/\s+/g, ' ').trim()
  return s
}

export const rtfParser: Parser = {
  name: 'rtf',
  accept(file: File): boolean {
    const lower = file.name.toLowerCase()
    return (
      lower.endsWith('.rtf') ||
      file.type === 'application/rtf' ||
      file.type === 'text/rtf'
    )
  },
  async parse(file: File): Promise<Section[]> {
    const raw = await file.text()
    const text = stripRtf(raw)
    if (!text) return []
    return [{ text, page: null, source: file.name }]
  },
}
