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

// Clean Notion export filenames. Notion appends a 32-char hex hash to
// every page filename and URL-encodes spaces — so a page titled
// "Project You" exports as "Project%20You%2018a37b25fcb44a6abc1234567890abcd.md".
// We URL-decode and strip the hash suffix from each path component so
// the `source` field reads cleanly. Heuristic: match a space + 16+ hex
// chars before `.md` (or at directory boundary). Dates like "2026-05-09"
// don't match (too short). Normal markdown filenames pass through unchanged.
function normalizeNotionPath(path: string): string {
  let decoded = path
  try {
    decoded = decodeURIComponent(path)
  } catch {
    // malformed URL encoding — fall through with the original
  }
  return decoded
    .split('/')
    .map((part) => part.replace(/\s+[a-f0-9]{16,}(\.md)?$/i, '$1'))
    .join('/')
}

// Convert YAML frontmatter to a readable single-line summary.
// Obsidian (and Jekyll/Hugo/Astro) markdown files often start with a
// `---\n...\n---` YAML block. Without conversion the raw YAML lands at
// the top of section 1, polluting the embedding and the passage preview.
// Output: "Properties: title: Galileo Galilei. tags: people, italian.
// aliases: Galileo." which preserves searchable signal in plain prose.
function extractFrontmatter(text: string): { body: string; frontmatter: string | null } {
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/)
  if (!match) return { body: text, frontmatter: null }
  const yamlBody = match[1]
  const rest = text.slice(match[0].length)
  const lines = yamlBody.split(/\r?\n/).filter((l) => l.trim().length > 0)
  const props: string[] = []
  for (const line of lines) {
    const m = line.match(/^([A-Za-z0-9_-]+)\s*:\s*(.*)$/)
    if (!m) continue
    const key = m[1]
    let value = m[2].trim()
    // Unwrap simple inline arrays: [a, b, c] -> "a, b, c"
    if (value.startsWith('[') && value.endsWith(']')) {
      value = value.slice(1, -1).trim()
    }
    // Strip surrounding quotes on simple strings
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    if (value.length > 0) props.push(`${key}: ${value}`)
  }
  if (props.length === 0) return { body: rest, frontmatter: null }
  return { body: rest, frontmatter: `Properties: ${props.join('. ')}.` }
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
  // Additionally strips YAML frontmatter (Obsidian / Jekyll / Hugo
  // convention) and replaces it with a readable Properties summary at
  // the top of section 1, so metadata stays embedded but doesn't
  // pollute passage previews.
  async parse(file: File): Promise<Section[]> {
    const raw = await file.text()
    const { body, frontmatter } = extractFrontmatter(raw)
    const text = frontmatter ? `${frontmatter}\n\n${body}` : body
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

    // Prefer webkitRelativePath when available (set by directory-drop) so
    // `source` carries the vault folder structure, e.g. "People/Galileo.md"
    // instead of just "Galileo.md". Falls back to file.name for normal drops.
    // Also normalize Notion export filenames (URL-decode + strip 32-char hash
    // suffix) so Notion-source carts read cleanly.
    const rawPath = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name
    const sourcePath = normalizeNotionPath(rawPath)

    const sections: Section[] = []
    rawSections.forEach((section, i) => {
      const trimmed = section.trim()
      if (trimmed) {
        sections.push({
          text: trimmed,
          page: i + 1,
          source: sourcePath,
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
