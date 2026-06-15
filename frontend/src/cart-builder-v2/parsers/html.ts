import type { Parser, Section } from '../types'

/**
 * HTML parser using browser-native DOMParser to strip tags and extract
 * readable text content. Removes <script>, <style>, <noscript> blocks
 * so embeddings + previews don't get polluted by JS/CSS source.
 * Falls back to the root element if no <body> is present.
 *
 * For .html / .htm files. Mirrors api/parsers.py:parse_html behavior
 * (BeautifulSoup-based on the server) so browser-built carts stay
 * close to server-built ones for the same input.
 */
export const htmlParser: Parser = {
  name: 'html',
  accept(file: File): boolean {
    const lower = file.name.toLowerCase()
    return lower.endsWith('.html') || lower.endsWith('.htm') || file.type === 'text/html'
  },
  async parse(file: File): Promise<Section[]> {
    const raw = await file.text()
    const parser = new DOMParser()
    const doc = parser.parseFromString(raw, 'text/html')
    // Strip non-content nodes that would pollute the embedding.
    doc.querySelectorAll('script, style, noscript').forEach((el) => el.remove())
    const root = doc.body || doc.documentElement
    // Collapse whitespace runs so multi-line markup doesn't produce ragged
    // gaps in the preview. Lossy on paragraph structure but clean for
    // retrieval — cosine similarity is token-bag at this scale anyway.
    const text = (root?.textContent || '').replace(/\s+/g, ' ').trim()
    if (!text) return []
    return [{ text, page: null, source: file.name }]
  },
}
