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
