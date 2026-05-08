import * as XLSX from 'xlsx'
import type { Parser, Section } from '../types'
import { ParseError } from '../types'

// CSV is included here (SheetJS handles it natively) — small deviation from
// Python parsers.py, which omits CSV. Worth it: zero extra code, common format.
export const xlsxParser: Parser = {
  name: 'xlsx',
  accept(file: File): boolean {
    const lower = file.name.toLowerCase()
    return (
      lower.endsWith('.xlsx') ||
      lower.endsWith('.xls') ||
      lower.endsWith('.csv') ||
      file.type ===
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
      file.type === 'text/csv'
    )
  },
  async parse(file: File): Promise<Section[]> {
    const buffer = await file.arrayBuffer()
    let workbook: XLSX.WorkBook
    try {
      workbook = XLSX.read(buffer, { type: 'array' })
    } catch (err) {
      throw new ParseError('Failed to read spreadsheet', file.name, err)
    }
    const sections: Section[] = []
    for (const sheetName of workbook.SheetNames) {
      const sheet = workbook.Sheets[sheetName]
      const aoa: unknown[][] = XLSX.utils.sheet_to_json(sheet, {
        header: 1,
        defval: '',
      })
      const rows: string[] = []
      for (const row of aoa) {
        const cells = row.map((c) => (c == null ? '' : String(c)))
        const line = cells.filter((c) => c).join(' | ')
        if (line.trim()) {
          rows.push(line)
        }
      }
      if (rows.length > 0) {
        const text = `Sheet: ${sheetName}\n${rows.join('\n')}`
        sections.push({
          text,
          page: null,
          source: `${file.name}:${sheetName}`,
        })
      }
    }
    return sections
  },
}
