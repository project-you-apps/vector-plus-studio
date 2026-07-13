// Report export helpers. Client-side converters for the download dropdown
// on the Reports results view. Zero-dep by design — we already have
// react-markdown + remark-gfm for on-screen rendering, but the exported
// HTML file should be self-contained + print-friendly with inline CSS,
// not the app's dark-theme Tailwind classes. So we roll a small
// markdown→HTML converter here instead of pulling in react-dom/server.
//
// v1 formats live: Markdown, plain text, HTML.
// v1.5 / v2 placeholders: PDF, DOCX, CSV, XLSX (rendered greyed in the UI).

// ---------------------------------------------------------------------------
// Filename helpers
// ---------------------------------------------------------------------------

/** Lowercase, strip whitespace, replace non-alphanumerics with `-`, dedupe. */
export function slugify(input: string): string {
  const s = (input || '')
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return s || 'untitled'
}

/** Build `<cart_ref>_<report_slug>_<YYYYMMDD>.<ext>` per the export naming rule. */
export function buildFilename(cartRef: string, reportSlug: string, ext: string): string {
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${slugify(cartRef)}_${slugify(reportSlug)}_${y}${m}${day}.${ext}`
}

/** Force-download a text blob via a temporary anchor. */
export function downloadTextFile(content: string, filename: string, mime: string): void {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  // Revoke on next tick so Firefox has time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

// ---------------------------------------------------------------------------
// Markdown → plain text
// ---------------------------------------------------------------------------

/** Strip markdown syntax for a readable .txt export. Best-effort, not perfect. */
export function markdownToPlainText(md: string): string {
  const lines = md.replace(/\r\n/g, '\n').split('\n')
  const out: string[] = []
  let inFence = false
  let tableBuffer: string[] = []

  const inlineStrip = (s: string): string => {
    // Inline code: keep content, drop backticks.
    s = s.replace(/`([^`]+)`/g, '$1')
    // Bold/italic — bold first so ** isn't caught by * pattern.
    s = s.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/__([^_]+)__/g, '$1')
    s = s.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1$2')
    s = s.replace(/(^|[^_])_([^_]+)_(?!_)/g, '$1$2')
    // Images before links so ![alt](url) doesn't get partial-matched.
    s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '$1 ($2)')
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1 ($2)')
    return s
  }

  const flushTable = () => {
    if (tableBuffer.length === 0) return
    const rows = tableBuffer
      .map(l => l.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map(c => c.trim()))
      // Drop the separator row (---|---, :--|--:, etc).
      .filter(cells => !cells.every(c => /^:?-+:?$/.test(c)))
    if (rows.length > 0) {
      const widths = rows[0].map((_, colIdx) =>
        Math.max(...rows.map(r => (r[colIdx] || '').length))
      )
      for (const row of rows) {
        out.push(
          row.map((c, i) => inlineStrip(c || '').padEnd(widths[i] || 0))
            .join('  ')
            .trimEnd()
        )
      }
      out.push('')
    }
    tableBuffer = []
  }

  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      flushTable()
      inFence = !inFence
      continue
    }
    if (inFence) {
      out.push(line)
      continue
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      tableBuffer.push(line.trim())
      continue
    }
    if (tableBuffer.length > 0) flushTable()

    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      out.push('----------------')
      continue
    }
    const hMatch = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/)
    if (hMatch) {
      out.push(inlineStrip(hMatch[2].trim()))
      continue
    }
    const bqMatch = line.match(/^\s*>\s?(.*)$/)
    if (bqMatch) {
      out.push('    ' + inlineStrip(bqMatch[1]))
      continue
    }
    const ulMatch = line.match(/^(\s*)[-*+]\s+(.*)$/)
    if (ulMatch) {
      out.push(`${ulMatch[1]}  - ${inlineStrip(ulMatch[2])}`)
      continue
    }
    const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/)
    if (olMatch) {
      out.push(`${olMatch[1]}  ${olMatch[2]}. ${inlineStrip(olMatch[3])}`)
      continue
    }
    out.push(inlineStrip(line))
  }
  flushTable()
  // Collapse 3+ blank lines to 2 for readable spacing.
  return out.join('\n').replace(/\n{3,}/g, '\n\n').trim() + '\n'
}

// ---------------------------------------------------------------------------
// Markdown → HTML
// ---------------------------------------------------------------------------

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function inlineToHtml(s: string): string {
  // Protect inline-code spans BEFORE HTML-escaping so we can restore them
  // as <code> after everything else has been transformed. Uses U+E000
  // (Private Use Area) as sentinel — that codepoint is guaranteed never
  // to appear in ordinary markdown text, so the restore regex is safe.
  const codeSpans: string[] = []
  s = s.replace(/`([^`]+)`/g, (_m, c) => {
    codeSpans.push(c)
    return `\uE000CODE${codeSpans.length - 1}\uE000`
  })
  s = escapeHtml(s)
  // Bold + italic. Bold first so the `**` isn't matched by the `*` rule.
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
       .replace(/__([^_]+)__/g, '<strong>$1</strong>')
       .replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>')
       .replace(/(^|[^_])_([^_]+)_(?!_)/g, '$1<em>$2</em>')
  // Links [text](url). URL is already HTML-escaped by escapeHtml above.
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
    (_m, t, u) => `<a href="${u}">${t}</a>`)
  // Restore code spans (escape their content since we protected pre-escape).
  s = s.replace(/\uE000CODE(\d+)\uE000/g,
    (_m, i) => `<code>${escapeHtml(codeSpans[Number(i)])}</code>`)
  return s
}

/** Small line-oriented markdown → HTML converter. Handles headers, paragraphs,
 *  ordered/unordered lists, GFM tables, blockquotes, code fences, HR, and the
 *  inline set (bold, italic, code, links). Enough for the shape our reports
 *  emit; not a full CommonMark implementation. */
export function markdownToHtml(md: string): string {
  const lines = md.replace(/\r\n/g, '\n').split('\n')
  const out: string[] = []
  let i = 0
  let para: string[] = []

  const flushPara = () => {
    if (para.length === 0) return
    out.push(`<p>${inlineToHtml(para.join(' '))}</p>`)
    para = []
  }

  while (i < lines.length) {
    const line = lines[i]

    // Code fence
    if (/^\s*```/.test(line)) {
      flushPara()
      i++
      const buf: string[] = []
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        buf.push(lines[i])
        i++
      }
      if (i < lines.length) i++ // consume closing fence
      out.push(`<pre><code>${escapeHtml(buf.join('\n'))}</code></pre>`)
      continue
    }

    // Blank line
    if (/^\s*$/.test(line)) {
      flushPara()
      i++
      continue
    }

    // Header
    const hMatch = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/)
    if (hMatch) {
      flushPara()
      const level = hMatch[1].length
      out.push(`<h${level}>${inlineToHtml(hMatch[2])}</h${level}>`)
      i++
      continue
    }

    // Horizontal rule
    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushPara()
      out.push('<hr />')
      i++
      continue
    }

    // Blockquote (potentially multi-line)
    if (/^\s*>/.test(line)) {
      flushPara()
      const buf: string[] = []
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ''))
        i++
      }
      out.push(`<blockquote>${inlineToHtml(buf.join(' '))}</blockquote>`)
      continue
    }

    // GFM table (header row + |---|---| separator + body rows)
    const tableSep = i + 1 < lines.length
      ? /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$/.test(lines[i + 1])
      : false
    if (/^\s*\|.*\|\s*$/.test(line) && tableSep) {
      flushPara()
      const splitRow = (l: string) =>
        l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim())
      const header = splitRow(line)
      i += 2 // skip header + separator
      const rows: string[][] = []
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        rows.push(splitRow(lines[i]))
        i++
      }
      const th = header.map(c => `<th>${inlineToHtml(c)}</th>`).join('')
      const body = rows
        .map(r => `<tr>${r.map(c => `<td>${inlineToHtml(c)}</td>`).join('')}</tr>`)
        .join('')
      out.push(`<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`)
      continue
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      flushPara()
      const items: string[] = []
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ''))
        i++
      }
      out.push(`<ol>${items.map(it => `<li>${inlineToHtml(it)}</li>`).join('')}</ol>`)
      continue
    }

    // Unordered list
    if (/^\s*[-*+]\s+/.test(line)) {
      flushPara()
      const items: string[] = []
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ''))
        i++
      }
      out.push(`<ul>${items.map(it => `<li>${inlineToHtml(it)}</li>`).join('')}</ul>`)
      continue
    }

    // Otherwise: accumulate as a paragraph line.
    para.push(line)
    i++
  }
  flushPara()
  return out.join('\n')
}

/** Wrap a rendered HTML body in a print-friendly HTML5 document with inline
 *  CSS. Users can open the file in a browser and Ctrl+P → Save as PDF as a
 *  workaround until native PDF export lands (v1.5). */
export function wrapHtmlDocument(
  bodyHtml: string,
  reportDisplayName: string,
  cartRef: string,
): string {
  const isoDate = new Date().toISOString().slice(0, 10)
  const title = escapeHtml(reportDisplayName)
  const cart = escapeHtml(cartRef)
  // Inline CSS: system font stack, centered 780px column, sensible header
  // spacing, table borders + zebra striping, code blocks with monospace +
  // light bg, blockquote with left border.
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>${title} &mdash; ${cart}</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
    max-width: 780px;
    margin: 2em auto;
    padding: 0 1em 3em;
    color: #222;
    line-height: 1.55;
  }
  h1 { font-size: 1.8em; margin: 1.4em 0 0.5em; }
  h2 { font-size: 1.35em; margin: 1.2em 0 0.4em;
       border-bottom: 1px solid #eee; padding-bottom: 0.15em; }
  h3 { font-size: 1.12em; margin: 1em 0 0.3em; }
  h4, h5, h6 { font-size: 1em; margin: 0.9em 0 0.25em; }
  p { margin: 0.6em 0; }
  ul, ol { padding-left: 1.5em; margin: 0.5em 0; }
  li { margin: 0.2em 0; }
  .meta { color: #777; font-size: 0.9em; margin: 0 0 2em; }
  table { border-collapse: collapse; margin: 1em 0; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 0.45em 0.7em;
           text-align: left; vertical-align: top; }
  th { background: #f4f4f4; font-weight: 600; }
  tbody tr:nth-child(even) { background: #fafafa; }
  code { background: #f4f4f4; padding: 0.12em 0.35em; border-radius: 3px;
         font-family: Menlo, Consolas, "Courier New", monospace;
         font-size: 0.92em; }
  pre { background: #f6f6f6; padding: 0.8em 1em; border-radius: 4px;
        overflow-x: auto; }
  pre code { background: transparent; padding: 0; }
  blockquote { border-left: 3px solid #ccc; margin: 1em 0;
               padding: 0.25em 1em; color: #555; }
  hr { border: 0; border-top: 1px solid #ddd; margin: 1.5em 0; }
  a { color: #0366d6; }
  @media print {
    body { margin: 0.5in 0.6in; max-width: none; }
    a { color: #222; text-decoration: none; }
  }
</style>
</head>
<body>
<h1>${title}</h1>
<p class="meta">Generated from cart <strong>${cart}</strong> on ${isoDate}</p>
${bodyHtml}
</body>
</html>
`
}
