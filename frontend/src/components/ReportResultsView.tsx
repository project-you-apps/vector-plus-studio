import { Children, isValidElement, useEffect, useRef, useState } from 'react'
import {
  X, RefreshCw, Copy, Check, ExternalLink, Clock3, AlertTriangle,
  Download, ChevronDown, ChevronRight, FileText, Code, FileType, File, Table, Sheet,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { GenerateReportResponse } from '../api/client'
import type { FieldSchema, ReportDefinition } from '../reports/report-definitions'
import { useAppStore } from '../store/appStore'
import {
  buildFilename,
  downloadTextFile,
  markdownToHtml,
  markdownToPlainText,
  wrapHtmlDocument,
} from '../lib/exportReport'

// Prefix used by all in-report source-file links. Kept as a constant so
// the custom react-markdown handler + any future exporter that wants to
// visit source refs agree on the syntax.
const VPS_SOURCE_PREFIX = 'vps://source/'

// Recursively flatten React children into a plain-text string. Used to
// recover the display text of a rendered markdown link when the
// react-markdown `a` handler needs to hand a source name off to the
// Search tab (see `focusSearchOnSource`). Handles nested `<strong>` /
// `<em>` inside link text — rare in report output but not impossible.
function childrenToText(children: React.ReactNode): string {
  let out = ''
  Children.forEach(children, (child) => {
    if (typeof child === 'string') {
      out += child
    } else if (typeof child === 'number') {
      out += String(child)
    } else if (isValidElement(child)) {
      out += childrenToText(
        (child.props as { children?: React.ReactNode }).children,
      )
    }
  })
  return out
}

// Full-width results view that REPLACES the reports grid when a report has
// been generated (design — Option 3). Lives inside the main
// content column so `max-w-6xl mx-auto` from ReportsScreen already caps its
// width sensibly on ultra-wide monitors.
//
// Toolbar buttons: Close (dismiss), Regenerate (re-open the input pane
// pre-filled with the last inputs), Copy markdown, Download (dropdown —
// v1 ships MD/TXT/HTML working with PDF/DOCX/CSV/XLSX as greyed
// placeholders), Open in new tab. The "Open in new tab" button is a
// coming-soon placeholder — the URL routing story is a follow-up. Right
// side of the toolbar shows the cart name + report display name +
// generation-time badge.
//
// Body renders the markdown response full-width via react-markdown +
// remark-gfm (same override map as the old SuccessPanel inside
// ReportInputPane) so long tables don't get squeezed like they did in the
// 440px slide-in pane.

export default function ReportResultsView({
  report,
  response,
  cartLabel,
  submittedInputs,
  onClose,
  onRegenerate,
}: {
  report: ReportDefinition
  response: GenerateReportResponse
  cartLabel: string | null
  submittedInputs: Record<string, unknown>
  onClose: () => void
  onRegenerate: () => void
}) {
  const [justCopied, setJustCopied] = useState(false)
  const copyTimerRef = useRef<number | null>(null)
  const [downloadOpen, setDownloadOpen] = useState(false)
  const downloadWrapRef = useRef<HTMLDivElement | null>(null)

  const copyMarkdown = async () => {
    try {
      await navigator.clipboard.writeText(response.markdown)
      setJustCopied(true)
      if (copyTimerRef.current !== null) window.clearTimeout(copyTimerRef.current)
      copyTimerRef.current = window.setTimeout(() => {
        setJustCopied(false)
        copyTimerRef.current = null
      }, 2500)
    } catch {
      // Insecure context / permissions denied — silent no-op; the raw
      // markdown is still selectable with the mouse.
    }
  }

  // Cart label is the display string; the actual cart_ref used for the
  // download filename is on the response metadata (the server echoes it
  // back). Fall back to the visible label if it's absent for any reason.
  const cartRefForFilename = String(
    (response.metadata && (response.metadata.cart_ref as string | undefined)) ||
    cartLabel ||
    'cart'
  )

  const doDownload = (fmt: 'md' | 'txt' | 'html') => {
    if (fmt === 'md') {
      downloadTextFile(
        response.markdown,
        buildFilename(cartRefForFilename, response.report_slug, 'md'),
        'text/markdown',
      )
    } else if (fmt === 'txt') {
      downloadTextFile(
        markdownToPlainText(response.markdown),
        buildFilename(cartRefForFilename, response.report_slug, 'txt'),
        'text/plain',
      )
    } else if (fmt === 'html') {
      const body = markdownToHtml(response.markdown)
      const doc = wrapHtmlDocument(body, report.displayName, cartRefForFilename)
      downloadTextFile(
        doc,
        buildFilename(cartRefForFilename, response.report_slug, 'html'),
        'text/html',
      )
    }
    setDownloadOpen(false)
  }

  // Close the download menu on outside click / Escape.
  useEffect(() => {
    if (!downloadOpen) return
    const onDocClick = (e: MouseEvent) => {
      if (!downloadWrapRef.current) return
      if (!downloadWrapRef.current.contains(e.target as Node)) {
        setDownloadOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDownloadOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [downloadOpen])

  // Prefer the report author's own elapsed_ms; fall back to the route
  // fallback in metadata.route_elapsed_ms. Both are integers in ms.
  const meta = response.metadata || {}
  const elapsed =
    typeof meta.elapsed_ms === 'number'
      ? meta.elapsed_ms as number
      : typeof meta.route_elapsed_ms === 'number'
        ? meta.route_elapsed_ms as number
        : null
  const footprint = _getRetrievalFootprint(response.metadata)

  return (
    <section
      className="rounded-lg border border-slate-700 bg-slate-800/30 flex flex-col
                 animate-in fade-in duration-200"
      aria-label={`${report.displayName} results`}
    >
      {/* Toolbar — left cluster is action buttons, right cluster is identity */}
      <div
        className="flex flex-wrap items-center gap-2 px-4 py-3 border-b border-slate-700
                   bg-slate-900/40 rounded-t-lg"
      >
        <div className="flex items-center gap-1.5">
          <ToolbarButton
            icon={<X size={13} />}
            label="Close"
            onClick={onClose}
            title="Close results and return to the report grid"
          />
          <ToolbarButton
            icon={<RefreshCw size={13} />}
            label="Regenerate"
            onClick={onRegenerate}
            title="Re-open the inputs pre-filled with the last-used values"
            highlight
          />
          <ToolbarButton
            icon={justCopied ? <Check size={13} /> : <Copy size={13} />}
            label={justCopied ? 'Copied' : 'Copy markdown'}
            onClick={copyMarkdown}
            title="Copy the raw markdown to your clipboard"
            active={justCopied}
          />
          <div ref={downloadWrapRef} className="relative">
            <ToolbarButton
              icon={<Download size={13} />}
              label="Download"
              trailingIcon={<ChevronDown size={11} />}
              onClick={() => setDownloadOpen(o => !o)}
              title="Download this report in a chosen format"
              menuOpen={downloadOpen}
            />
            {downloadOpen && (
              <DownloadMenu onSelect={doDownload} />
            )}
          </div>
          <ToolbarButton
            icon={<ExternalLink size={13} />}
            label="Open in new tab"
            onClick={() => { /* placeholder — see title tooltip */ }}
            title="Coming in a future release"
            disabled
          />
        </div>

        <div className="ml-auto flex flex-col items-end gap-0.5 text-right min-w-0">
          <div className="flex items-center gap-2 text-[11px] text-slate-400 min-w-0">
            {cartLabel && (
              <span className="font-mono truncate max-w-[220px]" title={cartLabel}>
                {cartLabel}
              </span>
            )}
            <span className="text-slate-600">·</span>
            <span className="text-slate-200 font-medium truncate max-w-[220px]" title={report.displayName}>
              {report.displayName}
            </span>
          </div>
          {elapsed !== null && (
            <span className="inline-flex items-center gap-1 font-mono text-[10px] text-slate-500">
              <Clock3 size={10} />
              {elapsed} ms
            </span>
          )}
          {footprint && (
            <span
              className="font-mono text-[10px] text-slate-500"
              title="Reports work over the patterns your cart contains. Some reports scan every live pattern; Q&A-shaped reports pull the most relevant few for your query."
            >
              scanned {footprint.patterns} pattern{footprint.patterns === 1 ? '' : 's'} from{' '}
              {footprint.sources} source{footprint.sources === 1 ? '' : 's'}
            </span>
          )}
        </div>
      </div>

      {/* Inputs — collapsible summary of the exact values that produced this
          report. Sits between the header row and the response body so users
          reviewing a saved-to-cart result can see WHAT was asked without
          re-opening the input pane. Single-input schemas auto-expand to a
          body-styled paragraph; multi-input schemas render a compact
          "▶ Inputs · Label: value · ..." summary that toggles to a full
          list. */}
      <InputsSection schema={report.inputSchema} values={submittedInputs} accent="emerald" />

      {/* Warnings — mirror the pane's amber style so the visual language
          stays consistent between the old narrow surface and the new
          full-width one. */}
      {response.warnings.length > 0 && (
        <div className="mx-4 mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5
                        px-3 py-2 text-[12px] text-amber-200 space-y-1">
          <div className="flex items-center gap-1.5 font-medium">
            <AlertTriangle size={12} />
            Warnings
          </div>
          <ul className="list-disc pl-4 space-y-0.5">
            {response.warnings.map((w, i) => (<li key={i}>{w}</li>))}
          </ul>
        </div>
      )}

      {/* Body — full-width markdown. `max-w-5xl` keeps line lengths
          readable on ultra-wide monitors without collapsing tables. */}
      <div className="px-6 py-5">
        <div className="mx-auto max-w-5xl text-slate-200 leading-relaxed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            urlTransform={(uri) => uri}
            components={{
              h1: ({ children }) => <h1 className="text-2xl font-bold text-slate-100 mt-4 mb-3">{children}</h1>,
              h2: ({ children }) => <h2 className="text-xl font-semibold text-slate-100 mt-4 mb-2">{children}</h2>,
              h3: ({ children }) => <h3 className="text-base font-semibold text-slate-200 mt-3 mb-1.5">{children}</h3>,
              p: ({ children }) => <p className="my-2 text-[14px] text-slate-300">{children}</p>,
              ul: ({ children }) => <ul className="list-disc pl-6 my-2 space-y-1 text-[14px]">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-6 my-2 space-y-1 text-[14px]">{children}</ol>,
              li: ({ children }) => <li className="text-slate-300">{children}</li>,
              code: ({ children }) => (
                <code className="px-1 py-0.5 rounded bg-slate-800 text-amber-200 font-mono text-[12px]">{children}</code>
              ),
              pre: ({ children }) => (
                <pre className="my-2 p-3 rounded-md bg-slate-900/80 border border-slate-800 overflow-x-auto text-[12px] font-mono text-slate-200">{children}</pre>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-emerald-500/40 pl-3 my-2 italic text-slate-400 text-[14px]">{children}</blockquote>
              ),
              // Tables get the whole extra width — no overflow-x wrapper
              // clamping them anymore. If they're too wide the outer body
              // scrolls, which is the same read as a document.
              table: ({ children }) => (
                <div className="my-4 overflow-x-auto">
                  <table className="border-collapse text-[13px] w-full">{children}</table>
                </div>
              ),
              th: ({ children }) => (
                <th className="border border-slate-700 px-3 py-1.5 bg-slate-800/60 text-slate-200 text-left font-medium">{children}</th>
              ),
              td: ({ children }) => (
                <td className="border border-slate-700 px-3 py-1.5 text-slate-300 align-top">{children}</td>
              ),
              hr: () => <hr className="my-4 border-slate-700/50" />,
              strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
              em: ({ children }) => <em className="italic text-slate-200">{children}</em>,
              a: ({ href, children }) => {
                // source-file links: intercept vps://source/{slug}
                // and hand off to the Search tab via the store. Never
                // navigate — the href is a marker scheme, not a real URL.
                // Non-vps:// links keep the old external-link behavior.
                if (href && href.startsWith(VPS_SOURCE_PREFIX)) {
                  const slug = href.slice(VPS_SOURCE_PREFIX.length)
                  return (
                    <a
                      href={href}
                      onClick={(e) => {
                        e.preventDefault()
                        const displayName = childrenToText(children).trim()
                        useAppStore.getState().focusSearchOnSource(
                          slug,
                          displayName,
                        )
                      }}
                      title={`Focus Search on ${childrenToText(children).trim() || slug}`}
                      className="text-purple-300 hover:text-purple-200 underline decoration-purple-400/50 decoration-dotted cursor-pointer"
                    >
                      {children}
                    </a>
                  )
                }
                return (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline decoration-cyan-400/40"
                  >
                    {children}
                  </a>
                )
              },
            }}
          >
            {response.markdown}
          </ReactMarkdown>
        </div>
      </div>

      {/* Footer — debug crumb; helps operators trace exported markdown
          back to which slug + when it ran. Same content as the pane
          previously showed at the bottom of its own SuccessPanel. */}
      <div className="border-t border-slate-700/60 px-4 py-2 text-[10px] text-slate-600 text-center rounded-b-lg">
        {report.displayName} · {response.report_slug} ·{' '}
        {new Date(response.generated_at).toLocaleString()}
      </div>
    </section>
  )
}

// Common toolbar button. Kept in-file — nothing else in the tree needs
// this shape. Highlight variant is used for Regenerate to nudge the user
// toward it as the "primary next thing you can do" after Close. The
// trailingIcon slot is used by the Download button to render a caret
// hinting at the dropdown affordance.
function ToolbarButton({
  icon,
  label,
  onClick,
  title,
  highlight,
  active,
  menuOpen,
  disabled,
  trailingIcon,
}: {
  icon: React.ReactNode
  label: string
  onClick: () => void
  title?: string
  highlight?: boolean
  active?: boolean
  menuOpen?: boolean
  disabled?: boolean
  trailingIcon?: React.ReactNode
}) {
  const base =
    'flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium border transition-colors'
  const style = disabled
    ? 'bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed'
    : menuOpen
      ? 'bg-purple-500/15 border-purple-500/40 text-purple-200'
      : active
        ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-200'
        : highlight
          ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/20'
          : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
  return (
    <button
      onClick={disabled ? undefined : onClick}
      className={`${base} ${style}`}
      title={title}
      disabled={disabled}
      aria-disabled={disabled}
      aria-haspopup={trailingIcon ? 'menu' : undefined}
      aria-expanded={trailingIcon ? menuOpen : undefined}
    >
      {icon}
      {label}
      {trailingIcon}
    </button>
  )
}

// Download dropdown menu. Anchored below the Download button (or above if
// there's no room). Working formats trigger a client-side download and
// close the menu; disabled placeholders show a tooltip pointing at the
// target version. Matches the toolbar's dark-theme + purple accent.
function DownloadMenu({
  onSelect,
}: {
  onSelect: (fmt: 'md' | 'txt' | 'html') => void
}) {
  return (
    <div
      role="menu"
      aria-label="Download format"
      className="absolute z-20 mt-1 left-0 min-w-[200px] rounded-md border
                 border-slate-700 bg-slate-900/95 shadow-lg shadow-black/40
                 backdrop-blur-sm py-1 text-[12px] animate-in fade-in
                 duration-100"
    >
      <DownloadMenuItem
        icon={<FileText size={13} />}
        label="Markdown (.md)"
        onClick={() => onSelect('md')}
      />
      <DownloadMenuItem
        icon={<FileText size={13} />}
        label="Plain text (.txt)"
        onClick={() => onSelect('txt')}
      />
      <DownloadMenuItem
        icon={<Code size={13} />}
        label="HTML (.html)"
        onClick={() => onSelect('html')}
      />
      <div className="my-1 border-t border-slate-700/60" />
      <DownloadMenuItem
        icon={<FileType size={13} />}
        label="PDF (.pdf)"
        disabled
        title="Coming in v1.5 — for now: download HTML and use browser Print → Save as PDF"
      />
      <DownloadMenuItem
        icon={<File size={13} />}
        label="Word (.docx)"
        disabled
        title="Coming in v2"
      />
      <DownloadMenuItem
        icon={<Table size={13} />}
        label="CSV (.csv)"
        disabled
        title="Coming in v2 (needs structured tables from report engine)"
      />
      <DownloadMenuItem
        icon={<Sheet size={13} />}
        label="Excel (.xlsx)"
        disabled
        title="Coming in v2"
      />
    </div>
  )
}

function DownloadMenuItem({
  icon,
  label,
  onClick,
  disabled,
  title,
}: {
  icon: React.ReactNode
  label: string
  onClick?: () => void
  disabled?: boolean
  title?: string
}) {
  const base =
    'w-full flex items-center gap-2 px-3 py-1.5 text-left transition-colors'
  const style = disabled
    ? 'opacity-50 cursor-not-allowed text-slate-400'
    : 'text-slate-200 hover:bg-purple-500/15 hover:text-purple-100'
  return (
    <button
      type="button"
      role="menuitem"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      aria-disabled={disabled}
      title={title}
      className={`${base} ${style}`}
    >
      {icon}
      <span>{label}</span>
    </button>
  )
}

// -----------------------------------------------------------------------------
// Inputs section (shared shape with AgentResultsView — kept inline in both
// files per MVP directive; extract if a third consumer appears).
//
// Renders the exact values a user submitted for this run. The values sit in
// the store as `submittedInputs: Record<string, unknown>` — a passthrough of
// the form payload sent to /api/reports/generate. We pair them here with the
// report's inputSchema so each row shows the human label + a nicely-rendered
// value, not the raw slug + raw form value.
//
// Interaction:
//   • Single-field schemas auto-expand to a body-styled paragraph. There is
//     no toggle — for a single-input recipe the value IS the summary.
//   • Multi-field schemas default collapsed. The header shows a one-line
//     "Inputs · Label: value · Label: value" summary of only the filled
//     fields. Click the chevron / header to expand into a dt/dd-style list
//     that includes empty fields (rendered as em-dash) so users can see the
//     full form state, not just what was populated.
//   • State is per-render — reopening the toggle on every result view is
//     acceptable; no need to persist to the store.
// -----------------------------------------------------------------------------

// Coerce a metadata value to a finite non-negative integer count, else null.
// Metadata values are `unknown` because the response envelope is a bag.
function _numOrNull(v: unknown): number | null {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) return null
  return Math.round(v)
}

// Read a "scanned N patterns from M sources" footprint from the response
// metadata. Kept keyname-tolerant because different reports populate
// slightly different keys (patterns_retrieved / patterns_sampled /
// live_pattern_count / pattern_count; retrieved_source_count /
// unique_source_count). Reports don't carry a top-level cited_patterns
// list, so the pattern count relies entirely on metadata. Returns null
// (line hidden) if either count is missing or zero — silent absence is
// better than "scanned 0 patterns from 0 sources".
function _getRetrievalFootprint(
  metadata: Record<string, unknown> | undefined | null,
): { patterns: number; sources: number } | null {
  const meta = metadata || {}
  const patterns =
    _numOrNull(meta.retrieved_pattern_count) ??
    _numOrNull(meta.patterns_retrieved) ??
    _numOrNull(meta.patterns_sampled) ??
    _numOrNull(meta.live_pattern_count) ??
    _numOrNull(meta.pattern_count)
  const sources =
    _numOrNull(meta.retrieved_source_count) ??
    _numOrNull(meta.unique_source_count)
  if (patterns === null || sources === null) return null
  if (patterns <= 0 || sources <= 0) return null
  return { patterns, sources }
}

function _isEmptyValue(v: unknown): boolean {
  if (v === null || v === undefined) return true
  if (typeof v === 'string' && v === '') return true
  if (typeof v === 'object') {
    const dr = v as { from?: unknown; to?: unknown }
    if ('from' in dr || 'to' in dr) {
      const from = typeof dr.from === 'string' ? dr.from : ''
      const to = typeof dr.to === 'string' ? dr.to : ''
      return !from && !to
    }
  }
  return false
}

function _formatValue(field: FieldSchema, v: unknown): string {
  if (_isEmptyValue(v)) return '—'
  if (field.type === 'date-range') {
    const dr = v as { from?: string; to?: string }
    const from = dr.from || '(any)'
    const to = dr.to || '(any)'
    if (from === '(any)' && to === '(any)') return '(any)'
    return `${from} → ${to}`
  }
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  return String(v)
}

function InputsSection({
  schema,
  values,
  accent,
}: {
  schema: FieldSchema[]
  values: Record<string, unknown>
  accent: 'purple' | 'emerald'
}) {
  if (!schema || schema.length === 0) return null

  const singleField = schema.length === 1 ? schema[0] : null
  const singleFieldValue = singleField ? values[singleField.name] : undefined
  const singleFieldFilled = singleField && !_isEmptyValue(singleFieldValue)

  const [expanded, setExpanded] = useState(false)

  const accentText = accent === 'purple' ? 'text-purple-300' : 'text-emerald-300'
  const accentHover = accent === 'purple' ? 'hover:text-purple-200' : 'hover:text-emerald-200'
  const accentBorder = accent === 'purple' ? 'border-purple-500/30' : 'border-emerald-500/30'
  const accentDim = accent === 'purple' ? 'text-purple-300/70' : 'text-emerald-300/70'

  if (singleField && singleFieldFilled) {
    return (
      <div className={`mx-4 mt-3 mb-1 rounded-lg border ${accentBorder} bg-slate-900/30 px-4 py-3`}>
        <div className={`text-[10px] uppercase tracking-wider font-medium ${accentDim} mb-1.5`}>
          {singleField.label}
        </div>
        <div className="text-[13px] text-slate-200 italic leading-relaxed whitespace-pre-wrap">
          {String(singleFieldValue ?? '')}
        </div>
      </div>
    )
  }

  const filledPairs = schema
    .filter((f) => !_isEmptyValue(values[f.name]))
    .map((f) => `${f.label}: ${_formatValue(f, values[f.name])}`)
  const summaryText =
    filledPairs.length > 0 ? filledPairs.join(' · ') : '(no values filled)'

  return (
    <div className={`mx-4 mt-3 mb-1 rounded-lg border ${accentBorder} bg-slate-900/30`}>
      <button
        type="button"
        onClick={() => setExpanded((x) => !x)}
        aria-expanded={expanded}
        className={`w-full flex items-center gap-2 px-4 py-2 text-left ${accentText} ${accentHover} transition-colors`}
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="text-[11px] uppercase tracking-wider font-medium">Inputs</span>
        {!expanded && (
          <span className="text-[11px] text-slate-400 font-normal normal-case tracking-normal truncate">
            <span className="text-slate-600 mx-1">·</span>
            <span className="text-slate-400">{summaryText}</span>
          </span>
        )}
      </button>
      {expanded && (
        <dl className="px-4 pb-3 pt-1 grid grid-cols-[minmax(0,180px)_1fr] gap-x-4 gap-y-1 text-[12px]">
          {schema.map((field) => {
            const raw = values[field.name]
            const empty = _isEmptyValue(raw)
            return (
              <div key={field.name} className="contents">
                <dt className="text-slate-500 truncate" title={field.label}>{field.label}</dt>
                <dd className={empty ? 'text-slate-600 italic' : 'text-slate-200 whitespace-pre-wrap break-words'}>
                  {_formatValue(field, raw)}
                </dd>
              </div>
            )
          })}
        </dl>
      )}
    </div>
  )
}
