import { useRef, useState } from 'react'
import {
  X, RefreshCw, Copy, Check, ExternalLink, Clock3, AlertTriangle,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { GenerateReportResponse } from '../api/client'
import type { ReportDefinition } from '../reports/report-definitions'

// Full-width results view that REPLACES the reports grid when a report has
// been generated (Andy 2026-07-13 design — Option 3). Lives inside the main
// content column so `max-w-6xl mx-auto` from ReportsScreen already caps its
// width sensibly on ultra-wide monitors.
//
// Toolbar buttons: Close (dismiss), Regenerate (re-open the input pane
// pre-filled with the last inputs), Copy markdown, Open in new tab. The
// "Open in new tab" button is a coming-soon placeholder for now — the URL
// routing story is a follow-up. Right side of the toolbar shows the cart
// name + report display name + generation-time badge.
//
// Body renders the markdown response full-width via react-markdown +
// remark-gfm (same override map as the old SuccessPanel inside
// ReportInputPane) so long tables don't get squeezed like they did in the
// 440px slide-in pane.

export default function ReportResultsView({
  report,
  response,
  cartLabel,
  onClose,
  onRegenerate,
}: {
  report: ReportDefinition
  response: GenerateReportResponse
  cartLabel: string | null
  onClose: () => void
  onRegenerate: () => void
}) {
  const [justCopied, setJustCopied] = useState(false)
  const copyTimerRef = useRef<number | null>(null)

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

  // Prefer the report author's own elapsed_ms; fall back to the route
  // fallback in metadata.route_elapsed_ms. Both are integers in ms.
  const meta = response.metadata || {}
  const elapsed =
    typeof meta.elapsed_ms === 'number'
      ? meta.elapsed_ms as number
      : typeof meta.route_elapsed_ms === 'number'
        ? meta.route_elapsed_ms as number
        : null

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
        </div>
      </div>

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
              a: ({ href, children }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan-400 hover:text-cyan-300 underline decoration-cyan-400/40"
                >
                  {children}
                </a>
              ),
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
// toward it as the "primary next thing you can do" after Close.
function ToolbarButton({
  icon,
  label,
  onClick,
  title,
  highlight,
  active,
  disabled,
}: {
  icon: React.ReactNode
  label: string
  onClick: () => void
  title?: string
  highlight?: boolean
  active?: boolean
  disabled?: boolean
}) {
  const base =
    'flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium border transition-colors'
  const style = disabled
    ? 'bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed'
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
    >
      {icon}
      {label}
    </button>
  )
}
