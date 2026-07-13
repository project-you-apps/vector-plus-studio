import { useMemo, useRef, useState } from 'react'
import { X, Sparkles, Info, PlayCircle, Loader2, Copy, Check, AlertTriangle, RefreshCw, Clock3 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { FieldSchema, ReportDefinition } from '../reports/report-definitions'
import {
  generateReport,
  GenerateReportError,
  type GenerateReportResponse,
} from '../api/client'

// Slide-in pane on the right when a card's Run button is clicked. Renders a
// form auto-generated from the report's input_schema, POSTs to
// /api/reports/generate on Generate, and renders the returned markdown via
// react-markdown + remark-gfm. Wave-2 reports (backend returns 501) surface
// as a friendly "future release" message; other errors get a red-tinted
// panel with a Try again button.

type FieldValue = string | number | { from: string; to: string }

interface InputValues {
  [name: string]: FieldValue
}

function initialValues(schema: FieldSchema[]): InputValues {
  const out: InputValues = {}
  for (const field of schema) {
    if (field.type === 'date-range') {
      out[field.name] = { from: '', to: '' }
    } else if (field.default !== undefined) {
      out[field.name] = field.default
    } else if (field.type === 'number') {
      out[field.name] = ''
    } else if (field.type === 'select' && field.options && field.options.length > 0) {
      out[field.name] = field.options[0]
    } else {
      out[field.name] = ''
    }
  }
  return out
}

function isFilled(field: FieldSchema, v: FieldValue): boolean {
  if (field.type === 'date-range') {
    const dr = v as { from: string; to: string }
    return !!(dr.from || dr.to)
  }
  return v !== '' && v !== undefined && v !== null
}

// Renders one form field. Style borrows from CRUDScreen's inputs
// (slate-950/60 fill, slate-800 border, purple focus ring).
function FieldRow({
  field, value, onChange,
}: {
  field: FieldSchema
  value: FieldValue
  onChange: (v: FieldValue) => void
}) {
  const baseInput =
    'w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 ' +
    'text-sm text-slate-200 focus:outline-none focus:border-emerald-500/60'

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium text-slate-200 flex items-center gap-1.5">
        {field.label}
        {field.required && <span className="text-rose-400 text-[10px]">*</span>}
      </label>

      {field.type === 'text' && (
        <input
          type="text"
          className={baseInput}
          value={value as string}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      )}

      {field.type === 'number' && (
        <input
          type="number"
          className={`${baseInput} font-mono`}
          value={value as string | number}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
        />
      )}

      {field.type === 'regex' && (
        <input
          type="text"
          className={`${baseInput} font-mono`}
          value={value as string}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
        />
      )}

      {field.type === 'textarea' && (
        <textarea
          className={`${baseInput} min-h-[80px] resize-none`}
          value={value as string}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      )}

      {field.type === 'select' && field.options && (
        <select
          className={baseInput}
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
        >
          {field.options.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      )}

      {field.type === 'date-range' && (
        <div className="grid grid-cols-2 gap-2">
          <input
            type="date"
            className={baseInput}
            value={(value as { from: string; to: string }).from}
            onChange={(e) => onChange({
              ...(value as { from: string; to: string }),
              from: e.target.value,
            })}
          />
          <input
            type="date"
            className={baseInput}
            value={(value as { from: string; to: string }).to}
            onChange={(e) => onChange({
              ...(value as { from: string; to: string }),
              to: e.target.value,
            })}
          />
        </div>
      )}

      {field.helpText && (
        <p className="text-[10px] text-slate-500 leading-snug">{field.helpText}</p>
      )}
    </div>
  )
}

// Result payload shape used by the pane after Generate. Split into three
// states so the render logic doesn't guess: 'loading' shows a spinner,
// 'success' renders markdown, 'error' shows the red panel + Try again.
type ResultState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'success'; response: GenerateReportResponse }
  | { kind: 'error'; status: number; error: string; message: string }

// Strip cart_ref prefixes ("local:" / "server:") the ReportsScreen selector
// injects so we send the API a clean identifier. The backend also strips
// "server:" defensively but keeping the wire payload clean is polite.
function normalizeCartRef(cartRef: string): string {
  if (cartRef.startsWith('server:')) return cartRef.slice('server:'.length)
  return cartRef
}

// True if the cart the user picked is a browser-only LocalCart. Reports
// require a server-side file; we short-circuit with a clear message
// rather than posting a request that will always 404.
function isLocalCart(cartRef: string | null): boolean {
  return !!cartRef && cartRef.startsWith('local:')
}

// Turn the form's FieldValue map into the untyped inputs dict the backend
// expects. Filters unfilled fields, coerces date ranges into the shape
// report authors read (from / to strings). Called on every Generate click
// so we send the current state, not a stale snapshot.
function buildInputsForApi(
  schema: FieldSchema[],
  values: InputValues,
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const field of schema) {
    const v = values[field.name]
    if (!isFilled(field, v)) continue
    if (field.type === 'date-range') {
      out[field.name] = v
    } else {
      out[field.name] = v
    }
  }
  return out
}

export default function ReportInputPane({
  report,
  cartName,
  cartRef,
  onClose,
}: {
  report: ReportDefinition
  cartName: string | null
  cartRef: string | null
  onClose: () => void
}) {
  const [values, setValues] = useState<InputValues>(() => initialValues(report.inputSchema))
  const [result, setResult] = useState<ResultState>({ kind: 'idle' })

  const missingRequired = useMemo(
    () => report.inputSchema.filter((f) => f.required && !isFilled(f, values[f.name])),
    [report.inputSchema, values],
  )
  const canGenerate = missingRequired.length === 0 && !!cartRef && !isLocalCart(cartRef)

  // Track the last-submitted payload so Try Again re-issues the same
  // request without the user having to re-click Generate on every field.
  // Cleared on Edit inputs so field edits don't silently drift out of
  // sync with what the button will re-post.
  const lastRequestRef = useRef<{
    slug: string
    cart_ref: string
    inputs: Record<string, unknown>
  } | null>(null)

  const dispatchGenerate = async (payload: {
    slug: string
    cart_ref: string
    inputs: Record<string, unknown>
  }) => {
    setResult({ kind: 'loading' })
    try {
      const res = await generateReport({
        report_slug: payload.slug,
        cart_ref: payload.cart_ref,
        inputs: payload.inputs,
      })
      setResult({ kind: 'success', response: res })
    } catch (err) {
      if (err instanceof GenerateReportError) {
        setResult({
          kind: 'error',
          status: err.status,
          error: err.detail.error || 'unknown_error',
          message: err.detail.message || err.message,
        })
      } else {
        // Network / unexpected error — surface as a generic 0 status.
        setResult({
          kind: 'error',
          status: 0,
          error: 'network_error',
          message: err instanceof Error ? err.message : 'Unable to reach the server.',
        })
      }
    }
  }

  const handleGenerate = () => {
    if (!canGenerate || !cartRef) return
    const payload = {
      slug: report.name,
      cart_ref: normalizeCartRef(cartRef),
      inputs: buildInputsForApi(report.inputSchema, values),
    }
    lastRequestRef.current = payload
    dispatchGenerate(payload)
  }

  const handleTryAgain = () => {
    if (lastRequestRef.current) {
      dispatchGenerate(lastRequestRef.current)
    }
  }

  const handleEditInputs = () => {
    setResult({ kind: 'idle' })
    lastRequestRef.current = null
  }

  return (
    // Fixed overlay on the right side. z-30 so it sits above the grid but
    // below toaster / modals (which use z-50).
    <div
      role="dialog"
      aria-label={`${report.displayName} inputs`}
      className="fixed inset-y-0 right-0 w-full sm:w-[440px] z-30
                 border-l border-slate-700 bg-slate-900/98 backdrop-blur
                 flex flex-col shadow-2xl animate-in slide-in-from-right duration-200"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-5 py-4 border-b border-slate-800">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-slate-100">{report.displayName}</h2>
            {report.llmDependency && (
              <span
                className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded
                           bg-purple-500/15 border border-purple-500/40 text-purple-200
                           font-mono flex items-center gap-1"
              >
                <Sparkles size={9} />
                LLM
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1 leading-snug">{report.description}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800
                     transition-colors shrink-0"
          aria-label="Close"
          title="Back to report grid"
        >
          <X size={16} />
        </button>
      </div>

      {/* Body — form (idle) OR loading / success / error result panels */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {result.kind === 'idle' && (
          <div className="flex flex-col gap-4">
            <div className="rounded-lg border border-slate-700/60 bg-slate-800/30 px-3 py-2
                            text-[11px] text-slate-400 flex items-start gap-2">
              <Info size={12} className="text-emerald-300 shrink-0 mt-0.5" />
              <span>
                Cart: <span className="font-mono text-slate-200">{cartName ?? '(none mounted)'}</span>
              </span>
            </div>

            {isLocalCart(cartRef) && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2
                              text-[11px] text-amber-200 flex items-start gap-2">
                <AlertTriangle size={12} className="shrink-0 mt-0.5" />
                <span>
                  Reports run server-side. Pick a server cart in the cart selector to
                  generate this report against a browser-only LocalCart.
                </span>
              </div>
            )}

            {report.inputSchema.length === 0 ? (
              <p className="text-xs text-slate-500 italic text-center py-6">
                This report takes no inputs.
              </p>
            ) : (
              report.inputSchema.map((field) => (
                <FieldRow
                  key={field.name}
                  field={field}
                  value={values[field.name]}
                  onChange={(v) => setValues((prev) => ({ ...prev, [field.name]: v }))}
                />
              ))
            )}
          </div>
        )}

        {result.kind === 'loading' && <LoadingPanel report={report} />}

        {result.kind === 'success' && (
          <SuccessPanel
            report={report}
            response={result.response}
            onEditInputs={handleEditInputs}
          />
        )}

        {result.kind === 'error' && (
          <ErrorPanel
            report={report}
            status={result.status}
            errorTag={result.error}
            message={result.message}
            onTryAgain={handleTryAgain}
            onEditInputs={handleEditInputs}
          />
        )}
      </div>

      {/* Footer — Generate button (hidden after generation) */}
      {result.kind === 'idle' && (
        <div className="px-5 py-4 border-t border-slate-800 bg-slate-900/60">
          {missingRequired.length > 0 && (
            <p className="text-[10px] text-amber-400 mb-2 leading-snug">
              Missing required: {missingRequired.map((f) => f.label).join(', ')}
            </p>
          )}
          <button
            onClick={handleGenerate}
            disabled={!canGenerate}
            className={`w-full flex items-center justify-center gap-2 px-4 py-2.5
                        rounded-lg text-sm font-medium transition-colors ${
                          canGenerate
                            ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-200 hover:bg-emerald-500/30'
                            : 'bg-slate-800/60 border border-slate-700 text-slate-600 cursor-not-allowed'
                        }`}
          >
            <PlayCircle size={14} />
            Generate Report
          </button>
        </div>
      )}
    </div>
  )
}

// Spinner + "Generating..." message. Kept simple; the reports return
// quickly enough (< 1s for the current fixtures) that a progress bar
// would be more noise than signal.
function LoadingPanel({ report }: { report: ReportDefinition }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <Loader2 size={28} className="text-emerald-300 animate-spin" />
      <p className="text-sm text-slate-200 font-medium">Generating {report.displayName}…</p>
      <p className="text-[11px] text-slate-500 max-w-[280px] leading-snug">
        Reading the cart and composing the markdown report. This usually takes
        a second or two.
      </p>
    </div>
  )
}

// Markdown result + Copy button top-right + Edit inputs affordance.
// Uses ReactMarkdown + remark-gfm so tables + fenced code blocks render.
// Component overrides are trimmed down from PassageModal's copy — this
// surface doesn't need graphic:N inline-image resolution, so we keep the
// mapping small enough to review at a glance.
function SuccessPanel({
  report,
  response,
  onEditInputs,
}: {
  report: ReportDefinition
  response: GenerateReportResponse
  onEditInputs: () => void
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
      // Insecure context / permissions denied — silent no-op; user can
      // select the rendered text with their mouse as a fallback.
    }
  }

  // Timing badge — surface the report's own elapsed_ms if the report set
  // it, otherwise the route-level fallback. Both are integers in ms.
  const meta = response.metadata || {}
  const elapsed =
    typeof meta.elapsed_ms === 'number'
      ? meta.elapsed_ms
      : typeof meta.route_elapsed_ms === 'number'
        ? meta.route_elapsed_ms
        : null

  return (
    <div className="flex flex-col gap-3">
      {/* Header row — display name + timing + Copy */}
      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0 text-[11px] text-slate-500 flex items-center gap-2">
          {elapsed !== null && (
            <span className="inline-flex items-center gap-1 font-mono">
              <Clock3 size={11} />
              {elapsed} ms
            </span>
          )}
        </div>
        <button
          onClick={copyMarkdown}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium
                      border transition-colors ${
                        justCopied
                          ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-200'
                          : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800'
                      }`}
          title="Copy raw markdown to clipboard"
        >
          {justCopied ? <Check size={12} /> : <Copy size={12} />}
          {justCopied ? 'Copied' : 'Copy'}
        </button>
      </div>

      {/* Warnings, if the report surfaced any. Empty list = happy path. */}
      {response.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-[11px] text-amber-200 space-y-1">
          <div className="font-medium">Warnings</div>
          <ul className="list-disc pl-4 space-y-0.5">
            {response.warnings.map((w, i) => (<li key={i}>{w}</li>))}
          </ul>
        </div>
      )}

      {/* Rendered markdown. Prose-ish container so tables + headers get
          reasonable spacing; individual component overrides do most of
          the styling to stay consistent with PassageModal. */}
      <div className="rounded-lg border border-slate-700 bg-slate-950/40 px-4 py-3 text-sm text-slate-200 leading-relaxed">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => <h1 className="text-lg font-bold text-slate-100 mt-3 mb-2">{children}</h1>,
            h2: ({ children }) => <h2 className="text-base font-semibold text-slate-100 mt-3 mb-1.5">{children}</h2>,
            h3: ({ children }) => <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-1">{children}</h3>,
            p: ({ children }) => <p className="my-2 text-[13px] text-slate-300">{children}</p>,
            ul: ({ children }) => <ul className="list-disc pl-5 my-2 space-y-1 text-[13px]">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal pl-5 my-2 space-y-1 text-[13px]">{children}</ol>,
            li: ({ children }) => <li className="text-slate-300">{children}</li>,
            code: ({ children }) => (
              <code className="px-1 py-0.5 rounded bg-slate-800 text-amber-200 font-mono text-[11px]">{children}</code>
            ),
            pre: ({ children }) => (
              <pre className="my-2 p-2 rounded-md bg-slate-900/80 border border-slate-800 overflow-x-auto text-[11px] font-mono text-slate-200">{children}</pre>
            ),
            blockquote: ({ children }) => (
              <blockquote className="border-l-2 border-emerald-500/40 pl-3 my-2 italic text-slate-400 text-[13px]">{children}</blockquote>
            ),
            table: ({ children }) => (
              <div className="my-3 overflow-x-auto">
                <table className="border-collapse text-[12px]">{children}</table>
              </div>
            ),
            th: ({ children }) => (
              <th className="border border-slate-700 px-2 py-1 bg-slate-800/60 text-slate-200 text-left font-medium">{children}</th>
            ),
            td: ({ children }) => (
              <td className="border border-slate-700 px-2 py-1 text-slate-300">{children}</td>
            ),
            hr: () => <hr className="my-3 border-slate-700/50" />,
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

      <button
        onClick={onEditInputs}
        className="w-full px-3 py-2 rounded-lg text-sm text-slate-300 border border-slate-700
                   hover:border-slate-500 hover:bg-slate-800/40 transition-colors"
      >
        Edit inputs
      </button>

      {/* Report label footer — helps operators debug "which report
          generated this?" when reviewing exported markdown. */}
      <p className="text-[10px] text-slate-600 text-center">
        {report.displayName} · {response.report_slug} · {new Date(response.generated_at).toLocaleString()}
      </p>
    </div>
  )
}

// Error state. 501 gets a friendly "future release" panel (Wave-2 reports).
// Everything else gets a red-tinted error panel + Try again + Edit inputs.
function ErrorPanel({
  report,
  status,
  errorTag,
  message,
  onTryAgain,
  onEditInputs,
}: {
  report: ReportDefinition
  status: number
  errorTag: string
  message: string
  onTryAgain: () => void
  onEditInputs: () => void
}) {
  const isNotYetAvailable = status === 501 || errorTag === 'not_yet_available'
  const isLocalCart = errorTag === 'local_cart_unsupported'

  if (isNotYetAvailable) {
    return (
      <div className="flex flex-col gap-4">
        <div className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-4 text-sm">
          <p className="text-purple-200 font-medium mb-2 flex items-center gap-2">
            <Sparkles size={14} />
            Coming soon
          </p>
          <p className="text-xs text-slate-300 leading-relaxed">
            {message || `The ${report.displayName} report will be available in a future release.`}
          </p>
          <p className="text-[11px] text-slate-500 leading-snug mt-2">
            The 5 Wave-1 reports (Summary, Entity Rollup, Change Log, Comparison,
            Coverage) are wired now; Timeline, Trend, Financial Rollup, and
            Executive TL;DR are up next.
          </p>
        </div>
        <button
          onClick={onEditInputs}
          className="w-full px-3 py-2 rounded-lg text-sm text-slate-300 border border-slate-700
                     hover:border-slate-500 hover:bg-slate-800/40 transition-colors"
        >
          Back to inputs
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4">
        <p className="text-rose-200 font-medium mb-2 flex items-center gap-2 text-sm">
          <AlertTriangle size={14} />
          {status === 0 ? 'Network error' : `Error ${status}`}
        </p>
        <p className="text-xs text-slate-200 leading-relaxed break-words">
          {message || 'Report generation failed.'}
        </p>
        {errorTag && (
          <p className="text-[10px] text-slate-500 font-mono mt-2">error: {errorTag}</p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        {!isLocalCart && (
          <button
            onClick={onTryAgain}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                       text-sm text-emerald-200 border border-emerald-500/40 bg-emerald-500/10
                       hover:bg-emerald-500/20 transition-colors"
          >
            <RefreshCw size={13} />
            Try again
          </button>
        )}
        <button
          onClick={onEditInputs}
          className="w-full px-3 py-2 rounded-lg text-sm text-slate-300 border border-slate-700
                     hover:border-slate-500 hover:bg-slate-800/40 transition-colors"
        >
          Edit inputs
        </button>
      </div>
    </div>
  )
}
