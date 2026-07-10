import { useMemo, useState } from 'react'
import { X, Sparkles, Info, PlayCircle } from 'lucide-react'
import type { FieldSchema, ReportDefinition } from '../reports/report-definitions'

// Slide-in pane on the right when a card's Run button is clicked. Renders a
// form auto-generated from the report's input_schema and, on Generate, shows
// a placeholder message describing what WOULD have been dispatched — because
// the backend report modules that actually run reports are future work
// (design doc §0.2, waves 1-3).

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

export default function ReportInputPane({
  report,
  cartName,
  onClose,
}: {
  report: ReportDefinition
  cartName: string | null
  onClose: () => void
}) {
  const [values, setValues] = useState<InputValues>(() => initialValues(report.inputSchema))
  const [generated, setGenerated] = useState(false)

  const missingRequired = useMemo(
    () => report.inputSchema.filter((f) => f.required && !isFilled(f, values[f.name])),
    [report.inputSchema, values],
  )
  const canGenerate = missingRequired.length === 0

  const handleGenerate = () => {
    if (!canGenerate) return
    setGenerated(true)
  }

  // Formatted inputs for the placeholder message. Empty date-ranges and
  // blank fields collapse so the output is compact.
  const displayInputs = useMemo(() => {
    const out: Record<string, string | number | { from: string; to: string }> = {}
    for (const field of report.inputSchema) {
      const v = values[field.name]
      if (!isFilled(field, v)) continue
      out[field.name] = v
    }
    return out
  }, [report.inputSchema, values])

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

      {/* Body — either the form or the post-Generate placeholder */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {!generated ? (
          <div className="flex flex-col gap-4">
            <div className="rounded-lg border border-slate-700/60 bg-slate-800/30 px-3 py-2
                            text-[11px] text-slate-400 flex items-start gap-2">
              <Info size={12} className="text-emerald-300 shrink-0 mt-0.5" />
              <span>
                Cart: <span className="font-mono text-slate-200">{cartName ?? '(none mounted)'}</span>
              </span>
            </div>

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
        ) : (
          <PlaceholderResult
            report={report}
            cartName={cartName}
            inputs={displayInputs}
            onEditInputs={() => setGenerated(false)}
          />
        )}
      </div>

      {/* Footer — Generate button (hidden after generation, replaced by
          Edit Inputs affordance inside PlaceholderResult) */}
      {!generated && (
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

// Post-Generate placeholder — the backend modules that actually run reports
// are future work (see design doc §0.1, waves 1-3). Until they land, this
// pane echoes what WOULD have been dispatched.
function PlaceholderResult({
  report,
  cartName,
  inputs,
  onEditInputs,
}: {
  report: ReportDefinition
  cartName: string | null
  inputs: Record<string, string | number | { from: string; to: string }>
  onEditInputs: () => void
}) {
  const inputsJson = JSON.stringify(inputs, null, 2)
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
        <p className="text-amber-200 font-medium mb-2">
          Report generation is not yet available in this build.
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          Backend report modules are next in the roadmap. This tab is the UX shell —
          buttons wire up, forms validate, but the compute side is unwired.
        </p>
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-4 space-y-3">
        <div className="text-[10px] uppercase tracking-wider text-slate-500">
          Would have generated
        </div>
        <div className="text-xs text-slate-300 space-y-1.5">
          <div>
            <span className="text-slate-500">Report:</span>{' '}
            <span className="font-mono text-emerald-300">{report.displayName}</span>
          </div>
          <div>
            <span className="text-slate-500">Cart:</span>{' '}
            <span className="font-mono text-emerald-300">{cartName ?? '(none mounted)'}</span>
          </div>
          <div className="text-slate-500">Inputs:</div>
          <pre className="font-mono text-[11px] text-slate-300 bg-slate-900/70
                          border border-slate-800 rounded-md px-3 py-2 overflow-x-auto">
{inputsJson || '{}'}
          </pre>
        </div>
      </div>

      <button
        onClick={onEditInputs}
        className="w-full px-3 py-2 rounded-lg text-sm text-slate-300 border border-slate-700
                   hover:border-slate-500 hover:bg-slate-800/40 transition-colors"
      >
        Edit inputs
      </button>
    </div>
  )
}
