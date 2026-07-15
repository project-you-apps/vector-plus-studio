import { useMemo, useRef, useState, type ReactNode } from 'react'
import { X, Sparkles, Info, Send, Loader2, AlertTriangle, RefreshCw, Lock, Clock } from 'lucide-react'
import type { AgentDefinition, FieldSchema } from '../agents/agent-definitions'
import { useAppStore } from '../store/appStore'
import {
  runAgent,
  RunAgentError,
  type RunAgentResponse,
} from '../api/client'

// Slide-in pane on the right when an agent card's Send button is clicked.
// Mirror shape of ReportInputPane — schema-driven form, loading spinner
// while the LLM synthesizes, error panels with recovery affordances. The
// success branch hoists the response to the parent via onSuccess() so
// results render full-width in AgentResultsView (matches Reports Option 3).
//
// Quota exception is agent-specific: a 429 quota_exceeded response renders
// a friendly purple "come back tomorrow / upgrade" panel showing the
// reset_at time.

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

// Auto-generated form field. Style shared with ReportInputPane so both
// panes feel visually identical.
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
          className={`${baseInput} min-h-[120px] resize-none`}
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

type ResultState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | {
      kind: 'error'
      status: number
      error: string
      message: string
      resetAt?: string
    }

function normalizeCartRef(cartRef: string): string {
  if (cartRef.startsWith('server:')) return cartRef.slice('server:'.length)
  return cartRef
}

function isLocalCart(cartRef: string | null): boolean {
  return !!cartRef && cartRef.startsWith('local:')
}

function buildInputsForApi(
  schema: FieldSchema[],
  values: InputValues,
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const field of schema) {
    const v = values[field.name]
    if (!isFilled(field, v)) continue
    out[field.name] = v
  }
  return out
}

export default function AgentInputPane({
  agent,
  cartName,
  cartRef,
  initialInputs,
  reportBuilderPaired = false,
  onClose,
  onSuccess,
  onPickAnotherCart,
}: {
  agent: AgentDefinition
  cartName: string | null
  cartRef: string | null
  initialInputs?: Record<string, unknown> | null
  // When true, local: carts route to the paired Report Builder on
  // localhost:7880 instead of failing. Gates both the amber warning
  // and the Send-button disable.
  reportBuilderPaired?: boolean
  onClose: () => void
  onSuccess?: (
    response: RunAgentResponse,
    submittedInputs: Record<string, unknown>,
  ) => void
  onPickAnotherCart?: () => void
}) {
  const sessionId = useAppStore((s) => s.agentSessionId)

  const [values, setValues] = useState<InputValues>(() => {
    const base = initialValues(agent.inputSchema)
    if (!initialInputs) return base
    const merged: InputValues = { ...base }
    for (const field of agent.inputSchema) {
      const v = initialInputs[field.name]
      if (v === undefined) continue
      if (field.type === 'date-range') {
        const dr = v as { from?: string; to?: string } | null
        merged[field.name] = { from: dr?.from ?? '', to: dr?.to ?? '' }
      } else {
        merged[field.name] = v as FieldValue
      }
    }
    return merged
  })
  const [result, setResult] = useState<ResultState>({ kind: 'idle' })

  const missingRequired = useMemo(
    () => agent.inputSchema.filter((f) => f.required && !isFilled(f, values[f.name])),
    [agent.inputSchema, values],
  )
  const canSend = missingRequired.length === 0 && !!cartRef && (!isLocalCart(cartRef) || reportBuilderPaired)

  const lastRequestRef = useRef<{
    slug: string
    cart_ref: string
    inputs: Record<string, unknown>
  } | null>(null)

  const dispatch = async (payload: {
    slug: string
    cart_ref: string
    inputs: Record<string, unknown>
  }) => {
    setResult({ kind: 'loading' })
    try {
      const res = await runAgent({
        agent_slug: payload.slug,
        cart_ref: payload.cart_ref,
        inputs: payload.inputs,
        session_id: sessionId,
      })
      setResult({ kind: 'idle' })
      onSuccess?.(res, payload.inputs)
    } catch (err) {
      if (err instanceof RunAgentError) {
        setResult({
          kind: 'error',
          status: err.status,
          error: err.detail.error || 'unknown_error',
          message: err.detail.message || err.message,
          resetAt: err.detail.reset_at,
        })
      } else {
        setResult({
          kind: 'error',
          status: 0,
          error: 'network_error',
          message: err instanceof Error ? err.message : 'Unable to reach the server.',
        })
      }
    }
  }

  const handleSend = () => {
    if (!canSend || !cartRef) return
    const payload = {
      slug: agent.name,
      cart_ref: normalizeCartRef(cartRef),
      inputs: buildInputsForApi(agent.inputSchema, values),
    }
    lastRequestRef.current = payload
    dispatch(payload)
  }

  const handleTryAgain = () => {
    if (lastRequestRef.current) dispatch(lastRequestRef.current)
  }

  const handleEditInputs = () => {
    setResult({ kind: 'idle' })
    lastRequestRef.current = null
  }

  return (
    <div
      role="dialog"
      aria-label={`${agent.displayName} inputs`}
      className="fixed inset-y-0 right-0 w-full sm:w-[440px] z-30
                 border-l border-slate-700 bg-slate-900/98 backdrop-blur
                 flex flex-col shadow-2xl animate-in slide-in-from-right duration-200"
    >
      {/* Header */}
      <div className="flex items-start gap-3 px-5 py-4 border-b border-slate-800">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-slate-100">{agent.displayName}</h2>
            {agent.llmDependency && (
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
          <p className="text-xs text-slate-400 mt-1 leading-snug">{agent.description}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800
                     transition-colors shrink-0"
          aria-label="Close"
          title="Back to agent grid"
        >
          <X size={16} />
        </button>
      </div>

      {/* Body */}
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

            {isLocalCart(cartRef) && !reportBuilderPaired && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2
                              text-[11px] text-amber-200 flex items-start gap-2">
                <AlertTriangle size={12} className="shrink-0 mt-0.5" />
                <span>
                  Agents run server-side. Pick a server cart in the cart selector,
                  OR launch Report Builder to run this agent locally against your
                  browser-only LocalCart.
                </span>
              </div>
            )}

            {agent.inputSchema.length === 0 ? (
              <p className="text-xs text-slate-500 italic text-center py-6">
                This agent takes no inputs.
              </p>
            ) : (
              agent.inputSchema.map((field) => (
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

        {result.kind === 'loading' && <LoadingPanel agent={agent} />}

        {result.kind === 'error' && (
          <ErrorPanel
            agent={agent}
            status={result.status}
            errorTag={result.error}
            cartName={cartName}
            message={result.message}
            resetAt={result.resetAt}
            onTryAgain={handleTryAgain}
            onEditInputs={handleEditInputs}
            onPickAnotherCart={onPickAnotherCart}
          />
        )}
      </div>

      {/* Footer — SEND button */}
      {result.kind === 'idle' && (
        <div className="px-5 py-4 border-t border-slate-800 bg-slate-900/60">
          {missingRequired.length > 0 && (
            <p className="text-[10px] text-amber-400 mb-2 leading-snug">
              Missing required: {missingRequired.map((f) => f.label).join(', ')}
            </p>
          )}
          <button
            onClick={handleSend}
            disabled={!canSend}
            className={`w-full flex items-center justify-center gap-2 px-4 py-2.5
                        rounded-lg text-sm font-medium transition-colors ${
                          canSend
                            ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-200 hover:bg-emerald-500/30'
                            : 'bg-slate-800/60 border border-slate-700 text-slate-600 cursor-not-allowed'
                        }`}
          >
            <Send size={14} />
            Send
          </button>
        </div>
      )}
    </div>
  )
}

function LoadingPanel({ agent }: { agent: AgentDefinition }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <Loader2 size={28} className="text-emerald-300 animate-spin" />
      <p className="text-sm text-slate-200 font-medium">Running {agent.displayName}…</p>
      <p className="text-[11px] text-slate-500 max-w-[280px] leading-snug">
        Reading the cart, retrieving relevant passages, and calling the language
        model. This usually takes a few seconds.
      </p>
    </div>
  )
}

// Error panel — cart-availability variants (amber, mirrors Reports) +
// quota_exceeded (purple, agent-specific because Reports doesn't have a
// neuron cap) + generic red for everything else.
function ErrorPanel({
  agent,
  status,
  errorTag,
  cartName,
  message,
  resetAt,
  onTryAgain,
  onEditInputs,
  onPickAnotherCart,
}: {
  agent: AgentDefinition
  status: number
  errorTag: string
  cartName: string | null
  message: string
  resetAt?: string
  onTryAgain: () => void
  onEditInputs: () => void
  onPickAnotherCart?: () => void
}) {
  const isLocalCartErr = errorTag === 'local_cart_unsupported'
  const isCartNotFound = errorTag === 'cart_not_found'
  const isLegacyCart = errorTag === 'cart_legacy_format'
  const isSandboxExpired = errorTag === 'sandbox_cart_expired'
  const isQuota = status === 429 || errorTag === 'quota_exceeded'
  const isCartAvailabilityIssue = isCartNotFound || isLegacyCart || isSandboxExpired

  if (isQuota) {
    const resetDisplay = resetAt
      ? new Date(resetAt).toLocaleString()
      : 'tomorrow at midnight UTC'
    return (
      <div className="flex flex-col gap-4">
        <div className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-4 text-sm">
          <p className="text-purple-200 font-medium mb-2 flex items-center gap-2">
            <Clock size={14} />
            Daily agent budget reached
          </p>
          <p className="text-xs text-slate-300 leading-relaxed">
            {message || "You've used your daily allotment of agent runs."}
          </p>
          <p className="text-[11px] text-slate-500 leading-snug mt-2">
            Resets at <span className="font-mono text-slate-300">{resetDisplay}</span>.
            Upgrade for a higher tier.
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

  if (isCartAvailabilityIssue) {
    const cartLabel = cartName || `'${message.match(/'([^']+)'/)?.[1] ?? 'this cart'}'`
    let heading = 'Cart not available'
    let bodyCopy: ReactNode = (
      <>
        {cartLabel} isn&rsquo;t available on the server anymore. Pick another cart above to continue.
      </>
    )
    if (isLegacyCart) {
      heading = 'Legacy cart format'
      bodyCopy = (
        <>
          {cartLabel} uses a legacy format that Agents can&rsquo;t read yet.
          Rebuild it via Cart Builder &rarr; Save as{' '}
          <span className="font-mono text-amber-200">.cart.npz</span>, then try again.
        </>
      )
    } else if (isSandboxExpired) {
      heading = 'Sandbox cart expired'
      bodyCopy = (
        <>
          This sandbox cart expired before the agent finished. Re-upload it to try again.
        </>
      )
    }
    return (
      <div className="flex flex-col gap-4">
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
          <p className="text-amber-200 font-medium mb-2 flex items-center gap-2">
            <Lock size={14} />
            {heading}
          </p>
          <p className="text-xs text-slate-300 leading-relaxed">
            {bodyCopy}
          </p>
        </div>
        <div className="flex flex-col gap-2">
          {onPickAnotherCart && (
            <button
              onClick={onPickAnotherCart}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                         text-sm text-amber-100 border border-amber-500/40 bg-amber-500/10
                         hover:bg-amber-500/20 transition-colors"
            >
              Pick another cart
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

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4">
        <p className="text-rose-200 font-medium mb-2 flex items-center gap-2 text-sm">
          <AlertTriangle size={14} />
          {status === 0 ? 'Network error' : `Error ${status}`}
        </p>
        <p className="text-xs text-slate-200 leading-relaxed break-words">
          {message || `${agent.displayName} run failed.`}
        </p>
        {errorTag && (
          <p className="text-[10px] text-slate-500 font-mono mt-2">error: {errorTag}</p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        {!isLocalCartErr && (
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
