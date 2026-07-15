import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot, Database, ChevronDown, Lock, Zap,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { AGENT_DEFINITIONS, type AgentDefinition } from '../agents/agent-definitions'
import AgentCard from './AgentCard'
import AgentInputPane from './AgentInputPane'
import AgentResultsView from './AgentResultsView'
import {
  fetchReportCarts, fetchLocalReportCarts,
  type ReportCartEntry, type RunAgentResponse,
} from '../api/client'
import ReportBuilderStatusPill from './ReportBuilderStatusPill'

// Agents screen — the UX shell for the 4 v1 agent recipes (Auto-Briefing,
// Q&A, Professor, Cart Curator). Structure mirrors ReportsScreen exactly:
//
//   • Recipe grid at the top (matched with ReportsScreen's card grid)
//   • Slide-in input pane on card click (matched with ReportInputPane)
//   • Full-width results view on Send success (matched with ReportResultsView)
//
// One user learns ONE pattern, knows both screens. See the AGENT-LOGBOOK
// entry filed 2026-07-13 for the design rationale.
//
// Cart selection reuses the same /api/reports/carts endpoint as Reports —
// agents run against .cart.npz files server-side, exactly like Reports do,
// so the compatibility rule is identical. LocalCarts are annotated
// incompatible (browser-only, agent runs server-side).
//
// Persistence: currentAgentRun in appStore — mirrors currentReport invariant.
// Only Close clears; tab switches preserve. Same INVARIANT enforcement
// (type-locked non-null setter, single clearer path).

export default function AgentsScreen() {
  const mountedCartridge = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const cartridges = useAppStore((s) => s.cartridges)
  const localCarts = useAppStore((s) => s.localCarts)

  const detectReportBuilder = useAppStore((s) => s.detectReportBuilder)
  const reportBuilderState = useAppStore((s) => s.reportBuilderState)
  const reportBuilderPaired = reportBuilderState === 'detected-paired'

  const [cartCompat, setCartCompat] = useState<Map<string, ReportCartEntry>>(new Map())
  const [localReportBuilderCarts, setLocalReportBuilderCarts] = useState<ReportCartEntry[]>([])
  const cartSelectorButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    void detectReportBuilder()
  }, [detectReportBuilder])

  useEffect(() => {
    let cancelled = false
    fetchReportCarts()
      .then((entries) => {
        if (cancelled) return
        const map = new Map<string, ReportCartEntry>()
        for (const e of entries) map.set(e.id, e)
        setCartCompat(map)
      })
      .catch(() => { /* silent fallback — pane surfaces cart_not_found */ })
    return () => { cancelled = true }
  }, [cartridges])

  useEffect(() => {
    let cancelled = false
    if (!reportBuilderPaired) {
      setLocalReportBuilderCarts([])
      return () => { cancelled = true }
    }
    fetchLocalReportCarts()
      .then((entries) => {
        if (cancelled) return
        setLocalReportBuilderCarts(entries ?? [])
      })
      .catch(() => {
        if (!cancelled) setLocalReportBuilderCarts([])
      })
    return () => { cancelled = true }
  }, [reportBuilderPaired])

  const cartOptions = useMemo(() => {
    const opts: CartOption[] = []
    const seen = new Set<string>()
    const push = (o: CartOption) => {
      if (seen.has(o.id)) return
      seen.add(o.id)
      opts.push(o)
    }
    // Browser LocalCarts: report_compatible flips true when Report
    // Builder is paired — the local exe knows how to open them.
    for (const name of localCarts.keys()) {
      push({
        id: `local:${name}`,
        label: name,
        kind: 'local',
        reportCompatible: reportBuilderPaired,
        format: 'npz',
        location: 'local',
      })
    }
    // Report Builder's own /reports/carts enumeration — carts sitting
    // in the user's cart folder that aren't browser-mounted.
    for (const meta of localReportBuilderCarts) {
      push({
        id: `local:${meta.id}`,
        label: meta.display_name,
        kind: 'local',
        reportCompatible: true,
        format: (meta.format as 'npz' | 'pkl') ?? 'npz',
        location: 'local',
      })
    }
    for (const c of cartridges) {
      const meta = cartCompat.get(c.name)
      push({
        id: `server:${c.name}`,
        label: meta?.display_name ?? c.name,
        kind: 'server',
        reportCompatible: meta?.report_compatible ?? true,
        format: (meta?.format as 'npz' | 'pkl' | undefined) ?? 'npz',
        location: (meta?.location as 'canonical' | 'sandbox' | undefined) ?? 'canonical',
      })
    }
    for (const meta of cartCompat.values()) {
      if (meta.location !== 'sandbox') continue
      push({
        id: `server:${meta.id}`,
        label: meta.display_name,
        kind: 'server',
        reportCompatible: meta.report_compatible,
        format: (meta.format as 'npz' | 'pkl' | undefined) ?? 'npz',
        location: 'sandbox',
      })
    }
    return opts
  }, [cartridges, localCarts, cartCompat, localReportBuilderCarts, reportBuilderPaired])

  const defaultCartId = useMemo(() => {
    if (activeLocalCart) return `local:${activeLocalCart}`
    if (mountedCartridge) return `server:${mountedCartridge}`
    const firstCompat = cartOptions.find((o) => o.reportCompatible)
    return firstCompat?.id ?? cartOptions[0]?.id ?? null
  }, [activeLocalCart, mountedCartridge, cartOptions])

  const compatibleCartOptions = useMemo(
    () => cartOptions.filter((o) => o.reportCompatible),
    [cartOptions],
  )

  const [selectedCartId, setSelectedCartId] = useState<string | null>(defaultCartId)
  const [activeAgent, setActiveAgent] = useState<AgentDefinition | null>(null)
  const [prefillInputs, setPrefillInputs] = useState<Record<string, unknown> | null>(null)

  const currentAgentRun = useAppStore((s) => s.currentAgentRun)
  const setCurrentAgentRun = useAppStore((s) => s.setCurrentAgentRun)
  const clearCurrentAgentRun = useAppStore((s) => s.clearCurrentAgentRun)

  const resultViewAgent = useMemo(
    () =>
      currentAgentRun
        ? AGENT_DEFINITIONS.find((a) => a.name === currentAgentRun.agentSlug) ?? null
        : null,
    [currentAgentRun],
  )
  // Same fallback pattern as ReportsScreen — synthesize a minimal
  // AgentDefinition if the slug lookup ever fails so the results view
  // survives dev HMR or slug renames.
  const displayedAgent: AgentDefinition | null = currentAgentRun
    ? (resultViewAgent ?? {
        name: currentAgentRun.agentSlug,
        displayName: currentAgentRun.agentDisplayName,
        description: '',
        icon: 'Bot',
        llmDependency: true,
        inputSchema: [],
      })
    : null
  const hasResultView = !!currentAgentRun && !!displayedAgent

  const effectiveCartId = selectedCartId ?? defaultCartId
  const selectedCart = effectiveCartId
    ? cartOptions.find((o) => o.id === effectiveCartId) ?? null
    : null
  const selectedCartLabel = selectedCart?.label ?? null

  const handleOpenCard = (agent: AgentDefinition) => {
    setPrefillInputs(null)
    setActiveAgent(agent)
  }

  const handleSendSuccess = useCallback(
    (response: RunAgentResponse, submittedInputs: Record<string, unknown>) => {
      if (!activeAgent) return
      const cartRefFromMeta = response.metadata && typeof response.metadata.cart_ref === 'string'
        ? (response.metadata.cart_ref as string)
        : (effectiveCartId ?? selectedCartLabel ?? '')
      setCurrentAgentRun({
        cartRef: cartRefFromMeta,
        cartLabel: selectedCartLabel,
        agentSlug: activeAgent.name,
        agentDisplayName: activeAgent.displayName,
        submittedInputs,
        response,
        generatedAt: response.generated_at,
      })
      setActiveAgent(null)
      setPrefillInputs(null)
    },
    [activeAgent, selectedCartLabel, effectiveCartId, setCurrentAgentRun],
  )

  const handleCloseResults = () => clearCurrentAgentRun()

  const handleRegenerate = () => {
    if (!currentAgentRun || !resultViewAgent) return
    setPrefillInputs(currentAgentRun.submittedInputs)
    setActiveAgent(resultViewAgent)
    // currentAgentRun stays in place — user can visually reference the
    // previous output while tweaking inputs. Overwritten on next Send.
  }

  const focusCartSelector = () => {
    setActiveAgent(null)
    setPrefillInputs(null)
    requestAnimationFrame(() => {
      cartSelectorButtonRef.current?.focus()
    })
  }

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto relative">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Header — same shape as ReportsScreen's header, purple accents
            instead of emerald to signal the LLM-first surface. */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
              <Bot size={28} className="text-purple-300" />
              Agents
            </h1>
            <p className="text-sm text-slate-500">
              Scoped agent recipes over your cart. Pick one, describe the task, hit Send.
            </p>
          </div>

          {compatibleCartOptions.length === 0 ? (
            <AgentsEmptyCartState />
          ) : (
            <CartSelector
              options={compatibleCartOptions}
              selectedId={effectiveCartId}
              onSelect={setSelectedCartId}
              buttonRef={cartSelectorButtonRef}
            />
          )}
        </div>

        {/* Report Builder pill — same visual language as ReportsScreen.
            When paired, local: carts route to 127.0.0.1:7880 so agent
            runs against on-disk carts don't leave the machine. */}
        <ReportBuilderStatusPill
          state={reportBuilderState}
          onRecheck={() => void detectReportBuilder()}
        />

        {hasResultView && displayedAgent && currentAgentRun ? (
          <AgentResultsView
            agent={displayedAgent}
            response={currentAgentRun.response}
            cartLabel={currentAgentRun.cartLabel}
            submittedInputs={currentAgentRun.submittedInputs}
            onClose={handleCloseResults}
            onRegenerate={handleRegenerate}
          />
        ) : (
          <>
            {/* Card grid — same 3/2/1 responsive layout as Reports */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {AGENT_DEFINITIONS.map((agent) => (
                <AgentCard
                  key={agent.name}
                  agent={agent}
                  onRun={() => handleOpenCard(agent)}
                />
              ))}
            </div>

            {/* Recent runs — empty state for now. When the "your own memory"
                loop lands (save-to-cart real Membot write, v1.5), each run
                becomes a pattern in the user's cart and the recent list is
                a query over their own history. */}
            <section className="rounded-lg border border-slate-700 bg-slate-800/30">
              <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
                <Zap size={13} className="text-slate-500" />
                <h2 className="text-xs uppercase tracking-wider text-slate-500">Recent runs</h2>
              </div>
              <div className="p-6 text-center text-xs text-slate-500 italic">
                Your agent history will appear here after you run one and save it to your cart.
              </div>
            </section>

            <p className="text-center text-[11px] text-slate-600 italic pt-2">
              v1: Auto-Briefing, Q&amp;A, Professor, Cart Curator. v1.5: Watchlist,
              scheduled briefings, agent-chaining, real save-to-cart.
            </p>
          </>
        )}
      </div>

      {activeAgent && (
        <AgentInputPane
          agent={activeAgent}
          cartName={selectedCartLabel}
          cartRef={effectiveCartId}
          initialInputs={prefillInputs}
          onClose={() => setActiveAgent(null)}
          onSuccess={handleSendSuccess}
          onPickAnotherCart={focusCartSelector}
        />
      )}
    </main>
  )
}

// Empty-state for the cart selector when no report-compatible carts exist.
// Same shape + amber palette as ReportsScreen's ReportsEmptyCartState.
function AgentsEmptyCartState() {
  return (
    <div
      className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2
                 text-xs text-amber-200 flex items-start gap-2 max-w-md"
      role="status"
    >
      <Lock size={13} className="shrink-0 mt-0.5 text-amber-400/80" />
      <span>
        No agent-compatible carts available yet. Agents need the .cart.npz
        format (same as Reports). Rebuild legacy carts via Cart Builder →
        Save as .cart.npz, or ask your admin to convert existing carts.
      </span>
    </div>
  )
}

interface CartOption {
  id: string
  label: string
  kind: 'local' | 'server'
  reportCompatible: boolean
  format: 'npz' | 'pkl'
  location?: 'canonical' | 'sandbox' | 'local'
}

function CartKindBadge({
  kind,
  location,
}: {
  kind: 'local' | 'server'
  location?: 'canonical' | 'sandbox' | 'local'
}) {
  const isSandbox = kind === 'server' && location === 'sandbox'
  if (isSandbox) {
    return (
      <span
        className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono
                   bg-amber-500/15 border border-amber-500/40 text-amber-300"
        title="Sandbox upload — expires after a short TTL. Re-upload if you need the cart later."
      >
        sandbox
      </span>
    )
  }
  if (kind === 'local') {
    return (
      <span
        className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono
                   bg-violet-500/15 border border-violet-500/40 text-violet-300"
        title="Local cart — reports + agents run on the paired Report Builder (127.0.0.1:7880). Cart data stays on your machine."
      >
        local
      </span>
    )
  }
  return (
    <span
      className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono
                 bg-purple-500/15 border border-purple-500/40 text-purple-300"
    >
      server
    </span>
  )
}

// Reuses the same visual shape as ReportsScreen's CartSelector. Kept as a
// separate copy rather than lifting to a shared component because the two
// screens are still evolving independently and a shared component would
// need a props explosion to cover both surfaces.
function CartSelector({
  options,
  selectedId,
  onSelect,
  buttonRef,
}: {
  options: CartOption[]
  selectedId: string | null
  onSelect: (id: string) => void
  buttonRef?: React.RefObject<HTMLButtonElement | null>
}) {
  const [open, setOpen] = useState(false)
  const selected = options.find((o) => o.id === selectedId) ?? null

  if (options.length === 0) {
    return (
      <div
        className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2
                   text-xs text-amber-200 flex items-center gap-2"
      >
        <Database size={13} />
        Load a cart in Search to use for an Agent.
      </div>
    )
  }

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-700
                   bg-slate-800/40 hover:bg-slate-800 text-sm text-slate-200
                   min-w-[240px] transition-colors"
        title="Choose which cart the agent runs against"
      >
        <Database size={14} className="text-purple-300 shrink-0" />
        <span className="flex-1 text-left truncate">
          {selected?.label ?? 'Pick a cart…'}
        </span>
        {selected && !selected.reportCompatible && (
          <Lock
            size={12}
            className="text-amber-400/80 shrink-0"
          />
        )}
        {selected && (
          <CartKindBadge kind={selected.kind} location={selected.location} />
        )}
        <ChevronDown size={14} className="text-slate-500 shrink-0" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div
            className="absolute right-0 top-full mt-1 z-20 min-w-[300px]
                       rounded-lg border border-slate-700 bg-slate-900 shadow-xl
                       max-h-[320px] overflow-y-auto"
          >
            {options.map((opt) => {
              const active = opt.id === selectedId
              const incompatible = !opt.reportCompatible
              const tooltip = incompatible
                ? 'This cart uses a legacy format. Agents need the newer .cart.npz format. Convert via Cart Builder → Save as .cart.npz.'
                : undefined
              return (
                <button
                  key={opt.id}
                  onClick={() => {
                    onSelect(opt.id)
                    setOpen(false)
                  }}
                  title={tooltip}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors ${
                    active
                      ? 'bg-purple-500/10 text-purple-200'
                      : incompatible
                        ? 'text-slate-500 hover:bg-slate-800/60'
                        : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {incompatible ? (
                    <Lock size={13} className="text-amber-400/70 shrink-0" />
                  ) : (
                    <Database size={13} className={active ? 'text-purple-300' : 'text-slate-500'} />
                  )}
                  <span className={`flex-1 truncate ${incompatible ? 'italic' : ''}`}>
                    {opt.label}
                  </span>
                  <CartKindBadge kind={opt.kind} location={opt.location} />
                </button>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
