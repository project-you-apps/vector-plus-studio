import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  FileBarChart, Database, ChevronDown, Clock, CalendarClock, Lock,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { REPORT_DEFINITIONS, type ReportDefinition } from '../reports/report-definitions'
import ReportCard from './ReportCard'
import ReportInputPane from './ReportInputPane'
import ReportResultsView from './ReportResultsView'
import { fetchReportCarts, type GenerateReportResponse, type ReportCartEntry } from '../api/client'

// Reports screen — the UX shell for the 8 report types (see
// docs/vps-internal/Report Types Design 2026-07-10.md).
//
// 2026-07-13 (Andy Stage-B follow-up):
//   • Full-width results display (Option 3). Report replaces the reports
//     grid until closed / regenerated. Fixes the "not wide enough" pain
//     from the 440px slide-in pane when Summary against gutenberg-poetry
//     dumped a ~2000-line markdown blob with tables.
//   • Cart selector filters/annotates report-compatibility. Backend
//     enumerates via GET /api/reports/carts; legacy .pkl carts show as
//     greyed-out with a Lock icon + tooltip. Still selectable (the input
//     pane's amber "legacy cart format" panel is the second line of
//     defense).
//   • Regenerate flow — after a report renders, the Regenerate button
//     in the results toolbar re-opens the input pane pre-filled with the
//     last submitted inputs.

export default function ReportsScreen() {
  // Cart identity — mirror the mounted-cart resolution used by SearchToolbar /
  // CRUDScreen so the selector defaults to whatever the user has active.
  const mountedCartridge = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const cartridges = useAppStore((s) => s.cartridges)
  const localCarts = useAppStore((s) => s.localCarts)

  // Server-side per-cart report compatibility map, populated on mount.
  // Missing entry means "backend didn't tell us about this cart" — for
  // LocalCarts (browser-only) that's the correct answer; we mark them
  // as incompatible because Reports run server-side.
  const [cartCompat, setCartCompat] = useState<Map<string, ReportCartEntry>>(new Map())

  const cartSelectorButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchReportCarts()
      .then((entries) => {
        if (cancelled) return
        const map = new Map<string, ReportCartEntry>()
        for (const e of entries) map.set(e.id, e)
        setCartCompat(map)
      })
      .catch(() => {
        // If the enumeration fails the selector falls back to showing
        // every cart as "unknown" — the input pane's cart_not_found
        // amber panel is still the safety net.
      })
    return () => { cancelled = true }
  }, [cartridges])

  // Build the cart-picker options. LocalCart names come first (they're the
  // browser-mounted "hot" carts the user just opened), then the server
  // cartridges. Selected value is the display string; if nothing is mounted
  // we fall through to null so the pane header shows "(none mounted)".
  //
  // Compatibility annotation:
  //   • LocalCart entries are always report_compatible=false (server-side
  //     engine can't reach them).
  //   • Server entries look up their stem in cartCompat; unknown ones
  //     (backend didn't enumerate the stem) fall back to a permissive
  //     "true" so nothing hides accidentally — the amber failure panel
  //     covers the mis-guess case.
  const cartOptions = useMemo(() => {
    const opts: CartOption[] = []
    const seen = new Set<string>()
    const push = (o: CartOption) => {
      if (seen.has(o.id)) return
      seen.add(o.id)
      opts.push(o)
    }
    for (const name of localCarts.keys()) {
      push({
        id: `local:${name}`,
        label: name,
        kind: 'local',
        reportCompatible: false,
        format: 'npz',
      })
    }
    for (const c of cartridges) {
      const meta = cartCompat.get(c.name)
      push({
        id: `server:${c.name}`,
        // Backend-provided display_name annotates "(legacy)"; fall back
        // to the bare cart name when we don't have an entry yet.
        label: meta?.display_name ?? c.name,
        kind: 'server',
        // Optimistic default: if we don't know, assume compatible so
        // the user isn't blocked on a slow enumerate.
        reportCompatible: meta?.report_compatible ?? true,
        format: (meta?.format as 'npz' | 'pkl' | undefined) ?? 'npz',
      })
    }
    return opts
  }, [cartridges, localCarts, cartCompat])

  const defaultCartId = useMemo(() => {
    if (activeLocalCart) return `local:${activeLocalCart}`
    if (mountedCartridge) return `server:${mountedCartridge}`
    // Prefer a report-compatible cart over an incompatible one for the
    // default selection — reduces the odds a fresh visitor picks a
    // legacy cart out of the gate.
    const firstCompat = cartOptions.find((o) => o.reportCompatible)
    return firstCompat?.id ?? cartOptions[0]?.id ?? null
  }, [activeLocalCart, mountedCartridge, cartOptions])

  const [selectedCartId, setSelectedCartId] = useState<string | null>(defaultCartId)
  const [activeReport, setActiveReport] = useState<ReportDefinition | null>(null)
  // Pre-fill payload passed to the input pane on Regenerate. Cleared
  // whenever the user opens a fresh card so a click on a new report
  // doesn't inherit stale values from an unrelated one.
  const [prefillInputs, setPrefillInputs] = useState<Record<string, unknown> | null>(null)
  // Full-width results — non-null hides the grid + hides the recent /
  // scheduled sections. When null, the grid is back.
  const [resultView, setResultView] = useState<ResultViewState | null>(null)

  const effectiveCartId = selectedCartId ?? defaultCartId
  const selectedCart = effectiveCartId
    ? cartOptions.find((o) => o.id === effectiveCartId) ?? null
    : null
  const selectedCartLabel = selectedCart?.label ?? null

  const handleOpenCard = (report: ReportDefinition) => {
    setPrefillInputs(null)
    setActiveReport(report)
  }

  const handleGenerateSuccess = useCallback(
    (response: GenerateReportResponse, submittedInputs: Record<string, unknown>) => {
      if (!activeReport) return
      setResultView({
        report: activeReport,
        response,
        cartLabel: selectedCartLabel,
        inputs: submittedInputs,
      })
      setActiveReport(null)
      setPrefillInputs(null)
    },
    [activeReport, selectedCartLabel],
  )

  const handleCloseResults = () => setResultView(null)

  const handleRegenerate = () => {
    if (!resultView) return
    setPrefillInputs(resultView.inputs)
    setActiveReport(resultView.report)
    // Intentionally leave resultView in place — the user can still see
    // the current results while adjusting inputs. When they click
    // Generate, the success handler overwrites the resultView.
  }

  const focusCartSelector = () => {
    // Called from the pane's [Pick another cart] button — dismiss the
    // pane and put focus on the cart selector button so keyboard users
    // land where they need to.
    setActiveReport(null)
    setPrefillInputs(null)
    requestAnimationFrame(() => {
      cartSelectorButtonRef.current?.focus()
    })
  }

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto relative">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
              <FileBarChart size={28} className="text-emerald-300" />
              Reports
            </h1>
            <p className="text-sm text-slate-500">
              Structured views over the mounted cart. Pick a report to see its inputs.
            </p>
          </div>

          {/* Cart selector — mirrors OverviewScreen's Mounted stat but as a
              dropdown so the user can run a report against any of their
              recently-used carts without leaving the tab. */}
          <CartSelector
            options={cartOptions}
            selectedId={effectiveCartId}
            onSelect={setSelectedCartId}
            buttonRef={cartSelectorButtonRef}
          />
        </div>

        {resultView ? (
          <ReportResultsView
            report={resultView.report}
            response={resultView.response}
            cartLabel={resultView.cartLabel}
            onClose={handleCloseResults}
            onRegenerate={handleRegenerate}
          />
        ) : (
          <>
            {/* Card grid — 3-up on wide, 2-up on medium, 1-up on narrow */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {REPORT_DEFINITIONS.map((report) => (
                <ReportCard
                  key={report.name}
                  report={report}
                  onRun={() => handleOpenCard(report)}
                />
              ))}
            </div>

            {/* Recent reports — empty state for now. Once report modules land,
                each run adds a row here with timestamp + parameters + open button. */}
            <section className="rounded-lg border border-slate-700 bg-slate-800/30">
              <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
                <Clock size={13} className="text-slate-500" />
                <h2 className="text-xs uppercase tracking-wider text-slate-500">Recent reports</h2>
              </div>
              <div className="p-6 text-center text-xs text-slate-500 italic">
                Your report history will appear here after you run one.
              </div>
            </section>

            {/* Scheduled briefings — empty state for now. Scheduling worker
                lands in wave 3 (design doc §C.4) alongside delivery adapters. */}
            <section className="rounded-lg border border-slate-700 bg-slate-800/30">
              <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
                <CalendarClock size={13} className="text-slate-500" />
                <h2 className="text-xs uppercase tracking-wider text-slate-500">Scheduled briefings</h2>
              </div>
              <div className="p-6 text-center text-xs text-slate-500 italic">
                Post-launch: schedule recurring reports for auto-delivery.
              </div>
            </section>

            <p className="text-center text-[11px] text-slate-600 italic pt-2">
              Report modules land wave-by-wave. Wave 1: Summary, Comparison, Entity Rollup, Change Log.
              Wave 2: Timeline, Trend, Financial. Wave 3: Executive TL;DR + scheduling.
            </p>
          </>
        )}
      </div>

      {/* Slide-in input pane. Renders as an overlay on top of the grid or
          the full-width results view so the user can jump back with the X
          button without losing scroll position. cartRef carries the full
          "server:foo" / "local:bar" identifier; the pane strips the prefix
          + short-circuits local carts with a friendly notice. */}
      {activeReport && (
        <ReportInputPane
          report={activeReport}
          cartName={selectedCartLabel}
          cartRef={effectiveCartId}
          initialInputs={prefillInputs}
          onClose={() => setActiveReport(null)}
          onSuccess={handleGenerateSuccess}
          onPickAnotherCart={focusCartSelector}
        />
      )}
    </main>
  )
}

// One entry in the cart-picker dropdown. Enriched with report-compat
// metadata pulled from /api/reports/carts so the selector can render
// legacy entries in a subtle greyed state.
interface CartOption {
  id: string
  label: string
  kind: 'local' | 'server'
  reportCompatible: boolean
  format: 'npz' | 'pkl'
}

// State carried while a report result is displayed full-width. `inputs`
// is the exact payload we posted, used to pre-fill the input pane on
// Regenerate.
interface ResultViewState {
  report: ReportDefinition
  response: GenerateReportResponse
  cartLabel: string | null
  inputs: Record<string, unknown>
}

// Cart selector dropdown. Same close-on-outside-click pattern as
// SearchToolbar's cart picker but slimmed down — no mount/unmount buttons
// here since Reports is a read-only surface. Clicking an option only changes
// which cart future report runs would target.
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
        Load a cart in Search to use for a Report.
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
        title="Choose which cart the report runs against"
      >
        <Database size={14} className="text-emerald-300 shrink-0" />
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
          <span
            className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono ${
              selected.kind === 'local'
                ? 'bg-cyan-500/15 border border-cyan-500/40 text-cyan-300'
                : 'bg-purple-500/15 border border-purple-500/40 text-purple-300'
            }`}
          >
            {selected.kind}
          </span>
        )}
        <ChevronDown size={14} className="text-slate-500 shrink-0" />
      </button>

      {open && (
        <>
          {/* click-outside catcher */}
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
                ? 'This cart uses a legacy format. Reports need the newer .cart.npz format. Convert via Cart Builder → Save as .cart.npz.'
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
                      ? 'bg-emerald-500/10 text-emerald-200'
                      : incompatible
                        ? 'text-slate-500 hover:bg-slate-800/60'
                        : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {incompatible ? (
                    <Lock size={13} className="text-amber-400/70 shrink-0" />
                  ) : (
                    <Database size={13} className={active ? 'text-emerald-300' : 'text-slate-500'} />
                  )}
                  <span className={`flex-1 truncate ${incompatible ? 'italic' : ''}`}>
                    {opt.label}
                  </span>
                  <span
                    className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono ${
                      opt.kind === 'local'
                        ? 'bg-cyan-500/15 border border-cyan-500/40 text-cyan-300'
                        : 'bg-purple-500/15 border border-purple-500/40 text-purple-300'
                    }`}
                  >
                    {opt.kind}
                  </span>
                </button>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
