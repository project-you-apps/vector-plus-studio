import { useMemo, useState } from 'react'
import {
  FileBarChart, Database, ChevronDown, Clock, CalendarClock,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { REPORT_DEFINITIONS, type ReportDefinition } from '../reports/report-definitions'
import ReportCard from './ReportCard'
import ReportInputPane from './ReportInputPane'

// Reports screen — the UX shell for the 8 report types (see
// docs/vps-internal/Report Types Design 2026-07-10.md).
//
// Scope of this pass (Andy 2026-07-10):
//   • Cart selector at top (defaults to the currently mounted cart)
//   • 8 report cards in a responsive grid
//   • Recent-reports and Scheduled-briefings sections as empty states
//   • Slide-in input pane when a card's Run button is clicked
//
// NOT WIRED YET: the backend report modules that actually run reports are
// future work. Clicking Generate in the input pane shows a placeholder.

export default function ReportsScreen() {
  // Cart identity — mirror the mounted-cart resolution used by SearchToolbar /
  // CRUDScreen so the selector defaults to whatever the user has active.
  const mountedCartridge = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const cartridges = useAppStore((s) => s.cartridges)
  const localCarts = useAppStore((s) => s.localCarts)

  // Build the cart-picker options. LocalCart names come first (they're the
  // browser-mounted "hot" carts the user just opened), then the server
  // cartridges. Selected value is the display string; if nothing is mounted
  // we fall through to null so the pane header shows "(none mounted)".
  const cartOptions = useMemo(() => {
    // Dedupe by generated id — cartridges[] can contain the same server-side
    // cart name more than once when the sandbox and multi-mount pools list
    // the same underlying file. Without dedup React logs the duplicate-key
    // warning and can misroute updates between the two entries.
    const opts: Array<{ id: string; label: string; kind: 'local' | 'server' }> = []
    const seen = new Set<string>()
    const push = (id: string, label: string, kind: 'local' | 'server') => {
      if (seen.has(id)) return
      seen.add(id)
      opts.push({ id, label, kind })
    }
    for (const name of localCarts.keys()) {
      push(`local:${name}`, name, 'local')
    }
    for (const c of cartridges) {
      push(`server:${c.name}`, c.name, 'server')
    }
    return opts
  }, [cartridges, localCarts])

  const defaultCartId = useMemo(() => {
    if (activeLocalCart) return `local:${activeLocalCart}`
    if (mountedCartridge) return `server:${mountedCartridge}`
    return cartOptions[0]?.id ?? null
  }, [activeLocalCart, mountedCartridge, cartOptions])

  const [selectedCartId, setSelectedCartId] = useState<string | null>(defaultCartId)
  const [activeReport, setActiveReport] = useState<ReportDefinition | null>(null)

  // Re-sync selection when the mounted-cart changes and the current selection
  // is still the auto-defaulted one. Preserves an explicit user pick.
  const effectiveCartId = selectedCartId ?? defaultCartId
  const selectedCartLabel = effectiveCartId
    ? cartOptions.find((o) => o.id === effectiveCartId)?.label ?? null
    : null

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
          />
        </div>

        {/* Card grid — 3-up on wide, 2-up on medium, 1-up on narrow */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {REPORT_DEFINITIONS.map((report) => (
            <ReportCard
              key={report.name}
              report={report}
              onRun={() => setActiveReport(report)}
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
      </div>

      {/* Slide-in input pane. Renders as an overlay on top of the grid so the
          user can jump back with the X button without losing scroll position.
          cartRef carries the full "server:foo" / "local:bar" identifier; the
          pane strips the prefix + short-circuits local carts with a friendly
          notice (reports run server-side, LocalCarts never touched the disk). */}
      {activeReport && (
        <ReportInputPane
          report={activeReport}
          cartName={selectedCartLabel}
          cartRef={effectiveCartId}
          onClose={() => setActiveReport(null)}
        />
      )}
    </main>
  )
}

// Cart selector dropdown. Same close-on-outside-click pattern as
// SearchToolbar's cart picker but slimmed down — no mount/unmount buttons
// here since Reports is a read-only surface. Clicking an option only changes
// which cart future report runs would target.
function CartSelector({
  options,
  selectedId,
  onSelect,
}: {
  options: Array<{ id: string; label: string; kind: 'local' | 'server' }>
  selectedId: string | null
  onSelect: (id: string) => void
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
            className="absolute right-0 top-full mt-1 z-20 min-w-[280px]
                       rounded-lg border border-slate-700 bg-slate-900 shadow-xl
                       max-h-[320px] overflow-y-auto"
          >
            {options.map((opt) => {
              const active = opt.id === selectedId
              return (
                <button
                  key={opt.id}
                  onClick={() => {
                    onSelect(opt.id)
                    setOpen(false)
                  }}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors ${
                    active
                      ? 'bg-emerald-500/10 text-emerald-200'
                      : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <Database size={13} className={active ? 'text-emerald-300' : 'text-slate-500'} />
                  <span className="flex-1 truncate">{opt.label}</span>
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
