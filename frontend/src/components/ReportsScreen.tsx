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
//
// 2026-07-13 (Phase A source-file links):
//   • Report state (`currentReport`) is now stored in appStore, not
//     local component state. Reports → Search (via a clicked source-file
//     link) → Reports resumes exactly where the user left off. Only
//     Close explicitly clears; tab-switches leave the report visible.
//     ReportDefinition is looked up by slug on read so we don't serialize
//     the full definition object into the store.
//
// 2026-07-13 (interim UX — hide legacy carts from Reports selector):
//   • Reports-only view: legacy (report_compatible=false) carts are
//     filtered out of the dropdown entirely rather than rendered greyed
//     with a Lock icon + tooltip. Legacy carts stay visible on every
//     OTHER tab (Search, Cart Builder, Edit Carts, …). If nothing
//     compatible is available, the selector swaps to a friendly empty-
//     state; the reports grid stays visible so users can still see the
//     shape of what's available.
//   • Rationale: Andy's demo carts are legacy .pkl; hitting the amber
//     "legacy cart format" panel over and over was noise. Hiding them
//     is an interim fix until the demo carts get rebuilt as .cart.npz.
//   • cartCompat map + optimistic default preserved. Amber cart_not_found
//     panel in ReportInputPane stays as defense-in-depth for the case
//     where the mounted cart is legacy but no compatible cart was picked
//     from the dropdown (defaultCartId still prefers the mounted cart).

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
        // Report-scope location: canonical vs sandbox. Only meaningful for
        // report-compatible entries; drives the SANDBOX badge in the
        // dropdown so users can tell a temporary upload apart from the
        // curated demo carts. Unknown or missing = 'canonical' so nothing
        // renders the sandbox badge accidentally.
        location: (meta?.location as 'canonical' | 'sandbox' | undefined) ?? 'canonical',
      })
    }
    // Also surface sandbox carts that the /api/reports/carts endpoint
    // enumerated but which don't correspond to a mounted cartridge in
    // the outer /api/cartridges list. Without this pass, uploaded carts
    // wouldn't appear in the dropdown until a mount cycle refreshed the
    // outer list (which sandbox uploads don't do — mount is a separate
    // step). Sandbox entries carry their own uuid-prefixed id so they
    // don't collide with the loop above.
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
  }, [cartridges, localCarts, cartCompat])

  const defaultCartId = useMemo(() => {
    if (activeLocalCart) return `local:${activeLocalCart}`
    if (mountedCartridge) return `server:${mountedCartridge}`
    // Prefer a report-compatible cart over an incompatible one for the
    // default selection — reduces the odds a fresh visitor picks a
    // legacy cart out of the gate. (Redundant now that the dropdown
    // filters out incompatible carts, but harmless — kept in case the
    // filter ever gets relaxed.)
    const firstCompat = cartOptions.find((o) => o.reportCompatible)
    return firstCompat?.id ?? cartOptions[0]?.id ?? null
  }, [activeLocalCart, mountedCartridge, cartOptions])

  // Reports-only filter: hide legacy (report_compatible=false) carts
  // from the selector entirely. Interim UX until legacy carts get
  // rebuilt as .cart.npz. Every other tab (Search, Cart Builder, etc.)
  // still lists them. When this list is empty we swap the selector
  // for a friendly empty-state below.
  const compatibleCartOptions = useMemo(
    () => cartOptions.filter((o) => o.reportCompatible),
    [cartOptions],
  )

  const [selectedCartId, setSelectedCartId] = useState<string | null>(defaultCartId)
  const [activeReport, setActiveReport] = useState<ReportDefinition | null>(null)
  // Pre-fill payload passed to the input pane on Regenerate. Cleared
  // whenever the user opens a fresh card so a click on a new report
  // doesn't inherit stale values from an unrelated one.
  const [prefillInputs, setPrefillInputs] = useState<Record<string, unknown> | null>(null)

  // Full-width results — non-null hides the grid + hides the recent /
  // scheduled sections. When null, the grid is back. Store-backed so it
  // survives tab switches (2026-07-13 Phase A). We resolve the report
  // definition by slug on read so the store payload stays serializable.
  //
  // 2026-07-13 Phase A follow-up (persistence bug fix): the render gate
  // depends ONLY on `currentReport`. If the slug lookup ever fails to
  // resolve (dev HMR reload of report-definitions.ts, a mid-refactor
  // rename, an unknown-slug payload snuck into the store), we still
  // render the results view using a synthesized fallback definition
  // built from the stored `reportSlug` + `reportDisplayName`. The
  // reason: the invariant is "the report stays visible until the user
  // clicks × Close" and gating on the derived slug lookup would break
  // that if the lookup ever returned null. The Regenerate button
  // early-returns when the slug isn't resolvable (the input pane
  // needs the real schema); Close + Copy + Download all work off the
  // stored response regardless.
  const currentReport = useAppStore((s) => s.currentReport)
  const setCurrentReport = useAppStore((s) => s.setCurrentReport)
  const clearCurrentReport = useAppStore((s) => s.clearCurrentReport)
  const resultViewReport = useMemo(
    () =>
      currentReport
        ? REPORT_DEFINITIONS.find((r) => r.name === currentReport.reportSlug) ?? null
        : null,
    [currentReport],
  )
  // Synthesized fallback when the slug lookup fails. Only the
  // `displayName` field is user-visible in `ReportResultsView` (title
  // in the toolbar + footer crumb + download filename), so the minimal
  // shape covers the render need. `inputSchema: []` keeps Regenerate
  // safe: if the user clicks it, the pane opens with zero required
  // fields and no defaults — clearly wrong, so Regenerate is guarded
  // in handleRegenerate to no-op when the real definition is missing.
  const displayedReport: ReportDefinition | null = currentReport
    ? (resultViewReport ?? {
        name: currentReport.reportSlug,
        displayName: currentReport.reportDisplayName,
        description: '',
        icon: 'FileText',
        llmDependency: false,
        inputSchema: [],
      })
    : null
  const hasResultView = !!currentReport && !!displayedReport

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
      // Cart ref used for filenames + downloads is the identifier we
      // actually POSTed with; the label is a human display string.
      const cartRefFromMeta = response.metadata && typeof response.metadata.cart_ref === 'string'
        ? (response.metadata.cart_ref as string)
        : (effectiveCartId ?? selectedCartLabel ?? '')
      setCurrentReport({
        cartRef: cartRefFromMeta,
        cartLabel: selectedCartLabel,
        reportSlug: activeReport.name,
        reportDisplayName: activeReport.displayName,
        submittedInputs,
        response,
        generatedAt: response.generated_at,
      })
      setActiveReport(null)
      setPrefillInputs(null)
    },
    [activeReport, selectedCartLabel, effectiveCartId, setCurrentReport],
  )

  const handleCloseResults = () => clearCurrentReport()

  const handleRegenerate = () => {
    if (!currentReport || !resultViewReport) return
    setPrefillInputs(currentReport.submittedInputs)
    setActiveReport(resultViewReport)
    // Intentionally leave currentReport in place — the user can still
    // see the current results while adjusting inputs. When they click
    // Generate, the success handler overwrites currentReport.
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
              recently-used carts without leaving the tab. Reports-only
              filter hides legacy .pkl carts (and LocalCarts, which run
              browser-side); if nothing compatible is available we swap
              for the empty-state below instead of showing an empty
              dropdown. */}
          {compatibleCartOptions.length === 0 ? (
            <ReportsEmptyCartState />
          ) : (
            <CartSelector
              options={compatibleCartOptions}
              selectedId={effectiveCartId}
              onSelect={setSelectedCartId}
              buttonRef={cartSelectorButtonRef}
            />
          )}
        </div>

        {hasResultView && displayedReport && currentReport ? (
          <ReportResultsView
            report={displayedReport}
            response={currentReport.response}
            cartLabel={currentReport.cartLabel}
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

// Empty-state shown in place of the cart selector when the droplet has
// zero report-compatible carts. Sized to sit in the same header slot as
// the CartSelector button so the header layout doesn't jump. Reports
// grid stays visible below (see ReportsScreen's main render) so users
// can still see what report types would be available.
function ReportsEmptyCartState() {
  return (
    <div
      className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2
                 text-xs text-amber-200 flex items-start gap-2 max-w-md"
      role="status"
    >
      <Lock size={13} className="shrink-0 mt-0.5 text-amber-400/80" />
      <span>
        No report-compatible carts available yet. Report engine needs the
        .cart.npz format. Rebuild legacy carts via Cart Builder → Save as
        .cart.npz, or ask your admin to convert existing carts.
      </span>
    </div>
  )
}

// One entry in the cart-picker dropdown. Enriched with report-compat
// metadata pulled from /api/reports/carts so the selector can render
// legacy entries in a subtle greyed state, and sandbox uploads with a
// distinct SANDBOX badge signalling their short-TTL nature.
interface CartOption {
  id: string
  label: string
  kind: 'local' | 'server'
  reportCompatible: boolean
  format: 'npz' | 'pkl'
  location?: 'canonical' | 'sandbox'
}

// Small uppercase pill that identifies which surface a cart lives on.
// Three states:
//   • local        (cyan)   — browser-only LocalCart, never touched server.
//   • server       (purple) — canonical droplet cart under cartridges/ or sample_data/.
//   • sandbox      (amber)  — short-TTL upload under _session_uploads/.
// Sandbox uses amber to match the visual language of the "temporary"
// warnings (amber panels for legacy/expired carts, amber ejection prompts).
function CartKindBadge({
  kind,
  location,
}: {
  kind: 'local' | 'server'
  location?: 'canonical' | 'sandbox'
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
  return (
    <span
      className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded font-mono ${
        kind === 'local'
          ? 'bg-cyan-500/15 border border-cyan-500/40 text-cyan-300'
          : 'bg-purple-500/15 border border-purple-500/40 text-purple-300'
      }`}
    >
      {kind}
    </span>
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
          <CartKindBadge kind={selected.kind} location={selected.location} />
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
