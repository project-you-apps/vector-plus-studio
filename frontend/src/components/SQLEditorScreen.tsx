import { useMemo, useRef, useState } from 'react'
import {
  Terminal, Database, ChevronDown, ChevronRight, Search, PlayCircle,
  Copy, ClipboardPaste, Eraser, Bookmark, Clock, Sparkles, Info,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import {
  OPERATION_TEMPLATES,
  SQL_OPERATIONS,
  type SQLOperation,
} from '../sql/operation-templates'

// SQL Editor screen — v1 shell.
//
// Scope of this pass:
//   • Cart selector at top (defaults to the currently-mounted cart)
//   • Always-visible NL search bar (primary input)
//   • Collapsible SQL Mode with textarea + 10 operation-template buttons
//   • Placeholder "not yet available" panel on Run Query / Search
//   • Empty-state Recent / Saved query panels
//   • Cross-mode handoff teaser at the bottom
//
// NOT WIRED YET: the SQL interpreter that translates queries into lattice ops
// is v1.5 architecture work. Clicking Run Query / Search shows a placeholder
// panel echoing what WOULD have run.

type QueryMode = 'nl' | 'sql'

interface PendingQuery {
  mode: QueryMode
  text: string
  cartLabel: string | null
}

export default function SQLEditorScreen() {
  // Cart identity — mirror ReportsScreen's mounted-cart resolution so the
  // selector defaults to the same cart the user has active elsewhere.
  const mountedCartridge = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const cartridges = useAppStore((s) => s.cartridges)
  const localCarts = useAppStore((s) => s.localCarts)

  const cartOptions = useMemo(() => {
    const opts: Array<{ id: string; label: string; kind: 'local' | 'server' }> = []
    for (const name of localCarts.keys()) {
      opts.push({ id: `local:${name}`, label: name, kind: 'local' })
    }
    // Dedupe by id — the same cartridge name can appear multiple times in
    // the cartridges[] array when the sandbox and multi-mount pools both
    // reference the same file; React logs a duplicate-key warning otherwise.
    const seen = new Set<string>()
    for (const opt of opts) seen.add(opt.id)
    for (const c of cartridges) {
      const id = `server:${c.name}`
      if (seen.has(id)) continue
      seen.add(id)
      opts.push({ id, label: c.name, kind: 'server' })
    }
    return opts
  }, [cartridges, localCarts])

  const defaultCartId = useMemo(() => {
    if (activeLocalCart) return `local:${activeLocalCart}`
    if (mountedCartridge) return `server:${mountedCartridge}`
    return cartOptions[0]?.id ?? null
  }, [activeLocalCart, mountedCartridge, cartOptions])

  const [selectedCartId, setSelectedCartId] = useState<string | null>(defaultCartId)
  const effectiveCartId = selectedCartId ?? defaultCartId
  const selectedCartLabel = effectiveCartId
    ? cartOptions.find((o) => o.id === effectiveCartId)?.label ?? null
    : null

  // Editor + surface state.
  const [nlQuery, setNlQuery] = useState('')
  const [sqlText, setSqlText] = useState('')
  const [sqlExpanded, setSqlExpanded] = useState(false)
  const [recentOpen, setRecentOpen] = useState(false)
  const [savedOpen, setSavedOpen] = useState(false)
  const [pending, setPending] = useState<PendingQuery | null>(null)

  // Textarea ref so operation buttons can splice their template at the caret
  // (rather than always appending). Falls back to append when the ref is unset
  // — e.g. the SQL section is still collapsed when a button fires (which
  // shouldn't happen since the buttons live inside the collapsed section, but
  // guarded anyway).
  const sqlTextareaRef = useRef<HTMLTextAreaElement>(null)

  const insertOperation = (op: SQLOperation) => {
    const snippet = OPERATION_TEMPLATES[op]
    const el = sqlTextareaRef.current
    if (!el) {
      // Collapsed — expand and append. Rare, but keeps the buttons functional
      // even if the layout evolves.
      setSqlExpanded(true)
      setSqlText((prev) => prev + snippet)
      return
    }
    const start = el.selectionStart ?? sqlText.length
    const end = el.selectionEnd ?? sqlText.length
    const next = sqlText.slice(0, start) + snippet + sqlText.slice(end)
    setSqlText(next)
    // Restore caret position after React commits the value. Placing it at the
    // end of the inserted snippet feels most natural — user picks up typing
    // right where the template stops.
    const nextCaret = start + snippet.length
    requestAnimationFrame(() => {
      el.focus()
      el.setSelectionRange(nextCaret, nextCaret)
    })
  }

  const runSql = () => {
    const trimmed = sqlText.trim()
    if (!trimmed) return
    setPending({ mode: 'sql', text: trimmed, cartLabel: selectedCartLabel })
  }

  const runNl = () => {
    const trimmed = nlQuery.trim()
    if (!trimmed) return
    setPending({ mode: 'nl', text: trimmed, cartLabel: selectedCartLabel })
  }

  // (2026-07-10 Andy feedback): after Copy, the button flips to a Paste action.
  // Rationale: a user who just copied a snippet is likely about to reuse it —
  // making the same button paste-at-cursor is a nice one-hand affordance and
  // saves a right-click. The "just copied" state clears after 5 seconds so the
  // button doesn't lie about clipboard contents indefinitely.
  const [justCopied, setJustCopied] = useState(false)
  const copyRevertTimerRef = useRef<number | null>(null)

  const copySql = async () => {
    if (!sqlText) return
    try {
      await navigator.clipboard.writeText(sqlText)
      setJustCopied(true)
      // Cancel any prior pending revert so successive copies extend the window.
      if (copyRevertTimerRef.current !== null) {
        window.clearTimeout(copyRevertTimerRef.current)
      }
      copyRevertTimerRef.current = window.setTimeout(() => {
        setJustCopied(false)
        copyRevertTimerRef.current = null
      }, 5000)
    } catch {
      // Clipboard API can fail under insecure contexts / permissions —
      // silent fail is fine here; user can Ctrl+C the textarea.
    }
  }

  const pasteSql = async () => {
    try {
      const clip = await navigator.clipboard.readText()
      if (!clip) return
      const el = sqlTextareaRef.current
      if (!el) {
        setSqlText((prev) => prev + clip)
      } else {
        const start = el.selectionStart ?? sqlText.length
        const end = el.selectionEnd ?? sqlText.length
        const next = sqlText.slice(0, start) + clip + sqlText.slice(end)
        setSqlText(next)
        const nextCaret = start + clip.length
        requestAnimationFrame(() => {
          el.focus()
          el.setSelectionRange(nextCaret, nextCaret)
        })
      }
      // Revert to Copy mode after paste — the copied buffer got consumed.
      setJustCopied(false)
      if (copyRevertTimerRef.current !== null) {
        window.clearTimeout(copyRevertTimerRef.current)
        copyRevertTimerRef.current = null
      }
    } catch {
      // Clipboard-read permission not granted — silently no-op.
    }
  }

  const clearSql = () => setSqlText('')
  const clearPending = () => setPending(null)

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto relative">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
              <Terminal size={28} className="text-emerald-300" />
              SQL Editor
            </h1>
            <p className="text-sm text-slate-500">
              Semantic-first query surface. Cart = the table, natural language is
              primary, SQL is the power-user mode.
            </p>
          </div>

          {/* Cart selector — same shape as ReportsScreen's. */}
          <CartSelector
            options={cartOptions}
            selectedId={effectiveCartId}
            onSelect={setSelectedCartId}
          />
        </div>

        {/* Semantic Search bar (always visible; primary input) */}
        <section className="rounded-lg border border-slate-700 bg-slate-800/30">
          <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
            <Search size={13} className="text-emerald-300" />
            <h2 className="text-xs uppercase tracking-wider text-slate-400">Semantic Search</h2>
          </div>
          <div className="p-4 flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              value={nlQuery}
              onChange={(e) => setNlQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') runNl()
              }}
              placeholder="What are you looking for?"
              className="flex-1 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2
                         text-sm text-slate-200 focus:outline-none focus:border-emerald-500/60"
            />
            <button
              onClick={runNl}
              disabled={!nlQuery.trim()}
              className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                          text-sm font-medium transition-colors ${
                            nlQuery.trim()
                              ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-200 hover:bg-emerald-500/30'
                              : 'bg-slate-800/60 border border-slate-700 text-slate-600 cursor-not-allowed'
                          }`}
            >
              <Sparkles size={14} />
              Search
            </button>
          </div>
        </section>

        {/* SQL Mode — collapsible so consumer users aren't intimidated */}
        <section className="rounded-lg border border-slate-700 bg-slate-800/30">
          <button
            onClick={() => setSqlExpanded((v) => !v)}
            className="w-full px-4 py-2 border-b border-slate-700 flex items-center gap-2
                       text-left hover:bg-slate-800/40 transition-colors"
            aria-expanded={sqlExpanded}
          >
            {/* (2026-07-10 Andy feedback): heading first, then icon + chevron on
                the RIGHT side so the terminal-prompt icon doesn't get visually
                confused with the un-clicked chevron next to it. Reading order
                becomes "SQL Mode → [prompt icon] → [expand chevron]". */}
            <h2 className="text-xs uppercase tracking-wider text-slate-400 flex-1">SQL Mode</h2>
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono">
              {sqlExpanded ? 'Collapse' : 'Expand'}
            </span>
            <Terminal size={13} className="text-purple-300 shrink-0" />
            {sqlExpanded ? (
              <ChevronDown size={13} className="text-slate-400 shrink-0" />
            ) : (
              <ChevronRight size={13} className="text-slate-400 shrink-0" />
            )}
          </button>

          {sqlExpanded && (
            <div className="p-4 space-y-3">
              {/* Editor */}
              <textarea
                ref={sqlTextareaRef}
                value={sqlText}
                onChange={(e) => setSqlText(e.target.value)}
                spellCheck={false}
                placeholder={`-- Compose a query, or click an operation button below.\nSELECT * FROM cart\nWHERE source LIKE 'sysco%'\nORDER BY relevance DESC\nLIMIT 10;`}
                className="w-full min-h-[180px] rounded-lg bg-slate-950/70 border border-slate-800
                           px-3 py-2 font-mono text-[13px] leading-relaxed text-slate-200
                           focus:outline-none focus:border-purple-500/60 resize-y"
              />

              {/* Editor action row */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={runSql}
                  disabled={!sqlText.trim()}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                              transition-colors ${
                                sqlText.trim()
                                  ? 'bg-purple-500/20 border border-purple-500/50 text-purple-200 hover:bg-purple-500/30'
                                  : 'bg-slate-800/60 border border-slate-700 text-slate-600 cursor-not-allowed'
                              }`}
                >
                  <PlayCircle size={14} />
                  Run Query
                </button>
                {justCopied ? (
                  <button
                    onClick={pasteSql}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-emerald-200
                               border border-emerald-500/40 bg-emerald-500/10 hover:bg-emerald-500/20
                               transition-colors"
                    title="Paste the copied query at the cursor position"
                  >
                    <ClipboardPaste size={13} />
                    Paste
                  </button>
                ) : (
                  <button
                    onClick={copySql}
                    disabled={!sqlText}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-300
                               border border-slate-700 hover:border-slate-500 hover:bg-slate-800/40
                               disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    title="Copy SQL to clipboard"
                  >
                    <Copy size={13} />
                    Copy
                  </button>
                )}
                <button
                  onClick={clearSql}
                  disabled={!sqlText}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-300
                             border border-slate-700 hover:border-slate-500 hover:bg-slate-800/40
                             disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Eraser size={13} />
                  Clear
                </button>
              </div>

              {/* Operations toolbar */}
              <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 p-3 space-y-2">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Info size={11} className="text-slate-500" />
                  Operations — click to insert a template at the cursor
                </div>
                <div className="flex flex-wrap gap-2">
                  {SQL_OPERATIONS.map((op) => (
                    <button
                      key={op}
                      onClick={() => insertOperation(op)}
                      className="px-3 py-1.5 rounded-full text-xs font-mono
                                 bg-slate-800/60 border border-slate-700 text-slate-200
                                 hover:border-purple-500/60 hover:bg-purple-500/10 hover:text-purple-200
                                 transition-colors"
                      title={`Insert ${op} template`}
                    >
                      {op}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Results panel — placeholder unless a query has been dispatched */}
        <section className="rounded-lg border border-slate-700 bg-slate-800/30">
          <div className="px-4 py-2 border-b border-slate-700 flex items-center gap-2">
            <Database size={13} className="text-emerald-300" />
            <h2 className="text-xs uppercase tracking-wider text-slate-400">Results</h2>
          </div>
          {pending ? (
            <PlaceholderResult pending={pending} onClear={clearPending} />
          ) : (
            <div className="p-8 text-center text-xs text-slate-500 italic">
              Query results will appear here. Enter a query above and click Search or Run.
            </div>
          )}
        </section>

        {/* Recent queries — empty-state stub for wave-1 */}
        <section className="rounded-lg border border-slate-700 bg-slate-800/30">
          <button
            onClick={() => setRecentOpen((v) => !v)}
            className="w-full px-4 py-2 border-b border-slate-700 flex items-center gap-2
                       text-left hover:bg-slate-800/40 transition-colors"
            aria-expanded={recentOpen}
          >
            {recentOpen ? (
              <ChevronDown size={13} className="text-slate-400 shrink-0" />
            ) : (
              <ChevronRight size={13} className="text-slate-400 shrink-0" />
            )}
            <Clock size={13} className="text-slate-500 shrink-0" />
            <h2 className="text-xs uppercase tracking-wider text-slate-400 flex-1">Recent queries</h2>
          </button>
          {recentOpen && (
            <div className="p-6 text-center text-xs text-slate-500 italic">
              Your query history will appear here.
            </div>
          )}
        </section>

        {/* Saved queries — empty-state stub for wave-1 */}
        <section className="rounded-lg border border-slate-700 bg-slate-800/30">
          <button
            onClick={() => setSavedOpen((v) => !v)}
            className="w-full px-4 py-2 border-b border-slate-700 flex items-center gap-2
                       text-left hover:bg-slate-800/40 transition-colors"
            aria-expanded={savedOpen}
          >
            {savedOpen ? (
              <ChevronDown size={13} className="text-slate-400 shrink-0" />
            ) : (
              <ChevronRight size={13} className="text-slate-400 shrink-0" />
            )}
            <Bookmark size={13} className="text-slate-500 shrink-0" />
            <h2 className="text-xs uppercase tracking-wider text-slate-400 flex-1">Saved queries</h2>
          </button>
          {savedOpen && (
            <div className="p-6 text-center text-xs text-slate-500 italic">
              Save queries with a name to reuse them.
            </div>
          )}
        </section>

        {/* Cross-mode handoff teaser */}
        <p className="text-center text-[11px] text-slate-500 italic pt-2">
          Cross-mode: results referencing a SQL-backed source will show a{' '}
          <span className="font-mono text-slate-400">[Query source DB &rarr;]</span>{' '}
          handoff button once the interpreter ships.
        </p>

        <p className="text-center text-[11px] text-slate-600 italic">
          Wave 1: UI shell + operation templates. Wave 2: SQL interpreter + persistence +
          cross-mode handoff. Wave 3: multi-cart JOIN, GROUP BY over attractor basins, full CRUD.
        </p>
      </div>
    </main>
  )
}

// Cart selector dropdown — same close-on-outside-click pattern as ReportsScreen.
// Duplicated intentionally so the two screens can diverge without a shared
// component becoming a coordination liability early on.
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
        Load a cart in Search to query with SQL or NL.
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
        title="Choose which cart the query targets"
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

// Amber "not yet available" panel — mirrors the ReportInputPane
// PlaceholderResult treatment so both tabs share the same "wave-1 shell,
// interpreter next" visual language. Includes a Clear button so the user can
// dismiss the panel and try a different query without leaving the tab.
function PlaceholderResult({
  pending,
  onClear,
}: {
  pending: PendingQuery
  onClear: () => void
}) {
  const modeLabel = pending.mode === 'nl' ? 'NL search' : 'SQL'
  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
        <p className="text-amber-200 font-medium mb-2">
          Query execution is not yet available in this build.
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          The SQL interpreter is v1.5 architecture work. This shell lets you compose queries
          and preview how they'd read; interpretation + execution against the substrate
          ships next.
        </p>
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-4 space-y-3">
        <div className="text-[10px] uppercase tracking-wider text-slate-500">
          Would have run
        </div>
        <div className="text-xs text-slate-300 space-y-1.5">
          <div>
            <span className="text-slate-500">Cart:</span>{' '}
            <span className="font-mono text-emerald-300">
              {pending.cartLabel ?? '(none mounted)'}
            </span>
          </div>
          <div>
            <span className="text-slate-500">Mode:</span>{' '}
            <span className="font-mono text-emerald-300">{modeLabel}</span>
          </div>
          <div className="text-slate-500">Query:</div>
          <pre className="font-mono text-[11px] text-slate-300 bg-slate-900/70
                          border border-slate-800 rounded-md px-3 py-2 overflow-x-auto whitespace-pre-wrap">
{pending.text}
          </pre>
        </div>
      </div>

      <button
        onClick={onClear}
        className="self-start px-3 py-2 rounded-lg text-sm text-slate-300 border border-slate-700
                   hover:border-slate-500 hover:bg-slate-800/40 transition-colors"
      >
        Clear
      </button>
    </div>
  )
}
