// Pattern-0 TOC Panel — v1 (Andy 2026-07-01).
//
// Left-side panel on the Search tab that shows metadata + table-of-contents
// summary of the currently-mounted cart. Read-only v1. Users see what's in a
// cart before searching.
//
// Data source: GET /api/cart/pattern-0 on the backend. Refetches whenever
// the server-mounted cart OR the browser-mounted LocalCart changes.
//
// Empty-state / derived-stats banner: when Pattern-0 is minimal/absent the
// backend flips is_derived=true and toc_items come from source_hash counts;
// we surface a small "No metadata available — showing derived stats" banner
// at the top so the user knows why the panel is thin.
//
// Filter + pagination adapted from CRUDScreen's PassageBrowser pattern
// (Prev/Next + jump-to-page). Filter is CLIENT-SIDE substring match on
// item names/descriptions — no backend round-trip per spec.
//
// BRIEFING button surfaces only when the response includes an
// `agent_briefing` field with content. Click opens a modal with the full
// briefing text; the X in the corner closes.

import { useEffect, useMemo, useState } from 'react'
import {
  BookOpen, Filter as FilterIcon, Loader2, ChevronLeft, ChevronRight,
  X, Sparkles, AlertCircle, Info,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import * as api from '../api/client'
import type { Pattern0Response, Pattern0TocItem, SearchResult } from '../api/types'

const TOC_PAGE_SIZE = 25
// Drill view (per-file passages preview) — smaller page so the panel doesn't
// grow past its ~45vh cap set by the parent container.
const DRILL_PAGE_SIZE = 25


function BriefingModal({
  briefing,
  onClose,
}: {
  briefing: string
  onClose: () => void
}) {
  // Close on Escape for keyboard-nav parity.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Agent briefing"
      onClick={onClose}
    >
      <div
        className="max-w-2xl w-full max-h-[80vh] rounded-2xl border border-purple-500/40 bg-slate-900 shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Sparkles size={14} className="text-purple-400" />
            Agent Briefing
          </h3>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-200 p-1 rounded hover:bg-slate-800 transition-colors"
            title="Close (Esc)"
            aria-label="Close briefing"
          >
            <X size={16} />
          </button>
        </div>
        <div className="px-5 py-4 overflow-y-auto text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
          {briefing}
        </div>
      </div>
    </div>
  )
}


export default function Pattern0TocPanel() {
  // Both mount states drive refetch: server-mounted carts change via
  // status.mounted_cartridge; browser LocalCarts via activeLocalCart. When
  // both are null there's nothing to show and App.tsx hides the panel
  // (parent-level check), but we still guard here so the effect can bail.
  const mountedCartridge = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const localCarts = useAppStore((s) => s.localCarts)
  const listPassagesBySource = useAppStore((s) => s.localCartListPassagesBySource)
  const openModal = useAppStore((s) => s.openModal)

  const [data, setData] = useState<Pattern0Response | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [filter, setFilter] = useState('')
  const [offset, setOffset] = useState(0)
  const [pageJump, setPageJump] = useState('')
  const [briefingOpen, setBriefingOpen] = useState(false)

  // Drill-down state (Andy 2026-07-02) — click a file row to expand its
  // passages inline. LocalCart-only (backend Pattern-0 doesn't expose
  // per-source passage lookup). Read-only preview; editing lives in Edit
  // Carts.
  const [drillPath, setDrillPath] = useState<string | null>(null)
  const [drillOffset, setDrillOffset] = useState(0)
  const [drillPageJump, setDrillPageJump] = useState('')

  // Fetch whenever the mounted cart changes. LocalCart doesn't have a
  // server-side Pattern-0 (nothing was uploaded); in that case we synthesize
  // a derived response from LocalCart's sourcePaths without touching the
  // backend. See localCartListSources() in appStore for the source-count
  // logic we mirror here.
  useEffect(() => {
    let cancelled = false
    setOffset(0)
    setPageJump('')
    setFilter('')
    // Exit drill on cart change — the drillPath belongs to whichever cart was
    // previously active and won't resolve against the new one.
    setDrillPath(null)
    setDrillOffset(0)
    setDrillPageJump('')

    if (activeLocalCart) {
      // Browser-mounted cart — synthesize response client-side. Prefer the
      // pattern0.npy sidecar baked at build time (Andy 2026-07-02 bug 3);
      // fall back to derived-only stats when the cart predates the sidecar.
      const cart = localCarts.get(activeLocalCart)
      if (!cart) {
        setData(null)
        return
      }
      // Live per-source counts respecting tombstones. Used both for the
      // derived-only branch and to override the baked pattern0's chunks
      // number so tombstoned edits show accurate counts.
      const liveCounts = new Map<string, number>()
      if (cart.sourcePaths) {
        for (let i = 0; i < cart.sourcePaths.length; i++) {
          const sp = cart.sourcePaths[i]
          if (!sp) continue
          if (cart.tombstones.has(i)) continue
          liveCounts.set(sp, (liveCounts.get(sp) ?? 0) + 1)
        }
      }

      const p0 = cart.pattern0Meta
      const hasBakedMeta = !!p0 && (
        !!(p0.description && p0.description.trim())
        || !!(p0.agent_briefing && p0.agent_briefing.trim())
        || !!(p0.creator && p0.creator.trim())
        || !!(p0.owner && p0.owner.trim())
        || (Array.isArray(p0.tags) && p0.tags.length > 0)
        || (Array.isArray(p0.files) && p0.files.length > 0)
      )

      let toc_items: Pattern0TocItem[]
      if (p0 && Array.isArray(p0.files) && p0.files.length > 0) {
        toc_items = p0.files.map((f) => ({
          name: f.name,
          description: f.description ?? null,
          pattern_count: liveCounts.get(f.name) ?? f.chunks ?? 0,
        }))
      } else if (liveCounts.size > 0) {
        toc_items = Array.from(liveCounts.entries()).map(([name, pattern_count]) => ({
          name,
          description: null,
          pattern_count,
        }))
      } else {
        toc_items = [{
          name: cart.filename || cart.name,
          description: null,
          pattern_count: cart.passages.length,
        }]
      }

      setData({
        mounted: true,
        name: (p0?.cart_name && p0.cart_name.trim()) || cart.filename || cart.name,
        description: (p0?.description && p0.description.trim()) || null,
        creator: (p0?.creator && p0.creator.trim()) || null,
        created_at: (p0?.created_at && p0.created_at.trim()) || null,
        owner: (p0?.owner && p0.owner.trim()) || null,
        pattern_count: cart.passages.length,
        agent_briefing: (p0?.agent_briefing && p0.agent_briefing.trim()) || null,
        toc_items,
        is_derived: !hasBakedMeta,
      })
      setError(null)
      setLoading(false)
      return
    }

    if (!mountedCartridge) {
      setData(null)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)
    api.getCartPattern0()
      .then((resp) => {
        if (cancelled) return
        setData(resp)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        setData(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [mountedCartridge, activeLocalCart, localCarts])

  // Client-side substring filter (case-insensitive on name + description).
  const filteredItems = useMemo(() => {
    if (!data?.toc_items) return []
    const q = filter.trim().toLowerCase()
    if (!q) return data.toc_items
    return data.toc_items.filter((item) => {
      if (item.name.toLowerCase().includes(q)) return true
      if (item.description && item.description.toLowerCase().includes(q)) return true
      return false
    })
  }, [data?.toc_items, filter])

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / TOC_PAGE_SIZE))
  const currentPage = Math.floor(offset / TOC_PAGE_SIZE) + 1
  const pageItems = filteredItems.slice(offset, offset + TOC_PAGE_SIZE)

  // Reset to page 1 when the filter narrows below the current page.
  useEffect(() => {
    if (offset >= filteredItems.length && filteredItems.length > 0) {
      setOffset(0)
    }
  }, [filteredItems.length, offset])

  // Nothing mounted at all — panel is hidden by the parent, but guard here
  // so the rendered tree is empty when we're between mount states.
  if (!mountedCartridge && !activeLocalCart) return null

  if (loading && !data) {
    return (
      <div className="h-full flex items-center justify-center text-slate-500 border border-slate-700/50 rounded-xl bg-slate-800/20 p-6">
        <Loader2 size={20} className="animate-spin mr-2" />
        <span className="text-sm">Loading cart summary…</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-rose-300 border border-rose-500/30 rounded-xl bg-rose-500/5 p-6">
        <AlertCircle size={20} className="mb-2 text-rose-400" />
        <p className="text-sm font-medium">Failed to load cart summary</p>
        <p className="text-xs text-slate-500 mt-1 max-w-xs text-center">{error}</p>
      </div>
    )
  }

  if (!data || !data.mounted) return null

  const goPrev = () => setOffset(Math.max(0, offset - TOC_PAGE_SIZE))
  const goNext = () => setOffset(Math.min((totalPages - 1) * TOC_PAGE_SIZE, offset + TOC_PAGE_SIZE))
  const goPage = (page: number) => {
    const clamped = Math.max(1, Math.min(totalPages, page))
    setOffset((clamped - 1) * TOC_PAGE_SIZE)
  }

  // Header text: use the returned cart name; fall back to the raw
  // mounted_cartridge string. Spec says "<filename> TOC".
  const cartName = data.name || mountedCartridge || activeLocalCart || 'Cart'
  const hasBriefing = !!(data.agent_briefing && data.agent_briefing.trim().length > 0)
  // Backend Pattern-0 has no per-source passage lookup; only LocalCarts get
  // the click-to-drill affordance for now.
  const drillEnabled = !!activeLocalCart

  // ── Drill view — read-only per-file passage preview ──────────────────────
  // Only reachable for LocalCarts (see drillEnabled). Uses the same selector
  // that Edit Carts' Source Files drill does, so pagination + preview format
  // stays consistent across surfaces. NO edit/tombstone/restore buttons —
  // Search tab is read-only; Andy's spec.
  if (drillPath !== null && drillEnabled) {
    const window = listPassagesBySource(drillPath, drillOffset, DRILL_PAGE_SIZE)
    const total = window.total
    const totalDrillPages = Math.max(1, Math.ceil(total / DRILL_PAGE_SIZE))
    const currentDrillPage = Math.floor(drillOffset / DRILL_PAGE_SIZE) + 1
    const drillPrev = () => setDrillOffset(Math.max(0, drillOffset - DRILL_PAGE_SIZE))
    const drillNext = () =>
      setDrillOffset(Math.min((totalDrillPages - 1) * DRILL_PAGE_SIZE, drillOffset + DRILL_PAGE_SIZE))
    const drillGoPage = (page: number) => {
      const clamped = Math.max(1, Math.min(totalDrillPages, page))
      setDrillOffset((clamped - 1) * DRILL_PAGE_SIZE)
    }
    const exitDrill = () => {
      setDrillPath(null)
      setDrillOffset(0)
      setDrillPageJump('')
    }

    // Andy 2026-07-02: clicking a drilled passage opens the standard
    // PassageModal (same MORE + PREV|NEXT UX as search results). PREV|NEXT
    // stays inside this file because the modal's navigator uses the cart's
    // sourcePaths sidecar to clip to same-source neighbors. NO edit affordance
    // on the Search tab per spec — editing lives in Edit Carts.
    const openDrilledPassage = (p: { idx: number; title: string; preview: string }) => {
      if (!activeLocalCart) return
      const cart = localCarts.get(activeLocalCart)
      if (!cart) return
      const fullText = cart.passages[p.idx] ?? ''
      const total = cart.passages.length
      const result: SearchResult = {
        rank: 0,
        idx: p.idx,
        score: 0,
        cosine_score: null,
        physics_score: null,
        hamming_score: null,
        keyword_boost: null,
        title: p.title || `#${p.idx}`,
        preview: p.preview,
        full_text: fullText,
        from_lattice: false,
        // openModal clips these to same-source when the cart carries a
        // sourcePaths sidecar (which is the case for every drill target —
        // drillEnabled implies activeLocalCart, and the drill only appears
        // when sourcePaths exists). So even though we pass raw ±1 neighbors,
        // they collapse to null at file boundaries.
        prev_idx: p.idx > 0 ? p.idx - 1 : null,
        next_idx: p.idx < total - 1 ? p.idx + 1 : null,
        source_path: drillPath,
      }
      openModal(result)
    }
    return (
      <div className="h-full flex flex-col border border-slate-700/50 rounded-xl bg-slate-800/20 overflow-hidden">
        {/* Drill header — back arrow + filename + counts. Matches Edit Carts'
            SourceFilesPanel drill visual language but slate (not cyan) since
            this is Search context, not Edit context. */}
        <div className="px-4 py-2 border-b border-slate-700/50 flex items-center gap-2">
          <button
            onClick={exitDrill}
            className="flex items-center gap-1 px-2 py-1 rounded text-[11px] uppercase tracking-wider
                       bg-slate-800/60 border border-slate-700/50 text-slate-300
                       hover:bg-slate-700/60 hover:text-slate-100 transition-colors"
            title="Back to file list"
          >
            <ChevronLeft size={11} />
            Back
          </button>
          <span
            className="text-xs font-mono text-slate-100 truncate flex-1"
            title={drillPath}
          >
            {drillPath}
          </span>
          <span className="text-[10px] font-mono text-slate-500 shrink-0">
            {window.activeCount} active / {total} total
          </span>
        </div>

        {/* Passage rows — idx + truncated preview. Click opens the standard
            PassageModal (MORE + PREV|NEXT scoped to this file). Read-only —
            no edit affordance per spec (Andy 2026-07-02). */}
        <ul className="flex-1 overflow-y-auto divide-y divide-slate-800/60 list-none">
          {total === 0 ? (
            <li className="py-6 text-xs text-slate-500 italic text-center list-none">
              No passages found for this file.
            </li>
          ) : (
            window.passages.map((p) => (
              <li key={p.idx} className="list-none">
                <button
                  type="button"
                  onClick={() => openDrilledPassage(p)}
                  className={`w-full text-left px-4 py-2 flex items-start gap-3 text-xs
                              hover:bg-slate-700/20 transition-colors cursor-pointer ${
                                p.tombstoned ? 'opacity-50' : ''
                              }`}
                  title={`Open passage #${p.idx}`}
                >
                  <span className="font-mono text-[10px] text-slate-500 w-10 shrink-0 mt-0.5">
                    #{p.idx}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div
                      className={`text-xs truncate ${
                        p.tombstoned ? 'text-slate-500 line-through' : 'text-slate-200'
                      }`}
                      title={p.title}
                    >
                      {p.title || '[empty]'}
                    </div>
                    {p.preview && (
                      <div className="text-[10px] text-slate-500 truncate mt-0.5">
                        {p.preview}
                      </div>
                    )}
                  </div>
                </button>
              </li>
            ))
          )}
        </ul>

        {/* Pagination — same Prev/Next/jump pattern as the file list. */}
        {total > DRILL_PAGE_SIZE && (
          <div className="px-4 py-2 border-t border-slate-800 flex items-center justify-between gap-2 text-xs text-slate-500">
            <button
              onClick={drillPrev}
              disabled={drillOffset === 0}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Previous page"
            >
              <ChevronLeft size={11} />
              Prev
            </button>
            <div className="flex items-center gap-2">
              <span className="font-mono">
                Page {currentDrillPage} of {totalDrillPages}
              </span>
              <span className="text-slate-600">·</span>
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  const n = parseInt(drillPageJump, 10)
                  if (!isNaN(n)) drillGoPage(n)
                  setDrillPageJump('')
                }}
                className="flex items-center gap-1"
              >
                <span className="text-[10px] uppercase tracking-wider">jump</span>
                <input
                  type="text"
                  value={drillPageJump}
                  onChange={(e) => setDrillPageJump(e.target.value)}
                  placeholder={String(currentDrillPage)}
                  className="w-12 rounded bg-slate-950/60 border border-slate-800 px-1.5 py-0.5 text-[11px] text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
                  aria-label="Jump to page"
                />
              </form>
            </div>
            <button
              onClick={drillNext}
              disabled={drillOffset + DRILL_PAGE_SIZE >= total}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Next page"
            >
              More
              <ChevronRight size={11} />
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      <div className="h-full flex flex-col border border-slate-700/50 rounded-xl bg-slate-800/20 overflow-hidden">
        {/* Header — cart name + TOC label */}
        <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <BookOpen size={14} className="text-purple-400 shrink-0" />
            <h2 className="text-sm font-semibold text-slate-200 truncate" title={cartName}>
              {cartName}
              <span className="text-slate-500 font-normal ml-1.5 text-xs uppercase tracking-wider">TOC</span>
            </h2>
          </div>
          {hasBriefing && (
            <button
              onClick={() => setBriefingOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold uppercase tracking-wider
                         bg-purple-500/20 border border-purple-500/50 text-purple-200
                         hover:bg-purple-500/30 hover:text-purple-100 transition-colors shrink-0"
              title="View agent briefing"
            >
              <Sparkles size={11} />
              Briefing
            </button>
          )}
        </div>

        {/* Metadata row — Created By / Created on / Owner. Hidden when all
            three are empty (derived carts). */}
        {(data.creator || data.created_at || data.owner) && (
          <div className="px-4 py-2 border-b border-slate-800 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-400">
            {data.creator && (
              <span>
                <span className="text-slate-500">Created with</span> <span className="text-slate-300">{data.creator}</span>
              </span>
            )}
            {data.created_at && (
              <span>
                <span className="text-slate-500">Created on</span> <span className="text-slate-300 font-mono">{data.created_at}</span>
              </span>
            )}
            {data.owner && (
              <span>
                <span className="text-slate-500">Owner</span> <span className="text-slate-300">{data.owner}</span>
              </span>
            )}
          </div>
        )}

        {/* Derived-stats banner. */}
        {data.is_derived && (
          <div className="px-4 py-1.5 border-b border-slate-800 flex items-center gap-2 text-[10px] text-amber-300 bg-amber-500/5">
            <Info size={11} className="shrink-0" />
            No metadata available — showing derived stats
          </div>
        )}

        {/* Description — truncated server-side at ~200 words with ellipsis. */}
        {data.description && (
          <div className="px-4 py-2 border-b border-slate-800 text-xs text-slate-300 leading-relaxed">
            {data.description}
          </div>
        )}

        {/* Filter input. Client-side only per spec. */}
        <div className="px-4 py-2 border-b border-slate-800 flex items-center gap-2">
          <FilterIcon size={11} className="text-slate-500 shrink-0" />
          <input
            type="text"
            value={filter}
            onChange={(e) => { setFilter(e.target.value); setOffset(0) }}
            placeholder="Filter TOC…"
            className="flex-1 bg-transparent text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none"
            aria-label="Filter table of contents"
          />
          {filter && (
            <button
              onClick={() => { setFilter(''); setOffset(0) }}
              className="text-slate-500 hover:text-slate-300 p-0.5"
              title="Clear filter"
              aria-label="Clear filter"
            >
              <X size={11} />
            </button>
          )}
        </div>

        {/* List — bulleted per Andy's preference for cleaner look. */}
        <ul className="flex-1 overflow-y-auto divide-y divide-slate-800/60 list-none px-4 py-1">
          {pageItems.length === 0 ? (
            <li className="py-6 text-xs text-slate-500 italic text-center list-none">
              {filter
                ? `No items match "${filter}".`
                : 'This cart has no TOC entries.'}
            </li>
          ) : (
            pageItems.map((item, i) => {
              const clickable = drillEnabled && item.pattern_count > 0
              const rowInner = (
                <>
                  <span className="text-slate-600 mt-0.5 shrink-0" aria-hidden>•</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2">
                      <span
                        className={`truncate ${clickable ? 'text-slate-200 group-hover:text-purple-200' : 'text-slate-200'}`}
                        title={item.name}
                      >
                        {item.name}
                      </span>
                      {item.pattern_count > 0 && (
                        <span className="text-[10px] text-slate-500 font-mono shrink-0">
                          {item.pattern_count}p
                        </span>
                      )}
                    </div>
                    {item.description && (
                      <div className="text-[10px] text-slate-500 truncate mt-0.5" title={item.description}>
                        {item.description}
                      </div>
                    )}
                  </div>
                </>
              )
              if (clickable) {
                return (
                  <li key={`${offset + i}-${item.name}`} className="list-none">
                    <button
                      type="button"
                      onClick={() => {
                        setDrillPath(item.name)
                        setDrillOffset(0)
                        setDrillPageJump('')
                      }}
                      className="w-full py-1.5 flex items-start gap-2 text-xs text-left group hover:bg-slate-700/20 rounded transition-colors px-1 -mx-1"
                      title={`Show passages from ${item.name}`}
                    >
                      {rowInner}
                    </button>
                  </li>
                )
              }
              return (
                <li key={`${offset + i}-${item.name}`} className="py-1.5 flex items-start gap-2 text-xs">
                  {rowInner}
                </li>
              )
            })
          )}
        </ul>

        {/* Pagination — MORE + JUMP pattern matching CRUDScreen PassageBrowser. */}
        {filteredItems.length > TOC_PAGE_SIZE && (
          <div className="px-4 py-2 border-t border-slate-800 flex items-center justify-between gap-2 text-xs text-slate-500">
            <button
              onClick={goPrev}
              disabled={offset === 0}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Previous page"
            >
              <ChevronLeft size={11} />
              Prev
            </button>
            <div className="flex items-center gap-2">
              <span className="font-mono">
                Page {currentPage} of {totalPages}
              </span>
              <span className="text-slate-600">·</span>
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  const n = parseInt(pageJump, 10)
                  if (!isNaN(n)) goPage(n)
                  setPageJump('')
                }}
                className="flex items-center gap-1"
              >
                <span className="text-[10px] uppercase tracking-wider">jump</span>
                <input
                  type="text"
                  value={pageJump}
                  onChange={(e) => setPageJump(e.target.value)}
                  placeholder={String(currentPage)}
                  className="w-12 rounded bg-slate-950/60 border border-slate-800 px-1.5 py-0.5 text-[11px] text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
                  aria-label="Jump to page"
                />
              </form>
            </div>
            <button
              onClick={goNext}
              disabled={offset + TOC_PAGE_SIZE >= filteredItems.length}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Next page"
            >
              More
              <ChevronRight size={11} />
            </button>
          </div>
        )}
      </div>

      {briefingOpen && hasBriefing && (
        <BriefingModal
          briefing={data.agent_briefing || ''}
          onClose={() => setBriefingOpen(false)}
        />
      )}
    </>
  )
}
