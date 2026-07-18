import { useEffect, useState } from 'react'
import { Search, Loader2, X, Filter } from 'lucide-react'
import { useAppStore } from '../store/appStore'

export default function SearchBar() {
  const {
    doSearch, searching, status, query, topK, setTopK,
    strictMode, setStrictMode, exactMatch, setExactMatch,
  } = useAppStore()
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const pendingSourceFocus = useAppStore((s) => s.pendingSourceFocus)
  const consumePendingSourceFocus = useAppStore((s) => s.consumePendingSourceFocus)
  const [input, setInput] = useState('')

  // Clear the local input draft whenever the active cart changes (either
  // server-mounted or local). Otherwise stale text could re-run against the
  // new cart unintentionally.
  useEffect(() => {
    setInput('')
  }, [status?.mounted_cartridge, activeLocalCart])

  // 2026-07-13 (source-file links): when the user clicks a
  // vps://source/{slug} link inside a rendered report, the store flips
  // activeScreen to 'search' AND stashes a pendingSourceFocus payload.
  // Here we consume it: prefill the input box with the source display
  // name and fire a search. Fallback behavior — no per-source filter
  // exists yet on Search; we're doing a search-by-name so at least the
  // relevant passages surface. Documented as + polish opportunity
  // in the AGENT-LOGBOOK.
  useEffect(() => {
    if (!pendingSourceFocus) return
    const hasCart = !!status?.mounted_cartridge || !!activeLocalCart
    if (!hasCart) {
      // Leave the focus payload set so the search fires once the user
      // mounts a cart. If they never do, the payload just sits — a
      // subsequent click replaces it (only the most recent focus wins).
      return
    }
    const focus = consumePendingSourceFocus()
    if (!focus) return
    setInput(focus.displayName)
    doSearch(focus.displayName)
  }, [pendingSourceFocus, status?.mounted_cartridge, activeLocalCart, consumePendingSourceFocus, doSearch])

  const hasActiveCart = !!status?.mounted_cartridge || !!activeLocalCart
  const disabled = !hasActiveCart || searching
  const hasResults = !!query

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim().length > 2 && !disabled) {
      doSearch(input.trim())
    }
  }

  const handleClear = () => {
    setInput('')
    useAppStore.setState({ query: '', results: [], searchModeLabel: '', searchElapsed: 0 })
  }

  return (
    <div className="flex flex-col gap-2">
    <form onSubmit={handleSubmit} className="flex gap-3">
      <div className="flex-1 relative">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') e.preventDefault() }}
          placeholder={hasActiveCart ? 'What are you looking for?' : 'Mount a cartridge first...'}
          disabled={!hasActiveCart}
          className="w-full pl-11 pr-10 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl
                     text-slate-200 placeholder-slate-600 focus:outline-none focus:border-purple-500/50
                     focus:ring-1 focus:ring-purple-500/20 transition-all disabled:opacity-50"
        />
        {/* Clear button -- visible when there are results */}
        {hasResults && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 px-2 py-0.5 rounded text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-700/50 transition-colors"
            title="Clear search results"
          >
            <X size={12} />
            Clear
          </button>
        )}
      </div>

      {/* Top-K selector */}
      <select
        value={topK}
        onChange={(e) => setTopK(Number(e.target.value))}
        className="px-2 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl text-slate-300 text-sm
                   focus:outline-none focus:border-purple-500/50 cursor-pointer"
        title="Number of results to return"
      >
        <option value={5}>Top 5</option>
        <option value={10}>Top 10</option>
        <option value={20}>Top 20</option>
        <option value={50}>Top 50</option>
      </select>

      <button
        type="submit"
        disabled={disabled || input.trim().length < 3}
        className="px-6 py-3 rounded-xl gradient-bg text-white font-medium
                   hover:opacity-90 transition-opacity disabled:opacity-40
                   flex items-center gap-2"
      >
        {searching ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            Searching
          </>
        ) : (
          <>
            <Search size={16} />
            Search
          </>
        )}
      </button>
    </form>

    {/* Filter toggles — under the Search button, above the TOC/results.
        pre-existed here and got mis-routed into ResultsList
        during the 7-01 layout reshuffle; restored to the original slot so the
        UX location matches user expectation on first search.
        justify-end so the row aligns under the Search button
        on the right side of the input row (matches the original screenshot). */}
    <div className="flex items-center justify-end gap-4 pr-1 text-sm text-slate-500">
      <label
        className="flex items-center gap-1.5 cursor-pointer select-none"
        title="When enabled, only results whose text contains at least one query keyword are shown"
      >
        <input
          type="checkbox"
          checked={strictMode}
          onChange={(e) => { setStrictMode(e.target.checked); if (e.target.checked) setExactMatch(false) }}
          className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-purple-500 focus:ring-purple-500/30 focus:ring-offset-0 cursor-pointer accent-purple-500"
        />
        <Filter size={12} className={strictMode ? 'text-purple-400' : 'text-slate-600'} />
        <span className={`text-xs ${strictMode ? 'text-purple-400' : 'text-slate-500'}`}>
          Must contain keywords
        </span>
      </label>

      <label
        className="flex items-center gap-1.5 cursor-pointer select-none"
        title="When enabled, only results containing the exact query phrase are shown (e.g. &quot;Reed Richards&quot; won't match Keith Richards)"
      >
        <input
          type="checkbox"
          checked={exactMatch}
          onChange={(e) => { setExactMatch(e.target.checked); if (e.target.checked) setStrictMode(false) }}
          className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-purple-500 focus:ring-purple-500/30 focus:ring-offset-0 cursor-pointer accent-purple-500"
        />
        <Filter size={12} className={exactMatch ? 'text-amber-400' : 'text-slate-600'} />
        <span className={`text-xs ${exactMatch ? 'text-amber-400' : 'text-slate-500'}`}>
          Must be exact match
        </span>
      </label>
    </div>
    </div>
  )
}
