import { useMemo } from 'react'
import { useAppStore } from '../store/appStore'
import ResultCard from './ResultCard'
import { Clock, Loader2, Search, Zap } from 'lucide-react'

const STOP_WORDS = new Set([
  'the', 'and', 'but', 'for', 'nor', 'not', 'yet', 'are', 'was', 'were',
  'has', 'had', 'have', 'does', 'did', 'will', 'can', 'may', 'use', 'its',
  'his', 'her', 'our', 'who', 'how', 'all', 'any', 'this', 'that', 'they',
  'them', 'then', 'than', 'these', 'those', 'with', 'from', 'into', 'each',
  'when', 'where', 'what', 'which', 'there', 'their', 'been', 'being',
  'would', 'could', 'should', 'about', 'also', 'just', 'more', 'some',
  'only', 'very', 'such', 'do', 'so', 'if', 'or', 'as', 'at', 'by',
  'in', 'is', 'it', 'no', 'of', 'on', 'to', 'up', 'we', 'an', 'be',
  'he', 'me', 'show', 'find', 'tell', 'give', 'list', 'get', 'want',
  'need', 'look', 'make', 'let', 'see', 'know',
])

/** Extract meaningful keywords from a query string. */
function extractKeywords(query: string): string[] {
  return query
    .split(/\s+/)
    .map((w) => w.toLowerCase().replace(/[^a-z0-9]/g, ''))
    .filter((w) => w.length >= 2 && !STOP_WORDS.has(w))
}

/** Check if text contains at least one keyword (case-insensitive substring). */
function textContainsKeyword(text: string, keywords: string[]): boolean {
  const lower = text.toLowerCase()
  return keywords.some((kw) => lower.includes(kw))
}

/** Check if text contains the exact phrase (case-insensitive). */
function textContainsExactPhrase(text: string, phrase: string): boolean {
  return text.toLowerCase().includes(phrase.toLowerCase())
}

export default function ResultsList() {
  // Note: strictMode / exactMatch toggles moved to SearchBar (Andy 2026-07-02);
  // ResultsList still reads them here to compute filteredResults + the
  // "N of M results" stat, but the checkboxes themselves live above the TOC now.
  const { results, searchModeLabel, searchElapsed, query, status, strictMode, exactMatch, searching, searchMode } = useAppStore()
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const localCarts = useAppStore((s) => s.localCarts)
  const deletedPatterns = useAppStore((s) => s.deletedPatterns)

  const keywords = useMemo(() => extractKeywords(query), [query])

  // Andy 2026-07-03 live tombstone filter: when a user edits a passage in
  // Edit Carts, the old idx gets tombstoned but any results array computed
  // before the edit still carries it. Filter tombstoned idx out at render
  // time (LocalCart tombstones for browser mounts, deletedPatterns for
  // backend mounts) so edits reflect in already-visible results without a
  // re-search. Untombstoning restores the card on the next render.
  const tombstonedIdx = useMemo(() => {
    if (activeLocalCart) {
      const cart = localCarts.get(activeLocalCart)
      return cart?.tombstones ?? new Set<number>()
    }
    return new Set<number>(deletedPatterns.map((d) => d.idx))
  }, [activeLocalCart, localCarts, deletedPatterns])

  const filteredResults = useMemo(() => {
    let filtered = results.filter((r) => !tombstonedIdx.has(r.idx))

    // Exact phrase match -- full query string must appear as-is
    if (exactMatch && query.trim().length > 0) {
      filtered = filtered.filter((r) => {
        const text = r.full_text || r.preview || r.title
        return textContainsExactPhrase(text, query.trim())
      })
    }

    // Keyword filter -- at least one extracted keyword must appear
    if (strictMode && keywords.length > 0 && !exactMatch) {
      filtered = filtered.filter((r) => {
        const text = r.full_text || r.preview || r.title
        return textContainsKeyword(text, keywords)
      })

      // Andy 2026-06-17 PM: rerank when strictMode is on. Pure cosine on
      // long-form text can put a slightly-denser-context result above a
      // verbatim phrase match by a 0.008 hair (e.g. "save to disk" → first
      // result at 0.573 beats literal "save to disk:" line at 0.565). The
      // user who turned ON Must-contain-keywords is asking for keyword
      // weight, so we bubble verbatim phrase matches to the top, then
      // higher keyword density, then preserve original cosine order for
      // ties. Stopwords are still stripped from `keywords` so the density
      // score reflects content words; the PHRASE check uses the raw query
      // so "save to disk" matches as a literal substring.
      const phrase = query.trim().toLowerCase()
      const rankedCopy = filtered.map((r, originalIdx) => {
        const text = (r.full_text || r.preview || r.title).toLowerCase()
        const hasPhrase = phrase.length > 0 && text.includes(phrase) ? 1 : 0
        const keywordHits = keywords.reduce(
          (n, kw) => (text.includes(kw) ? n + 1 : n),
          0,
        )
        return { r, originalIdx, hasPhrase, keywordHits }
      })
      rankedCopy.sort((a, b) => {
        if (a.hasPhrase !== b.hasPhrase) return b.hasPhrase - a.hasPhrase
        if (a.keywordHits !== b.keywordHits) return b.keywordHits - a.keywordHits
        return a.originalIdx - b.originalIdx
      })
      filtered = rankedCopy.map((x) => x.r)
    }

    return filtered
  }, [results, strictMode, exactMatch, keywords, query, tombstonedIdx])

  if (!status?.mounted_cartridge && !activeLocalCart) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
        <div className="w-16 h-16 rounded-2xl bg-slate-800/50 flex items-center justify-center mb-4">
          <Zap size={28} />
        </div>
        <p className="text-lg font-medium">Welcome to Vector+ Studio</p>
        <p className="text-sm mt-1">Mount a cartridge from the picker above to begin</p>
      </div>
    )
  }

  const totalPatterns = activeLocalCart
    ? (localCarts.get(activeLocalCart)?.passages.length ?? 0)
    : (status?.pattern_count ?? 0)

  if (!query) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
        <div className="w-16 h-16 rounded-2xl bg-slate-800/50 flex items-center justify-center mb-4">
          <Search size={28} />
        </div>
        <p className="text-lg font-medium">Ready to search</p>
        <p className="text-sm mt-1">{totalPatterns.toLocaleString()} patterns loaded</p>
      </div>
    )
  }

  const filtered = (strictMode || exactMatch) && filteredResults.length !== results.length

  const modeLabels: Record<string, string> = {
    smart: 'Smart Search (physics + cosine)',
    pure_brain: 'Pure Brain (lattice physics)',
    fast: 'Fast Search (cosine)',
  }

  if (searching) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-400">
        <Loader2 size={40} className="animate-spin text-purple-400 mb-4" />
        <p className="text-lg font-medium">{modeLabels[searchMode] || 'Searching'}...</p>
        <p className="text-sm text-slate-600 mt-1">
          {searchMode === 'pure_brain' ? 'Settling lattice physics — this takes a moment' : 'Querying'}
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Stats bar — checkboxes moved to SearchBar (Andy 2026-07-02). */}
      <div className="flex items-center gap-4 mb-4 pr-4 text-sm text-slate-500">
        <span className="font-medium text-slate-300">
          {filtered ? `${filteredResults.length} of ${results.length}` : filteredResults.length} results
        </span>
        <span className="flex items-center gap-1">
          <Clock size={12} /> {searchElapsed.toFixed(0)}ms
        </span>
        <span className="px-2 py-0.5 rounded bg-slate-800/60 text-xs">{searchModeLabel}</span>
      </div>

      {/* Results */}
      <div className="space-y-3">
        {filteredResults.map((r) => (
          <ResultCard key={`${r.idx}-${r.rank}`} result={r} />
        ))}
      </div>

      {filteredResults.length === 0 && (
        <p className="text-center text-slate-600 mt-8">
          {filtered ? 'No results match your filter. Try unchecking the filter.' : 'No results found'}
        </p>
      )}
    </div>
  )
}
