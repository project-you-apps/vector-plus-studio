import { useMemo } from 'react'
import { useAppStore } from '../store/appStore'
import ResultCard from './ResultCard'
import { Clock, Filter, Search, Zap } from 'lucide-react'

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

export default function ResultsList() {
  const { results, searchModeLabel, searchElapsed, query, status, strictMode, setStrictMode } = useAppStore()

  const keywords = useMemo(() => extractKeywords(query), [query])

  const filteredResults = useMemo(() => {
    if (!strictMode || keywords.length === 0) return results
    return results.filter((r) => {
      const text = r.full_text || r.preview || r.title
      return textContainsKeyword(text, keywords)
    })
  }, [results, strictMode, keywords])

  if (!status?.mounted_cartridge) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
        <div className="w-16 h-16 rounded-2xl bg-slate-800/50 flex items-center justify-center mb-4">
          <Zap size={28} />
        </div>
        <p className="text-lg font-medium">Welcome to Vector+ Studio</p>
        <p className="text-sm mt-1">Mount a cartridge from the sidebar to begin</p>
      </div>
    )
  }

  if (!query) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
        <div className="w-16 h-16 rounded-2xl bg-slate-800/50 flex items-center justify-center mb-4">
          <Search size={28} />
        </div>
        <p className="text-lg font-medium">Ready to search</p>
        <p className="text-sm mt-1">{status.pattern_count.toLocaleString()} patterns loaded</p>
      </div>
    )
  }

  const filtered = strictMode && keywords.length > 0 && filteredResults.length !== results.length

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Stats bar */}
      <div className="flex items-center gap-4 mb-4 text-sm text-slate-500">
        <span className="font-medium text-slate-300">
          {filtered ? `${filteredResults.length} of ${results.length}` : results.length} results
        </span>
        <span className="flex items-center gap-1">
          <Clock size={12} /> {searchElapsed.toFixed(0)}ms
        </span>
        <span className="px-2 py-0.5 rounded bg-slate-800/60 text-xs">{searchModeLabel}</span>

        {/* Strict filter toggle */}
        <label
          className="ml-auto flex items-center gap-1.5 cursor-pointer select-none"
          title="When enabled, only results whose text contains at least one query keyword are shown"
        >
          <input
            type="checkbox"
            checked={strictMode}
            onChange={(e) => setStrictMode(e.target.checked)}
            className="w-3.5 h-3.5 rounded border-slate-600 bg-slate-800 text-purple-500 focus:ring-purple-500/30 focus:ring-offset-0 cursor-pointer accent-purple-500"
          />
          <Filter size={12} className={strictMode ? 'text-purple-400' : 'text-slate-600'} />
          <span className={`text-xs ${strictMode ? 'text-purple-400' : 'text-slate-600'}`}>
            Must contain keywords
          </span>
        </label>
      </div>

      {/* Results */}
      <div className="space-y-3">
        {filteredResults.map((r) => (
          <ResultCard key={`${r.idx}-${r.rank}`} result={r} />
        ))}
      </div>

      {filteredResults.length === 0 && (
        <p className="text-center text-slate-600 mt-8">
          {filtered ? 'No results contain your keywords. Try unchecking the filter.' : 'No results found'}
        </p>
      )}
    </div>
  )
}
