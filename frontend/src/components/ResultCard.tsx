import { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight, Pencil, Trash2, X, Zap } from 'lucide-react'
import type { SearchResult } from '../api/types'
import { useAppStore } from '../store/appStore'

const STOP_WORDS = new Set([
  'the', 'and', 'but', 'for', 'nor', 'not', 'yet', 'are', 'was', 'were',
  'has', 'had', 'have', 'does', 'did', 'will', 'can', 'may', 'use', 'its',
  'his', 'her', 'our', 'who', 'how', 'all', 'any', 'this', 'that', 'they',
  'them', 'then', 'than', 'these', 'those', 'with', 'from', 'into', 'each',
  'when', 'where', 'what', 'which', 'there', 'their', 'been', 'being',
  'would', 'could', 'should', 'about', 'also', 'just', 'more', 'some',
  'only', 'very', 'such', 'do', 'so', 'if', 'or', 'as', 'at', 'by',
  'in', 'is', 'it', 'no', 'of', 'on', 'to', 'up', 'we', 'an', 'be',
  'he', 'me',
])

// Same suffix list as backend simple_stem (search.py)
const SUFFIXES = ['ings', 'ing', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ed', 'es', 's', 'ly']
const SUFFIX_RE = SUFFIXES.join('|')

/** Strip leading/trailing punctuation from a word. */
function cleanWord(word: string): string {
  return word.replace(/^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$/g, '')
}

/** Port of backend simple_stem(): strip common English suffixes. */
function simpleStem(word: string): string {
  word = word.toLowerCase()
  for (const suffix of SUFFIXES) {
    if (word.endsWith(suffix) && word.length > suffix.length + 2) {
      return word.slice(0, -suffix.length)
    }
  }
  return word
}

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Build a regex fragment that matches a word's stem + any common suffix.
 *  Dots/hyphens inside words become flexible separators matching [.\s\-]* */
function stemPattern(word: string): string {
  // If word contains dots or hyphens (e.g. "t.rex", "dot-product"),
  // split on them and join with flexible separator
  if (/[.\-]/.test(word)) {
    const parts = word.split(/[.\-]+/).filter(Boolean)
    if (parts.length >= 2) {
      return parts.map((p) => {
        const stem = simpleStem(p)
        return `${esc(stem)}(?:${SUFFIX_RE})?`
      }).join('[.\\s\\-]*')
    }
  }
  const stem = simpleStem(word)
  return `${esc(stem)}(?:${SUFFIX_RE})?`
}

/** Split text on query keywords and wrap matches in highlighted spans. */
function highlightText(text: string, query: string): ReactNode {
  if (!query.trim()) return text

  // Clean punctuation, filter: 2+ chars and not a stop word
  const words = query.split(/\s+/).map(cleanWord).filter((w) => w.length >= 2 && !STOP_WORDS.has(w.toLowerCase()))
  if (words.length === 0) return text

  const patterns: string[] = []

  // Multi-word compound first (longer matches win): "dot product" matches "dot-product" and "dot product"
  if (words.length >= 2) {
    patterns.push(words.map((w) => stemPattern(w)).join('[\\s\\-]+'))
  }

  // Individual words with stem matching: "products" highlights "product", "productive", etc.
  for (const w of words) {
    patterns.push(stemPattern(w))
  }

  const re = new RegExp(`(${patterns.join('|')})`, 'gi')
  const parts = text.split(re)
  // split with capturing group: even indices = non-match, odd indices = match
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="bg-purple-500/30 text-purple-200 rounded px-0.5">{part}</mark>
    ) : (
      part
    )
  )
}

interface Props {
  result: SearchResult
}

export default function ResultCard({ result }: Props) {
  const [expanded, setExpanded] = useState(false)
  const deleteResult = useAppStore((s) => s.deleteResult)
  const openEditor = useAppStore((s) => s.openEditor)
  const query = useAppStore((s) => s.query)
  const confirmDeleteIdx = useAppStore((s) => s.confirmDeleteIdx)
  const setConfirmDeleteIdx = useAppStore((s) => s.setConfirmDeleteIdx)
  const isConfirming = confirmDeleteIdx === result.idx

  return (
    <div className="border border-slate-700/40 rounded-xl bg-slate-800/20 hover:bg-slate-800/40 transition-all overflow-hidden">
      {/* Header */}
      <div className="flex items-start gap-3 p-4">
        {/* Expand + Rank */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="shrink-0 flex items-center gap-1 p-1 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
          title={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="w-6 h-6 rounded bg-slate-800 flex items-center justify-center text-xs font-bold text-slate-400">
            {result.rank}
          </span>
        </button>

        {/* Title + preview */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-medium text-slate-200 truncate">{result.title}</h3>
            {result.from_lattice && (
              <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full gradient-bg text-white font-medium flex items-center gap-1">
                <Zap size={8} /> FROM LATTICE
              </span>
            )}
          </div>
          {result.preview && (
            <p className="text-sm text-slate-500 line-clamp-2">{highlightText(result.preview, query)}</p>
          )}
        </div>

        {/* Scores */}
        <div className="shrink-0 text-right">
          <div className="text-lg font-bold text-slate-200">{result.score.toFixed(3)}</div>
          {result.cosine_score != null && result.physics_score != null && (
            <div className="text-[10px] text-slate-500 space-x-2">
              <span>C:{result.cosine_score.toFixed(3)}</span>
              <span className="text-purple-400">P:{result.physics_score.toFixed(3)}</span>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="shrink-0 flex items-center gap-1">
          <button
            onClick={() => openEditor(result.full_text, result.idx)}
            className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-purple-400 transition-colors"
            title="Edit passage"
          >
            <Pencil size={16} />
          </button>
          <button
            onClick={() => setConfirmDeleteIdx(isConfirming ? null : result.idx)}
            className={`p-2 rounded-lg transition-colors ${
              isConfirming
                ? 'bg-red-500/20 text-red-400'
                : 'hover:bg-slate-700/50 text-slate-500 hover:text-red-400'
            }`}
            title="Delete"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      {/* Delete confirmation bar */}
      {isConfirming && (
        <div className="flex items-center justify-between px-4 py-2.5 bg-red-500/10 border-t border-red-500/20">
          <span className="text-sm text-red-300">Delete this pattern? This can be restored later.</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => deleteResult(result.idx)}
              className="px-3 py-1 text-xs font-medium rounded bg-red-500/30 text-red-300 hover:bg-red-500/50 transition-colors"
            >
              Delete
            </button>
            <button
              onClick={() => setConfirmDeleteIdx(null)}
              className="p-1 rounded hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
              title="Cancel"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Expanded text */}
      {expanded && (
        <div className="border-t border-slate-700/30 p-4 bg-slate-900/30">
          <pre className="text-sm text-slate-400 whitespace-pre-wrap font-mono leading-relaxed max-h-96 overflow-y-auto">
            {result.full_text ? highlightText(result.full_text, query) : '[No text available]'}
          </pre>
        </div>
      )}
    </div>
  )
}
