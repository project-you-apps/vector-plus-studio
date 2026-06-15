import { useMemo, useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight, BookOpen, Pencil, Trash2, X, Zap, Lock, Footprints, Image as ImageIcon } from 'lucide-react'
import type { SearchResult } from '../api/types'
import { useAppStore } from '../store/appStore'

/**
 * Parse a figure-passage header emitted by the artifact ingestor:
 *   [figure | source: foo.pdf | page: 12 | hash: abc123 | size: 800x600 | format: png ]
 *   caption: ...
 */
function parseFigureHeader(text: string): null | {
  hash: string
  source: string
  page: number
  size: string
  format: string
  caption: string
} {
  if (!text || !text.startsWith('[figure')) return null
  const nl = text.indexOf('\n')
  if (nl === -1) return null
  const header = text.slice(0, nl)
  const body = text.slice(nl + 1).trim()
  const hash = header.match(/hash:\s*([a-f0-9]+)/)?.[1]
  if (!hash) return null
  return {
    hash,
    source: header.match(/source:\s*([^|\]]+)/)?.[1]?.trim() ?? '',
    page: parseInt(header.match(/page:\s*(\d+)/)?.[1] ?? '0', 10),
    size: header.match(/size:\s*(\S+)/)?.[1] ?? '',
    format: header.match(/format:\s*(\w+)/)?.[1]?.toLowerCase() ?? 'png',
    caption: body.replace(/^caption:\s*/, '').trim(),
  }
}

/**
 * Module-level cache of figure blob URLs keyed by `${cartName}::${filename}`.
 * Each figure produces exactly one URL on first sight; URLs live until page
 * reload. This sidesteps React's re-render churn around useMemo + useEffect
 * cleanup that was revoking URLs before <img> could fetch them
 * (ERR_FILE_NOT_FOUND in the console). Memory is bounded by the number of
 * distinct figures the user has displayed in this session.
 */
const FIGURE_URL_CACHE: Map<string, string> = new Map()

function getFigureUrl(cartName: string, filename: string, bytes: Uint8Array, format: string): string {
  const key = `${cartName}::${filename}`
  const cached = FIGURE_URL_CACHE.get(key)
  if (cached) return cached
  // Copy into a fresh ArrayBuffer so TypeScript's Blob ctor accepts it even
  // when the source byte view is backed by a SharedArrayBuffer (rare; never
  // happens in our path, but TS doesn't know that).
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  const blob = new Blob([copy.buffer], { type: `image/${format}` })
  const url = URL.createObjectURL(blob)
  FIGURE_URL_CACHE.set(key, url)
  return url
}

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
export function highlightText(text: string, query: string): ReactNode {
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

/** Render text with http(s) URLs wrapped in <a target="_blank"> AND query-keyword highlighting
 *  applied to non-URL segments. URLs themselves are not highlighted (would risk breaking the link).
 *  Trailing punctuation (.,;:!?) immediately after a URL is excluded from the link.
 *
 *  Ported from the Membot demo's provenance feature 2026-05-04 — VPS-shaped (React node tree
 *  rather than HTML string injection, so structurally safer than the Membot version). */
export function renderTextWithLinks(text: string, query: string): ReactNode {
  if (!text) return text
  const URL_RE = /(https?:\/\/[^\s<]+?)([.,;:!?)\]]?)(?=\s|$|<)/g
  const out: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let key = 0
  while ((match = URL_RE.exec(text)) !== null) {
    // Non-URL text before this URL — highlight it
    if (match.index > lastIndex) {
      const before = text.slice(lastIndex, match.index)
      out.push(<span key={`t-${key++}`}>{highlightText(before, query)}</span>)
    }
    const url = match[1]
    const trailing = match[2]
    out.push(
      <a
        key={`u-${key++}`}
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-cyan-400 underline decoration-dotted hover:decoration-solid"
      >
        {url}
      </a>
    )
    if (trailing) out.push(trailing)
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    const after = text.slice(lastIndex)
    out.push(<span key={`t-${key++}`}>{highlightText(after, query)}</span>)
  }
  return out.length > 0 ? <>{out}</> : highlightText(text, query)
}

interface Props {
  result: SearchResult
}

export default function ResultCard({ result }: Props) {
  const [expanded, setExpanded] = useState(false)
  const deleteResult = useAppStore((s) => s.deleteResult)
  const openEditor = useAppStore((s) => s.openEditor)
  const openModal = useAppStore((s) => s.openModal)
  const query = useAppStore((s) => s.query)
  const confirmDeleteIdx = useAppStore((s) => s.confirmDeleteIdx)
  const setConfirmDeleteIdx = useAppStore((s) => s.setConfirmDeleteIdx)
  const status = useAppStore((s) => s.status)
  const cartridges = useAppStore((s) => s.cartridges)
  const webgpuStatus = useAppStore((s) => s.webgpuStatus)
  const walkFrom = useAppStore((s) => s.walkFrom)
  const walkTrail = useAppStore((s) => s.walkTrail)
  const isWalking = walkTrail.length > 0
  const searching = useAppStore((s) => s.searching)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const localCarts = useAppStore((s) => s.localCarts)
  const isConfirming = confirmDeleteIdx === result.idx

  // Figure detection — if this result is a passage emitted by the artifact
  // ingestor's PDF figure extractor, parse the header, look up the bytes from
  // the active local cart's figures Map, and hand back a cached blob URL.
  const figureMeta = useMemo(() => parseFigureHeader(result.full_text || ''), [result.full_text])
  const figureUrl: string | null = (() => {
    if (!figureMeta || !activeLocalCart) return null
    const cart = localCarts.get(activeLocalCart)
    if (!cart) return null
    const filename = `${figureMeta.hash}.${figureMeta.format}`
    const bytes = cart.figures.get(filename)
    if (!bytes) return null
    return getFigureUrl(activeLocalCart, filename, bytes, figureMeta.format)
  })()

  // In walk mode the user isn't searching for keywords — they're physics-walking.
  // Suppress text-keyword highlighting so old query words don't reinforce a stale
  // frame on the walk results. The trail dropdown already communicates the path.
  const highlightQuery = isWalking ? '' : query

  // Walk button is available whenever Associate physics is available:
  // either server CUDA (gpu_available + physics_trained) or browser WebGPU
  // with a brain file on the currently-mounted cart. Local-disk carts have
  // neither in F1-A — Walk surfaces again once we wire brain/sigs alongside
  // the cart fetch (deferred).
  const mountedName = status?.mounted_cartridge ?? null
  const mountedCart = mountedName ? cartridges.find((c) => c.name === mountedName) : null
  const cartHasBrain = !!mountedCart?.has_brain
  const walkAvailable =
    !activeLocalCart && !!mountedName && (
      (!!status?.gpu_available && !!status?.physics_trained) ||
      (webgpuStatus === 'available' && cartHasBrain)
    )
  const hasLinks = result.prev_idx != null || result.next_idx != null

  // Compute write-availability from the same three layers Header.tsx checks:
  //   1. Server-wide VPS_READ_ONLY env var (status.read_only_mode)
  //   2. Cart-declared permissions sidecar (status.cart_permissions.default)
  //   3. Per-session lock toggle (status.read_only)
  // Plus pattern-level: result.perms.w from the hippocampus row.
  // When any layer denies writes, Edit + Delete are disabled.
  const cartDefault = String(status?.cart_permissions?.default ?? 'rw')
  const cartReadOnly =
    !!status?.read_only_mode
    || !cartDefault.includes('w')
    || !!status?.read_only
    || !!activeLocalCart  // local-disk carts are read-only in F1-A (RW writeback comes later)
  const patternLocked = !!(result.perms && !result.perms.w)
  const writeDisabled = cartReadOnly || patternLocked

  // Surface the specific reason in tooltips so the user knows WHY a button
  // is disabled (server vs cart vs session vs pattern lock).
  const disabledReason =
    activeLocalCart
      ? 'Local-disk cart is read-only in this version (RW writeback to your file is a later feature).'
    : status?.read_only_mode
      ? 'Server is in read-only mode (public demo).'
    : !cartDefault.includes('w')
      ? `Cart is read-only (.permissions.json default="${cartDefault}").`
    : status?.read_only
      ? 'Cart is currently locked. Click the lock icon in the header to unlock.'
    : patternLocked
      ? 'This pattern is locked at the hippocampus level.'
    : null

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

        {/* Title + preview (click to expand) */}
        <div className="flex-1 min-w-0 cursor-pointer" onClick={() => setExpanded(!expanded)}>
          {/* Provenance v1 sidecar — source filename per result. Renders only
              when the cart's source_paths.npy is present (browser-built carts
              2026-06-15+). Quiet slate-500 caption above the title so it
              doesn't fight the title for visual attention but is always
              available for traceability. Hover to see the full path. */}
          {result.source_path && (
            <div className="text-[10px] text-slate-500 font-mono truncate mb-0.5" title={result.source_path}>
              from <span className="text-slate-400">{result.source_path}</span>
            </div>
          )}
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-medium text-slate-200 truncate">
              {figureMeta
                ? `Figure · ${figureMeta.source}${figureMeta.page > 0 ? `, page ${figureMeta.page}` : ''}`
                : result.title}
            </h3>
            {figureMeta && (
              <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-medium flex items-center gap-1">
                <ImageIcon size={8} /> FIGURE{figureMeta.size ? ` ${figureMeta.size}` : ''}
              </span>
            )}
            {result.from_lattice && (
              <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full gradient-bg text-white font-medium flex items-center gap-1">
                <Zap size={8} /> FROM LATTICE
              </span>
            )}
            {result.perms && !result.perms.w && (
              <span
                className="shrink-0 text-[10px] px-2 py-0.5 rounded-full bg-slate-700/50 text-slate-400 border border-slate-600/50 font-medium flex items-center gap-1"
                title={`Pattern locked at the hippocampus level (perms=0x${result.perms.raw.toString(16)}). Edit attempts will return 403.`}
              >
                <Lock size={8} /> LOCKED
              </span>
            )}
          </div>
          {figureMeta ? (
            <div className="flex gap-3 items-start">
              {figureUrl ? (
                <img
                  src={figureUrl}
                  alt={figureMeta.caption || `Figure from ${figureMeta.source}`}
                  className="h-24 w-24 object-contain bg-slate-900/60 rounded border border-slate-700/40 shrink-0"
                />
              ) : (
                <div className="h-24 w-24 bg-slate-900/40 rounded border border-dashed border-slate-700/40 flex items-center justify-center text-slate-600 text-[10px] text-center px-1 shrink-0">
                  no image bytes
                </div>
              )}
              <p className="text-sm text-slate-400 line-clamp-3 flex-1">
                {figureMeta.caption || <span className="italic text-slate-600">(no caption detected)</span>}
              </p>
            </div>
          ) : result.preview ? (
            <p className="text-sm text-slate-500 line-clamp-2">{renderTextWithLinks(result.preview, highlightQuery)}</p>
          ) : null}
        </div>

        {/* Scores */}
        <div className="shrink-0 text-right">
          <div className="text-lg font-bold text-slate-200">{result.score.toFixed(3)}</div>
          {result.cosine_score != null && result.hamming_score != null ? (
            <div className="text-[10px] text-slate-500 space-x-2">
              <span>C:{result.cosine_score.toFixed(3)}</span>
              <span className="text-cyan-400">H:{result.hamming_score.toFixed(3)}</span>
              {result.keyword_boost != null && result.keyword_boost > 0 && (
                <span className="text-amber-400">+{result.keyword_boost.toFixed(3)}</span>
              )}
            </div>
          ) : result.cosine_score != null && result.physics_score != null ? (
            <div className="text-[10px] text-slate-500 space-x-2">
              <span>C:{result.cosine_score.toFixed(3)}</span>
              <span className="text-purple-400">P:{result.physics_score.toFixed(3)}</span>
            </div>
          ) : null}
        </div>

        {/* Actions */}
        <div className="shrink-0 flex items-center gap-1">
          {walkAvailable && (
            <button
              onClick={() => walkFrom(result.idx, result.title)}
              disabled={searching}
              className={`p-2 rounded-lg transition-colors ${
                searching
                  ? 'text-slate-600 opacity-40 cursor-not-allowed'
                  : 'hover:bg-slate-700/50 text-slate-500 hover:text-cyan-400'
              }`}
              title="Walk from here — run Associate anchored on this passage's embedding"
            >
              <Footprints size={16} />
            </button>
          )}
          <button
            onClick={() => openEditor(result.full_text, result.idx)}
            disabled={writeDisabled}
            className={`p-2 rounded-lg transition-colors ${
              writeDisabled
                ? 'text-slate-600 opacity-40 cursor-not-allowed'
                : 'hover:bg-slate-700/50 text-slate-500 hover:text-purple-400'
            }`}
            title={writeDisabled ? (disabledReason ?? 'Read-only') : 'Edit passage'}
          >
            <Pencil size={16} />
          </button>
          <button
            onClick={() => setConfirmDeleteIdx(isConfirming ? null : result.idx)}
            disabled={writeDisabled}
            className={`p-2 rounded-lg transition-colors ${
              writeDisabled
                ? 'text-slate-600 opacity-40 cursor-not-allowed'
                : isConfirming
                ? 'bg-red-500/20 text-red-400'
                : 'hover:bg-slate-700/50 text-slate-500 hover:text-red-400'
            }`}
            title={writeDisabled ? (disabledReason ?? 'Read-only') : 'Delete'}
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
          {figureMeta && figureUrl && (
            <div className="mb-3 flex justify-center">
              <img
                src={figureUrl}
                alt={figureMeta.caption || `Figure from ${figureMeta.source}`}
                className="max-h-96 max-w-full object-contain bg-slate-950 rounded border border-slate-700/40"
              />
            </div>
          )}
          <pre className="text-sm text-slate-400 whitespace-pre-wrap font-mono leading-relaxed max-h-96 overflow-y-auto">
            {result.full_text ? renderTextWithLinks(result.full_text, highlightQuery) : '[No text available]'}
          </pre>
          {hasLinks && (
            <button
              onClick={() => openModal(result)}
              className="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 text-xs font-medium transition-colors"
            >
              <BookOpen size={12} />
              MORE
            </button>
          )}
        </div>
      )}
    </div>
  )
}
