import { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, X, Loader2, FolderOpen } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import { useAppStore } from '../store/appStore'

// Chunker overlap detection (Andy 2026-05-09 proposal). The browser-side
// chunker produces 300-word chunks with 50-word overlap — consecutive chunks
// share their boundary text. When user clicks Next/Prev to walk through
// search results in source order, the overlap creates a "stitched-together"
// feel: ~50 words repeat at every chunk boundary.
//
// Display-layer fix: detect the longest common suffix-of-prev that's
// prefix-of-current (or vice versa for Prev navigation), gray it out, and
// add a visual divider so the user understands they've already read it.
// Pure UX-layer fix; zero chunker / cart-format changes.
//
// Edge cases handled cleanly: section-boundary chunks (different markdown
// section / different file) have no overlap → algorithm returns 0 → no
// graying. Random-access from search results clears the prevText state so
// the first view is always full content, no false-positive graying.
function findOverlap(suffixOf: string, prefixOf: string, maxChars = 600): number {
  const cap = Math.min(maxChars, suffixOf.length, prefixOf.length)
  for (let len = cap; len >= 5; len--) {
    if (suffixOf.slice(-len) === prefixOf.slice(0, len)) {
      return len
    }
  }
  return 0
}

// Strip the chunker-prepended filename header from a passage body. The browser
// Cart Builder's PDF chunker prepends `<filename> (part N/M)` to every chunk's
// text. That duplicates the modal title (which already shows the same string)
// and breaks findOverlap because the header changes per chunk (part 403/2615 →
// part 404/2615) while the body content is what carries the ~50-word overlap
// window. Return value: text with the header removed if present, original
// text otherwise. Carts without this header pattern (raw text, markdown,
// non-PDF sources) are unaffected — they don't match the regex.
function stripChunkerHeader(text: string): string {
  // Match either:
  //   - PDF chunker:    `<filename> (part N/M)` followed by whitespace
  //   - Bracketed tag:  `[Poem: "..." from ...]`, `[Note: ...]`, etc.
  // The bracket case covers gutenberg-poetry and any other chunker that
  // prefixes each chunk with a `[label: ...]` marker. Carts with no header
  // pattern pass through unchanged.
  const m = text.match(/^(?:.+?\(part \d+\/\d+\)|\[[^\]\n]*\])\s*/)
  return m ? text.slice(m[0].length) : text
}

// Ensure verse numbers / line markers like `[5791]Some text` always have a
// space between the closing bracket and the following word. The chunker
// sometimes drops the line break that separated the marker from the verse
// in the source, producing `[5791]Some` runs that read as one token.
function spaceAfterBrackets(text: string): string {
  return text.replace(/\](\S)/g, '] $1')
}

export default function PassageModal() {
  const modalOpen = useAppStore((s) => s.modalOpen)
  const passage = useAppStore((s) => s.modalPassage)
  const loading = useAppStore((s) => s.modalLoading)
  const closeModal = useAppStore((s) => s.closeModal)
  const navigateModal = useAppStore((s) => s.navigateModal)
  const loadSource = useAppStore((s) => s.loadSourceForCurrentPassage)
  const mountedCart = useAppStore((s) => s.status?.mounted_cartridge)

  const hasPrev = passage?.prev_idx != null
  const hasNext = passage?.next_idx != null

  // Track the previously-displayed passage's text + the navigation direction
  // so the render pass can detect chunker overlap and gray it out. Reset on
  // modal close (so reopening from a search result is a clean view).
  const [prevTextForOverlap, setPrevTextForOverlap] = useState<string | null>(null)
  const [navDirection, setNavDirection] = useState<'next' | 'prev' | null>(null)

  useEffect(() => {
    if (!modalOpen) {
      setPrevTextForOverlap(null)
      setNavDirection(null)
    }
  }, [modalOpen])

  const handleNavNext = () => {
    if (!hasNext || loading || !passage) return
    setPrevTextForOverlap(passage.full_text)
    setNavDirection('next')
    navigateModal(passage.next_idx!)
  }

  const handleNavPrev = () => {
    if (!hasPrev || loading || !passage) return
    setPrevTextForOverlap(passage.full_text)
    setNavDirection('prev')
    navigateModal(passage.prev_idx!)
  }

  // Compute the [grayText, brightText] split based on detected overlap.
  // We work against the chunker-header-stripped body (not the raw full_text)
  // for two reasons:
  //   1. The header is already shown in the modal title bar above; rendering
  //      it inline duplicates information (red highlight in 2026-05-11 QA).
  //   2. The header differs per chunk (part 403/2615 → 404/2615), which
  //      breaks findOverlap. Stripping reveals the actual ~50-word overlap
  //      window at position 0 of the body (yellow highlight in QA).
  // Carts without a chunker header pattern pass through unchanged.
  const { grayText, brightText, overlapPosition } = useMemo(() => {
    const fullText = passage?.full_text ?? ''
    const body = spaceAfterBrackets(stripChunkerHeader(fullText))
    if (!prevTextForOverlap || !navDirection || !body) {
      return { grayText: '', brightText: body, overlapPosition: null as 'prefix' | 'suffix' | null }
    }
    const prevBody = spaceAfterBrackets(stripChunkerHeader(prevTextForOverlap))
    if (navDirection === 'next') {
      // prev's suffix should be current's prefix (the chunker's overlap window)
      const len = findOverlap(prevBody, body)
      if (len === 0) return { grayText: '', brightText: body, overlapPosition: null }
      return {
        grayText: body.slice(0, len),
        brightText: body.slice(len),
        overlapPosition: 'prefix' as const,
      }
    } else {
      // 'prev' direction: current's suffix should be prev's prefix
      const len = findOverlap(body, prevBody)
      if (len === 0) return { grayText: '', brightText: body, overlapPosition: null }
      return {
        grayText: body.slice(-len),
        brightText: body.slice(0, -len),
        overlapPosition: 'suffix' as const,
      }
    }
  }, [prevTextForOverlap, navDirection, passage?.full_text])

  // Close on Escape key + arrow-key navigation (with overlap-tracking handlers)
  useEffect(() => {
    if (!modalOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeModal()
      if (e.key === 'ArrowLeft') handleNavPrev()
      if (e.key === 'ArrowRight') handleNavNext()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modalOpen, hasPrev, hasNext, passage, closeModal, navigateModal])

  if (!modalOpen || !passage) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) closeModal() }}
    >
      <div className="relative w-full max-w-3xl max-h-[85vh] mx-4 flex flex-col rounded-2xl border border-slate-700/50 bg-slate-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700/40">
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-xs text-slate-500 font-mono shrink-0">#{passage.idx}</span>
            <h2 className="text-lg font-medium text-slate-200 truncate">{passage.title}</h2>
          </div>
          <button
            onClick={closeModal}
            className="shrink-0 p-2 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body — markdown rendered with remark-gfm (tables, autolinks, strikethrough).
              Trade-off vs the previous <pre>/renderTextWithLinks: per-query term
              highlighting is lost in the modal (it stays on the result cards, where
              it's more useful). The modal is the readable, formatted view; cards
              are the scan view. */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-500">
              <Loader2 size={20} className="animate-spin mr-2" />
              Loading passage...
            </div>
          ) : passage.full_text ? (
            <div className="text-sm text-slate-300 leading-relaxed space-y-3">
              {/* Overlap-graying (Andy 2026-05-09): when the user navigated
                  via Next from the previous chunk, the first ~50 words of
                  this chunk are usually the same as the last ~50 words of
                  the previous chunk (chunker's overlap window). Gray them
                  out + add a visual divider so the user sees clearly that
                  it's continuation, not redundancy. */}
              {overlapPosition === 'prefix' && grayText && (
                <>
                  <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500 font-mono">
                    <span className="flex-1 h-px bg-slate-700/40"></span>
                    continued from previous
                    <span className="flex-1 h-px bg-slate-700/40"></span>
                  </div>
                  <div className="text-slate-500 opacity-70 italic whitespace-pre-wrap text-xs leading-relaxed">
                    {grayText}
                  </div>
                  <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500 font-mono">
                    <span className="flex-1 h-px bg-slate-700/40"></span>
                    new content below
                    <span className="flex-1 h-px bg-slate-700/40"></span>
                  </div>
                </>
              )}
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkBreaks]}
                components={{
                  h1: ({ children }) => <h1 className="text-xl font-semibold text-slate-100 mt-4 mb-2">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-lg font-semibold text-slate-100 mt-3 mb-2">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-base font-semibold text-slate-200 mt-2 mb-1">{children}</h3>,
                  p: ({ children }) => <p className="my-2">{children}</p>,
                  ul: ({ children }) => <ul className="list-disc pl-6 my-2 space-y-1">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal pl-6 my-2 space-y-1">{children}</ol>,
                  li: ({ children }) => <li className="text-slate-300">{children}</li>,
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-cyan-400 hover:text-cyan-300 underline decoration-cyan-400/40 hover:decoration-cyan-300"
                    >
                      {children}
                    </a>
                  ),
                  code: ({ children }) => (
                    <code className="px-1 py-0.5 rounded bg-slate-800 text-amber-200 font-mono text-[12px]">{children}</code>
                  ),
                  pre: ({ children }) => (
                    <pre className="my-3 p-3 rounded-lg bg-slate-950/80 border border-slate-800 overflow-x-auto text-[12px] font-mono text-slate-200">{children}</pre>
                  ),
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-2 border-purple-500/40 pl-3 my-3 italic text-slate-400">{children}</blockquote>
                  ),
                  table: ({ children }) => (
                    <div className="my-3 overflow-x-auto">
                      <table className="border-collapse text-[13px]">{children}</table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="border border-slate-700 px-3 py-1 bg-slate-800/60 text-slate-200 text-left font-medium">{children}</th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-slate-700 px-3 py-1 text-slate-300">{children}</td>
                  ),
                  hr: () => <hr className="my-4 border-slate-700/50" />,
                  strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
                  em: ({ children }) => <em className="italic text-slate-200">{children}</em>,
                }}
              >
                {brightText}
              </ReactMarkdown>
              {/* Overlap-graying suffix variant — when user navigated via
                  Prev, gray out the overlap that was the *prefix* of the
                  chunk we just left. Symmetric treatment to the prefix case. */}
              {overlapPosition === 'suffix' && grayText && (
                <>
                  <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500 font-mono">
                    <span className="flex-1 h-px bg-slate-700/40"></span>
                    continues into next
                    <span className="flex-1 h-px bg-slate-700/40"></span>
                  </div>
                  <div className="text-slate-500 opacity-70 italic whitespace-pre-wrap text-xs leading-relaxed">
                    {grayText}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-500 italic">[No text available]</div>
          )}
        </div>

        {/* RAG+ provenance — three states:
              1. Split-cart, source not yet loaded:
                  show "Load full passage from <db> →" CTA
              2. Split-cart, source loaded (paper_id present):
                  show full source line with paper_id
              3. Standard cart (no source_db):
                  show cart-name + pattern idx (the simpler honest version)
            Mirrors the membot demo's modal UX. */}
        {!loading && passage.source_db && !passage.paper_id && (
          <div className="px-6 py-3 border-t border-slate-700/40 flex items-center justify-center">
            <button
              onClick={loadSource}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-500/20 border border-purple-500/40 text-purple-300 hover:bg-purple-500/30 hover:text-purple-200 text-sm font-medium transition-colors"
              title={`Fetch the full passage from ${passage.source_db}`}
            >
              <FolderOpen size={14} />
              Load full passage from {passage.source_db} →
            </button>
          </div>
        )}
        {!loading && passage.source_db && passage.paper_id && (
          <div className="px-6 py-2 border-t border-slate-700/40 text-[11px] text-slate-500 font-mono">
            source: {passage.source_db} · id: {passage.paper_id}
          </div>
        )}
        {!loading && !passage.source_db && mountedCart && (
          <div className="px-6 py-2 border-t border-slate-700/40 text-[11px] text-slate-500 font-mono">
            source: {mountedCart} · pattern #{passage.idx}
          </div>
        )}

        {/* Footer with navigation */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-700/40">
          <button
            onClick={handleNavPrev}
            disabled={!hasPrev || loading}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              hasPrev && !loading
                ? 'bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300'
                : 'bg-slate-800/50 text-slate-600 cursor-not-allowed'
            }`}
          >
            <ChevronLeft size={16} />
            Prev
          </button>

          <span className="text-xs text-slate-600">
            {hasPrev || hasNext ? 'Arrow keys to navigate' : 'No linked passages'}
          </span>

          <button
            onClick={handleNavNext}
            disabled={!hasNext || loading}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              hasNext && !loading
                ? 'bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300'
                : 'bg-slate-800/50 text-slate-600 cursor-not-allowed'
            }`}
          >
            Next
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
