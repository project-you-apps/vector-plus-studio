import { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, X, Loader2, FolderOpen } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import { useAppStore } from '../store/appStore'

// Chunker overlap detection. The browser-side
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
function findOverlap(suffixOf: string, prefixOf: string, maxChars = 300): number {
  // reduced default cap from 600 → 300 chars. The
  // line-aware chunker landed 2026-07-05 preserves whole lines in overlap,
  // so a single long line at the boundary can drive the actual overlap
  // north of 500 chars. Capping the graying at 300 keeps the divider
  // legible without eating half the visible chunk. The larger structural
  // question (should the chunker's overlap window be smaller?) is filed
  // for post-demo revisit.
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

// Collapse pathological vertical whitespace from PDF-extracted text. PDFs
// often produce runs of 3+ consecutive newlines (with optional whitespace
// between) from layout-positional spacing — words/phrases scattered across
// vertical space because the source PDF wrapped lines mid-paragraph or used
// positional layout. Renders as dramatic per-word gaps in the modal (Andy
// 2026-06-26 QA on wiki_nomic_100k pattern #26 et al).
//
// This normalizes 3+ newlines into a single paragraph break (\n\n) without
// touching:
//   - Single-newline structure (intentional line breaks: poetry, verse, code)
//   - Single-blank-line paragraph breaks (legitimate \n\n stays)
//
// Pure UX/display fix; zero changes to cart-format or chunker. PDF-derived
// chunks become readable; poetry/verse/code rendering unaffected.
function collapseExcessWhitespace(text: string): string {
  return text.replace(/(\n\s*){3,}/g, '\n\n')
}

export default function PassageModal() {
  const modalOpen = useAppStore((s) => s.modalOpen)
  const passage = useAppStore((s) => s.modalPassage)
  const loading = useAppStore((s) => s.modalLoading)
  const closeModal = useAppStore((s) => s.closeModal)
  const navigateModal = useAppStore((s) => s.navigateModal)
  const loadSource = useAppStore((s) => s.loadSourceForCurrentPassage)
  const mountedCart = useAppStore((s) => s.status?.mounted_cartridge)
  // surface the query term in the modal so users don't
  // lose the "why am I looking at this?" thread when navigating Prev/Next.
  const currentQuery = useAppStore((s) => s.query)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const localCarts = useAppStore((s) => s.localCarts)
  const sandboxPerPatternMeta = useAppStore((s) => s.sandboxPerPatternMeta)

  // Resolve the per-pattern-meta array to use: prefer the active LocalCart's
  // in-memory copy, otherwise fall back to the sandbox-mounted cart's
  // server-fetched mirror. parity fix so droplet-uploaded
  // carts show images in the modal + inline-image replacement + drill-down.
  const perPatternMeta = useMemo(() => {
    if (activeLocalCart) {
      return localCarts.get(activeLocalCart)?.perPatternMeta ?? null
    }
    return sandboxPerPatternMeta
  }, [activeLocalCart, localCarts, sandboxPerPatternMeta])

  // Day 2 graphic — if the current passage is a graphic-type pattern (Image
  // Builder extracted), pull the base64 PNG out of per_pattern_meta and hand
  // back a data URL for inline rendering above the caption.
  // PM: this is the wow moment for the pitch deck demo — click a graphic
  // result, see the actual graphic.
  const graphicPatternMeta = useMemo(() => {
    if (!passage || !perPatternMeta) return null
    const rec = perPatternMeta[passage.idx]
    if (!rec || rec.content_type !== 'graphic' || !rec.image_b64) return null
    return rec
  }, [passage, perPatternMeta])
  const graphicDataUrl: string | null = graphicPatternMeta
    ? `data:image/png;base64,${graphicPatternMeta.image_b64}`
    : null

  // click-to-zoom lightbox for graphic images. Any
  // inline `<img>` in the passage viewer can be clicked to open a
  // full-viewport view. Escape or backdrop click closes.
  const [zoomedSrc, setZoomedSrc] = useState<{ src: string; alt: string } | null>(null)
  useEffect(() => {
    if (!zoomedSrc) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        setZoomedSrc(null)
      }
    }
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [zoomedSrc])

  // inline-image replacement. Docling leaves
  // `<!-- image -->` markers wherever graphics live in the reading flow.
  // If the current passage's source has extracted graphics baked into
  // perPatternMeta, replace each marker with a markdown image referencing a
  // short placeholder like `graphic:0`. The ReactMarkdown img override
  // below resolves that placeholder to the real data URL. Effect: images
  // render in the same order they appeared on the page.
  const sourceGraphics = useMemo(() => {
    if (!passage || !perPatternMeta) return []
    // Prefer sourcePaths from LocalCart if available; otherwise derive the
    // source for this passage from the per-pattern-meta record itself
    // (sandbox mode has no in-memory sourcePaths sidecar).
    let src: string | undefined
    if (activeLocalCart) {
      const cart = localCarts.get(activeLocalCart)
      src = cart?.sourcePaths?.[passage.idx] ?? undefined
    }
    if (!src) {
      src = perPatternMeta[passage.idx]?.source ?? undefined
    }
    if (!src) return []
    return perPatternMeta.filter(
      (r) => r.content_type === 'graphic' && r.source === src && r.image_b64,
    )
  }, [passage, perPatternMeta, activeLocalCart, localCarts])

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
    const body = collapseExcessWhitespace(spaceAfterBrackets(stripChunkerHeader(fullText)))
    if (!prevTextForOverlap || !navDirection || !body) {
      return { grayText: '', brightText: body, overlapPosition: null as 'prefix' | 'suffix' | null }
    }
    const prevBody = collapseExcessWhitespace(spaceAfterBrackets(stripChunkerHeader(prevTextForOverlap)))
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

  // Replace Docling's inline `<!-- image -->` HTML-comment markers with
  // markdown image references pointing to short placeholders (`graphic:N`).
  // The ReactMarkdown `img` component override resolves each placeholder to
  // the corresponding graphic's base64 data URL. Ordering: use the source
  // file's graphic sequence, indexed globally within this passage — so the
  // Nth `<!-- image -->` on the page maps to the Nth graphic from that file.
  // Won't always be pixel-perfect (chunks span pages), but the images are
  // guaranteed to be from the same source and appear in reading order.
  const brightTextWithGraphics = useMemo(() => {
    if (!brightText || sourceGraphics.length === 0) return brightText
    let i = 0
    return brightText.replace(/<!--\s*image\s*-->/g, () => {
      if (i >= sourceGraphics.length) return '' // no more graphics; drop the marker rather than show a broken image
      const marker = `![Graphic ${i + 1}](graphic:${i})`
      i += 1
      return marker
    })
  }, [brightText, sourceGraphics])

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
        <div className="flex flex-col px-6 py-4 border-b border-slate-700/40 gap-1.5">
          <div className="flex items-center justify-between gap-3 min-w-0">
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
          {/* Query-pill — surfaces the query that led here.
              AM: helps keep the "why am I looking at this" thread across
              Prev/Next navigation. Truncated to 47 chars + ellipsis. */}
          {currentQuery && currentQuery.trim().length > 0 && (
            <div className="flex items-center gap-1.5 text-[11px] pl-8">
              <span className="text-slate-500 font-mono uppercase tracking-wider">Query</span>
              <span
                className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-mono truncate max-w-full"
                title={currentQuery}
              >
                {currentQuery.length > 47 ? `${currentQuery.slice(0, 47)}…` : currentQuery}
              </span>
            </div>
          )}
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
              {graphicDataUrl && (
                <div className="mb-4 flex justify-center">
                  <img
                    src={graphicDataUrl}
                    alt={graphicPatternMeta?.caption || 'Extracted graphic'}
                    onClick={() =>
                      setZoomedSrc({
                        src: graphicDataUrl,
                        alt: graphicPatternMeta?.caption || 'Extracted graphic',
                      })
                    }
                    className="max-h-[50vh] max-w-full object-contain bg-slate-950 rounded-lg border border-slate-700/40 cursor-zoom-in hover:brightness-110 transition-all"
                  />
                </div>
              )}
              {/* Overlap-graying: when the user navigated
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
                // pass-through URL transform so our
                // `graphic:N` placeholder scheme survives ReactMarkdown's
                // default sanitizer (which strips non-http/https/data schemes
                // and rewrites src to empty). The img component override below
                // resolves the placeholder to a real data URL.
                urlTransform={(uri) => uri}
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
                  // inline-image resolver. Preprocessed
                  // markdown carries `graphic:N` placeholders where Docling's
                  // `<!-- image -->` markers used to be. Look up the real
                  // base64 payload from sourceGraphics and render inline.
                  // Non-graphic:N srcs (regular markdown images) pass through
                  // as normal <img>.
                  img: ({ src, alt }) => {
                    const srcStr = typeof src === 'string' ? src : ''
                    if (srcStr.startsWith('graphic:')) {
                      const idx = parseInt(srcStr.slice('graphic:'.length), 10)
                      const g = sourceGraphics[idx]
                      if (!g?.image_b64) return null
                      const dataUrl = `data:image/png;base64,${g.image_b64}`
                      const altText = g.caption || alt || 'Extracted graphic'
                      return (
                        <span className="block my-3 flex flex-col items-center gap-1">
                          <img
                            src={dataUrl}
                            alt={altText}
                            onClick={() => setZoomedSrc({ src: dataUrl, alt: altText })}
                            className="max-h-96 max-w-full object-contain bg-slate-950 rounded-lg border border-slate-700/40 cursor-zoom-in hover:brightness-110 transition-all"
                          />
                          {g.caption && (
                            <span className="text-[11px] text-slate-500 italic text-center">
                              {g.caption}
                            </span>
                          )}
                        </span>
                      )
                    }
                    return <img src={srcStr} alt={alt ?? ''} className="max-w-full rounded" />
                  },
                }}
              >
                {brightTextWithGraphics}
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

        {/* Footer with navigation.
            UX: at first/last passage of a drilled
            file the disabled button was previously silent — the browser's
            not-allowed cursor read as a STOP glyph with no explanation.
            Now the disabled side shows an explicit "No previous" / "No next"
            label so the boundary state is legible without hovering. */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-700/40">
          <button
            onClick={handleNavPrev}
            disabled={!hasPrev || loading}
            title={!hasPrev ? 'No linked previous passage' : undefined}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              hasPrev && !loading
                ? 'bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300'
                : 'bg-slate-800/50 text-slate-600 cursor-not-allowed'
            }`}
          >
            <ChevronLeft size={16} />
            {hasPrev ? 'Prev' : 'No previous'}
          </button>

          <span className="text-xs text-slate-600">
            {hasPrev || hasNext ? 'Arrow keys to navigate' : 'No linked passages'}
          </span>

          <button
            onClick={handleNavNext}
            disabled={!hasNext || loading}
            title={!hasNext ? 'No linked next passage' : undefined}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              hasNext && !loading
                ? 'bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300'
                : 'bg-slate-800/50 text-slate-600 cursor-not-allowed'
            }`}
          >
            {hasNext ? 'Next' : 'No next'}
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
      {/* Click-to-zoom lightbox — full-viewport overlay above the passage
          modal. Escape or backdrop click closes. the
          modal-from-modal we joked about earlier. */}
      {zoomedSrc && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-md cursor-zoom-out"
          onClick={() => setZoomedSrc(null)}
        >
          <img
            src={zoomedSrc.src}
            alt={zoomedSrc.alt}
            onClick={(e) => e.stopPropagation()}
            className="max-h-[92vh] max-w-[92vw] object-contain rounded-xl border border-slate-700/40 shadow-2xl cursor-default"
          />
          <button
            onClick={() => setZoomedSrc(null)}
            className="absolute top-4 right-4 p-2 rounded-full bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-slate-100 transition-colors"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>
      )}
    </div>
  )
}
