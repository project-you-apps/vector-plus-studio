import { useEffect } from 'react'
import { ChevronLeft, ChevronRight, X, Loader2, FolderOpen } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAppStore } from '../store/appStore'

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

  // Close on Escape key
  useEffect(() => {
    if (!modalOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeModal()
      if (e.key === 'ArrowLeft' && hasPrev) navigateModal(passage!.prev_idx!)
      if (e.key === 'ArrowRight' && hasNext) navigateModal(passage!.next_idx!)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
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
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
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
                {passage.full_text}
              </ReactMarkdown>
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
            onClick={() => hasPrev && navigateModal(passage.prev_idx!)}
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
            onClick={() => hasNext && navigateModal(passage.next_idx!)}
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
