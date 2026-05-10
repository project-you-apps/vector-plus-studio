import { useEffect } from 'react'
import { X, HelpCircle, BookOpen, Rocket, Construction } from 'lucide-react'

interface Props {
  open: boolean
  onClose: () => void
}

export default function HelpModal({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="relative w-full max-w-lg mx-4 rounded-2xl border border-slate-700/50 bg-slate-900 shadow-2xl">
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <HelpCircle size={16} className="text-cyan-400" />
            <h2 className="text-sm font-medium text-slate-200">Help</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
          >
            <X size={14} />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 text-sm text-slate-300">
          <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-4">
            <div className="flex items-center gap-2 mb-2">
              <Rocket size={14} className="text-amber-400" />
              <span className="font-semibold text-slate-100">Getting Started</span>
              <span className="ml-auto flex items-center gap-1 text-xs text-amber-400/80">
                <Construction size={12} />
                Coming soon
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              A walk-through of mounting a cartridge, running your first search, and
              building your own cart from local files. Placeholder for now — full
              guide ships shortly.
            </p>
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-4">
            <div className="flex items-center gap-2 mb-2">
              <BookOpen size={14} className="text-purple-400" />
              <span className="font-semibold text-slate-100">Docs</span>
              <span className="ml-auto flex items-center gap-1 text-xs text-amber-400/80">
                <Construction size={12} />
                Coming soon
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Architecture, API reference, cartridge format, search modes, and the
              browser-side build pipeline. Placeholder for now — docs site lands on
              wavingcat.dev shortly.
            </p>
          </div>

          <div className="pt-2 border-t border-slate-700/40 text-xs text-slate-500">
            Got a question that isn't answered yet?{' '}
            <a
              href="mailto:andy@project-you.app"
              className="text-cyan-400 hover:text-cyan-300"
            >
              andy@project-you.app
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
