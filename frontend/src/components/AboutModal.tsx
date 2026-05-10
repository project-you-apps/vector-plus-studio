import { useEffect } from 'react'
import { X, Zap, ExternalLink } from 'lucide-react'

interface Props {
  open: boolean
  onClose: () => void
}

export default function AboutModal({ open, onClose }: Props) {
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
            <div className="gradient-bg w-7 h-7 rounded-md flex items-center justify-center">
              <Zap size={14} className="text-white" />
            </div>
            <h2 className="text-sm font-medium text-slate-200">About Vector+ Studio</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
          >
            <X size={14} />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 text-sm text-slate-300">
          <div>
            <div className="text-slate-100 font-semibold">Vector+ Studio</div>
            <div className="text-xs text-slate-500">v1.1 — Hosted Demo + Browser-Side Cart Builder</div>
          </div>

          <p className="text-slate-400 leading-relaxed">
            Physics-enhanced semantic search. Queries settle through a 16-million-neuron
            Hopfield lattice — the substrate finds <em>related</em> patterns, not just
            matching ones.
          </p>

          <p className="text-slate-400 leading-relaxed">
            Build cartridges in your browser (your files never leave your machine), or
            mount one of the bundled samples. The hosted demo runs the same CUDA physics
            engine as the desktop app.
          </p>

          <div className="pt-2 border-t border-slate-700/40 space-y-2">
            <a
              href="https://wavingcat.dev"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300"
            >
              <ExternalLink size={14} />
              wavingcat.dev
            </a>
            <a
              href="https://project-you.app"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300"
            >
              <ExternalLink size={14} />
              project-you.app
            </a>
            <a
              href="https://github.com/project-you-apps"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300"
            >
              <ExternalLink size={14} />
              GitHub
            </a>
          </div>

          <div className="pt-2 border-t border-slate-700/40 text-xs text-slate-500">
            Built with physics, not just math. Patterns stored holographically, not as records.
          </div>
        </div>
      </div>
    </div>
  )
}
