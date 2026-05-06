import { useEffect } from 'react'
import { AlertTriangle, X } from 'lucide-react'

// ConfirmDialog — reusable destructive-action confirmation modal.
// Andy 2026-05-06: standard practice for destructive actions in VPS.
//
// Usage:
//   const [confirmState, setConfirmState] = useState<ConfirmState>(null)
//   ...
//   <ConfirmDialog
//     state={confirmState}
//     onCancel={() => setConfirmState(null)}
//     onConfirm={() => { confirmState?.onConfirm(); setConfirmState(null) }}
//   />
//
// Or use the helper hook below for most cases.

export interface ConfirmState {
  title: string
  body: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  destructive?: boolean   // toggles red vs neutral confirm-button styling
  onConfirm: () => void | Promise<void>
}

interface Props {
  state: ConfirmState | null
  onCancel: () => void
  onConfirm: () => void
}

export default function ConfirmDialog({ state, onCancel, onConfirm }: Props) {
  // Esc to cancel, Enter to confirm
  useEffect(() => {
    if (!state) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
      if (e.key === 'Enter') onConfirm()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [state, onCancel, onConfirm])

  if (!state) return null

  const isDestructive = state.destructive !== false  // default true — most uses are destructive

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="relative w-full max-w-md mx-4 rounded-2xl border border-slate-700/50 bg-slate-900 shadow-2xl">
        {/* Header */}
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <AlertTriangle
              size={16}
              className={isDestructive ? 'text-rose-400 shrink-0' : 'text-amber-400 shrink-0'}
            />
            <h2 className="text-sm font-medium text-slate-200">{state.title}</h2>
          </div>
          <button
            onClick={onCancel}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
            title="Cancel (Esc)"
          >
            <X size={14} />
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 text-sm text-slate-300 leading-relaxed">
          {state.body}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-700/40 flex items-center justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
          >
            {state.cancelLabel ?? 'Cancel'}
          </button>
          <button
            onClick={onConfirm}
            autoFocus
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              isDestructive
                ? 'bg-rose-500/20 border border-rose-500/40 text-rose-300 hover:bg-rose-500/30'
                : 'bg-purple-500/20 border border-purple-500/40 text-purple-300 hover:bg-purple-500/30'
            }`}
          >
            {state.confirmLabel ?? (isDestructive ? 'Delete' : 'Confirm')}
          </button>
        </div>

        <div className="px-5 pb-2 text-[10px] text-slate-600 italic text-right">
          Enter to confirm · Esc to cancel
        </div>
      </div>
    </div>
  )
}
