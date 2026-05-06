import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import type { Toast } from '../store/cartBuilderStore'

// Toaster — fixed bottom-right stack of transient notifications.
// Triggered from cartBuilderStore.pushToast(). Cart Builder owns the slice
// today; other screens can borrow by importing the same store.

export default function Toaster() {
  const toasts = useCartBuilderStore((s) => s.toasts)
  const dismiss = useCartBuilderStore((s) => s.dismissToast)

  if (toasts.length === 0) return null

  return (
    <div className="fixed bottom-20 right-4 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const palette = {
    success: { icon: CheckCircle2, ring: 'border-green-500/40 bg-green-500/10', text: 'text-green-200' },
    error:   { icon: AlertCircle,  ring: 'border-rose-500/40 bg-rose-500/10',   text: 'text-rose-200'  },
    info:    { icon: Info,         ring: 'border-cyan-500/40 bg-cyan-500/10',   text: 'text-cyan-200'  },
  }[toast.kind]
  const Icon = palette.icon
  return (
    <div
      className={`pointer-events-auto rounded-lg border px-4 py-2.5 backdrop-blur-md shadow-xl flex items-start gap-3 ${palette.ring} animate-fadein`}
      role="status"
    >
      <Icon size={16} className={`${palette.text} mt-0.5 shrink-0`} />
      <span className={`flex-1 text-sm leading-snug ${palette.text}`}>{toast.text}</span>
      <button
        onClick={onDismiss}
        className="text-slate-500 hover:text-slate-300 mt-0.5"
        title="Dismiss"
      >
        <X size={13} />
      </button>
    </div>
  )
}
