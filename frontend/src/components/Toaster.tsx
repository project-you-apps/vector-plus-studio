import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import type { Toast } from '../store/cartBuilderStore'

// Toaster — TOP-CENTRE stack of transient notifications, where a JS alert would appear.
// Triggered from cartBuilderStore.pushToast(). Cart Builder owns the slice
// today; other screens can borrow by importing the same store.
//
// MOVED FROM BOTTOM-RIGHT, 2026-08-13. Andy: "it pops up almost out of the corner of one's
// eye down at the bottom right." A refusal nobody sees is the same as no refusal, which is
// the failure this whole week has been about — and it matters more now that a write can be
// refused because someone ELSE holds the cart.
//
// TOP-centre, not dead centre, and that is deliberate: the thing being read is in the middle
// of the screen, so a centred toast lands on top of it — worse as they stack, and worse
// again for the transient ones nobody asked to dismiss. Top-centre sits in the reading path
// without covering the text. It is also Andy's own earlier call, 2026-02-28, for the
// draggable panels: "middle top where a javascript Alert would pop up."
//
// `pointer-events-none` on the stack with `pointer-events-auto` on each toast stays: the
// gaps between toasts must not eat clicks aimed at the content underneath.

export default function Toaster() {
  const toasts = useCartBuilderStore((s) => s.toasts)
  const dismiss = useCartBuilderStore((s) => s.dismissToast)

  if (toasts.length === 0) return null

  return (
    <div className="fixed top-20 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center gap-2 w-full max-w-md px-4 pointer-events-none">
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
