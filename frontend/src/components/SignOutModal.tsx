import { X, Check, Loader2, AlertCircle } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

/**
 * What happened when you signed out, while it is happening.
 *
 * ⭐ ANDY'S IDEA, 2026-11-06, with one correction to its premise. He asked for a modal
 * reading *"Signing Out. Do not close until the cartridge is dismounted."* to mask the
 * delay between sign-out and unmount.
 *
 * ⛔ THERE IS NO SUCH DELAY TO MASK. `unmountLocalCart()` is a synchronous store update;
 * the cart is gone before the modal can paint. A modal waiting on something already
 * finished is theater, and a user who catches one stops believing every other progress
 * indicator in the product. ⇒ So each line here flips when that step ACTUALLY completes.
 *
 * ⭐ The delay that IS real is `supabase.auth.signOut()` — a network round trip, and the
 * window in which the UI sits half-signed-out (it is when the /api/cartridges 401 race
 * fires). That is what this covers, and covering it is worth doing.
 *
 * ⚠ The shape anticipates the server-side unmount (Andy's 2026-08-06 occupancy design).
 * That one WILL be slow, because it drops a multi-GB mount — and when it lands, this modal
 * earns Andy's original wording honestly rather than by assertion.
 *
 * ⛔ CLOSES ON X ONLY. Andy, 2026-02-26: *"the user modals should only close when
 * dismissed by an X or CLOSE button."* No timer, no click-away: a sign-out confirmation
 * that vanishes before it is read has told nobody anything.
 */
export default function SignOutModal() {
  const state = useAuthStore((s) => s.signOutState)
  const dismiss = useAuthStore((s) => s.dismissSignOut)
  if (!state) return null

  const done = state.sessionEnded && !state.error

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-sm rounded-lg border border-slate-700 bg-slate-900 p-5 shadow-2xl">
        <div className="mb-4 flex items-start justify-between gap-3">
          <h2 className="text-base font-semibold text-slate-100">
            {state.error ? 'Sign-out failed' : done ? 'Signed out' : 'Signing out…'}
          </h2>
          {/* ⚠ Enabled the whole time, including mid-flight. Trapping someone in a modal
              because WE are still working is a worse failure than a half-finished
              sign-out they can see the state of. */}
          <button
            type="button"
            onClick={dismiss}
            aria-label="Close"
            className="rounded p-1 text-slate-500 hover:bg-slate-800 hover:text-slate-200"
          >
            <X size={16} />
          </button>
        </div>

        <ul className="space-y-2 text-sm">
          <Step
            done={state.cartCleared}
            label={state.hadCart
              ? 'Cartridge unmounted'
              : 'No cartridge was open'}
          />
          <Step done={state.sessionEnded} failed={!!state.error} label="Session ended" />
        </ul>

        {state.error && (
          <div className="mt-4 flex items-start gap-2 rounded border border-rose-800/60 bg-rose-950/40 p-2 text-xs text-rose-200">
            <AlertCircle size={13} className="mt-0.5 shrink-0" />
            <span>
              {state.error}
              {/* ⛔ The consequence, not just the message. "Sign-out failed" means nothing
                  to someone about to walk away from the machine. */}
              <strong className="mt-1 block font-semibold">
                You may still be signed in on this device. Close the browser before leaving it.
              </strong>
            </span>
          </div>
        )}

        {done && (
          <p className="mt-4 text-xs leading-relaxed text-slate-500">
            ⓘ A cartridge opened from your own computer was only ever in this browser, so
            nothing had to be sent anywhere to close it.
          </p>
        )}

        <button
          type="button"
          onClick={dismiss}
          className="mt-4 w-full rounded bg-slate-800 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700"
        >
          Close
        </button>
      </div>
    </div>
  )
}

function Step({ done, failed, label }: { done: boolean; failed?: boolean; label: string }) {
  return (
    <li className="flex items-center gap-2">
      {failed ? (
        <AlertCircle size={14} className="shrink-0 text-rose-400" />
      ) : done ? (
        <Check size={14} className="shrink-0 text-emerald-400" />
      ) : (
        <Loader2 size={14} className="shrink-0 animate-spin text-slate-500" />
      )}
      <span className={done && !failed ? 'text-slate-300' : 'text-slate-400'}>{label}</span>
    </li>
  )
}
