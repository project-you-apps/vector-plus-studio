import { useEffect, useState } from 'react'
import { Cpu, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'

// Pair the web app with the local Desktop Cart Builder exe. The exe prints
// a URL-safe pairing code to its console on every startup; the user copies
// it and pastes here. Success stores the token in localStorage + flips the
// BackendBadge to the purple DESKTOP HELPER state. Same visual language as
// SignInModal / BriefingModal — store-driven open state, Escape + backdrop
// close, purple accent header.
export default function DesktopHelperPairModal() {
  const open = useAppStore((s) => s.desktopHelperPairModalOpen)
  const close = useAppStore((s) => s.closeDesktopHelperPairModal)
  const pair = useAppStore((s) => s.pairDesktopHelper)
  const storeError = useAppStore((s) => s.desktopHelperError)
  const caps = useAppStore((s) => s.desktopHelperCapabilities)

  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    // Reset form each time we open — stale error/token from a previous session
    // shouldn't linger past a close.
    setToken('')
    setBusy(false)
    setLocalError(null)

    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, close])

  if (!open) return null

  const submit = async () => {
    setLocalError(null)
    setBusy(true)
    try {
      const resp = await pair(token)
      if (!resp.success) setLocalError(resp.message)
      // Success closes the modal from within the store action.
    } finally {
      setBusy(false)
    }
  }

  const displayError = localError ?? storeError
  const version = caps?.version ? ` v${caps.version}` : ''

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Connect to Desktop Cart Builder"
      onClick={(e) => { if (e.target === e.currentTarget) close() }}
    >
      <div
        className="relative w-full max-w-md rounded-2xl border border-purple-500/40 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md flex items-center justify-center bg-purple-500/20 border border-purple-500/40">
              <Cpu size={14} className="text-purple-300" />
            </div>
            <h2 className="text-sm font-medium text-slate-200">
              Connect to Desktop Cart Builder
            </h2>
          </div>
          <button
            onClick={close}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
            aria-label="Close"
            title="Close (Esc)"
          >
            <X size={14} />
          </button>
        </div>

        <div className="px-5 py-5 space-y-4">
          <div className="text-xs text-slate-400 leading-relaxed space-y-2">
            <p>
              Detected the Vector+ Desktop Cart Builder{version} on your machine.
              Pairing lets this browser tab delegate <em>Build</em> to the local
              exe so files never leave your computer and embedding runs on your
              own GPU.
            </p>
            <p className="text-slate-500">
              Copy the pairing code from the Desktop Builder console window and
              paste it below.
            </p>
          </div>

          <div className="space-y-2">
            <label htmlFor="dh-token" className="block text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
              Pairing code
            </label>
            <input
              id="dh-token"
              type="text"
              autoFocus
              autoComplete="off"
              spellCheck={false}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') submit() }}
              disabled={busy}
              placeholder="abc123-def456-ghi789"
              className="w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/60 text-slate-200 placeholder:text-slate-600 text-sm font-mono focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
            />
          </div>

          {displayError && (
            <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-300">
              {displayError}
            </div>
          )}

          <div className="flex items-center justify-end gap-2">
            <button
              onClick={close}
              disabled={busy}
              className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={busy || !token.trim()}
              className="px-4 py-1.5 rounded-lg bg-purple-500/20 border border-purple-500/50 text-purple-200 hover:bg-purple-500/30 text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {busy ? 'Pairing...' : 'Pair'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
