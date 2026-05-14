import { useEffect, useState } from 'react'
import { Github, Mail, Sparkles, X, Zap } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

type Mode = 'signin' | 'signup' | 'magiclink'

// Sign-in modal: email/password + magic link + Google + GitHub. Web3 wallet
// providers (Ethereum, Solana) are enabled in the Supabase project but the
// custom button flow lands in Block B.1 (follow-up — requires window.ethereum
// / window.solana detection + signInWithWeb3 wiring).
export default function SignInModal() {
  const open = useAuthStore((s) => s.signInModalOpen)
  const close = useAuthStore((s) => s.closeSignInModal)
  const signInWithProvider = useAuthStore((s) => s.signInWithProvider)
  const signInWithEmail = useAuthStore((s) => s.signInWithEmail)
  const signUpWithEmail = useAuthStore((s) => s.signUpWithEmail)
  const signInWithMagicLink = useAuthStore((s) => s.signInWithMagicLink)

  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [info, setInfo] = useState<string | null>(null)  // post-action message (check your email, etc.)

  useEffect(() => {
    if (!open) return
    // Reset form whenever the modal opens.
    setMode('signin')
    setEmail('')
    setPassword('')
    setError(null)
    setInfo(null)

    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, close])

  if (!open) return null

  const submit = async () => {
    setError(null)
    setInfo(null)
    setBusy(true)
    try {
      if (mode === 'signin') {
        const { error: err } = await signInWithEmail(email, password)
        if (err) { setError(err); return }
        close()
      } else if (mode === 'signup') {
        const { error: err } = await signUpWithEmail(email, password)
        if (err) { setError(err); return }
        setInfo('Check your email to confirm your account.')
      } else {
        const { error: err } = await signInWithMagicLink(email)
        if (err) { setError(err); return }
        setInfo(`Magic link sent to ${email}. Check your inbox.`)
      }
    } finally {
      setBusy(false)
    }
  }

  const oauth = async (provider: 'google' | 'github') => {
    setError(null)
    setBusy(true)
    const { error: err } = await signInWithProvider(provider)
    setBusy(false)
    if (err) setError(err)
    // On success the browser is mid-redirect; modal cleanup will happen post-redirect.
  }

  const title =
    mode === 'signin'  ? 'Sign in to Vector+ Studio'
    : mode === 'signup' ? 'Create your account'
    : 'Sign in with magic link'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) close() }}
    >
      <div className="relative w-full max-w-md mx-4 rounded-2xl border border-slate-700/50 bg-slate-900 shadow-2xl">
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="gradient-bg w-7 h-7 rounded-md flex items-center justify-center">
              <Zap size={14} className="text-white" />
            </div>
            <h2 className="text-sm font-medium text-slate-200">{title}</h2>
          </div>
          <button
            onClick={close}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
          >
            <X size={14} />
          </button>
        </div>

        <div className="px-5 py-5 space-y-4">
          {/* OAuth providers — hidden on magic-link confirmation screen */}
          {!info && (
            <>
              <div className="space-y-2">
                <button
                  onClick={() => oauth('google')}
                  disabled={busy}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border border-slate-700/60 bg-slate-800/40 hover:bg-slate-800 text-slate-200 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  <GoogleIcon />
                  Continue with Google
                </button>
                <button
                  onClick={() => oauth('github')}
                  disabled={busy}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border border-slate-700/60 bg-slate-800/40 hover:bg-slate-800 text-slate-200 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  <Github size={16} />
                  Continue with GitHub
                </button>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-slate-700/50" />
                <span className="text-xs text-slate-500 uppercase tracking-wider">or</span>
                <div className="flex-1 h-px bg-slate-700/50" />
              </div>

              <div className="space-y-2">
                <input
                  type="email"
                  placeholder="email@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={busy}
                  className="w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/60 text-slate-200 placeholder:text-slate-500 text-sm focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
                />
                {mode !== 'magiclink' && (
                  <input
                    type="password"
                    placeholder="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={busy}
                    onKeyDown={(e) => { if (e.key === 'Enter') submit() }}
                    className="w-full px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/60 text-slate-200 placeholder:text-slate-500 text-sm focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
                  />
                )}
                <button
                  onClick={submit}
                  disabled={busy || !email || (mode !== 'magiclink' && !password)}
                  className="w-full px-4 py-2.5 rounded-lg gradient-bg text-white text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {busy
                    ? 'Working...'
                    : mode === 'signin' ? 'Sign in'
                    : mode === 'signup' ? 'Create account'
                    : 'Send magic link'}
                </button>
              </div>

              {error && (
                <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-300">
                  {error}
                </div>
              )}

              <div className="flex items-center justify-between text-xs">
                {mode === 'signin' ? (
                  <>
                    <button onClick={() => setMode('signup')} className="text-purple-400 hover:text-purple-300">
                      Need an account?
                    </button>
                    <button onClick={() => setMode('magiclink')} className="text-slate-500 hover:text-slate-300 flex items-center gap-1">
                      <Sparkles size={12} />
                      Magic link
                    </button>
                  </>
                ) : (
                  <button onClick={() => setMode('signin')} className="text-purple-400 hover:text-purple-300">
                    ← Back to sign in
                  </button>
                )}
              </div>
            </>
          )}

          {/* Post-submit confirmation (sign-up email or magic-link sent) */}
          {info && (
            <div className="text-center py-4 space-y-3">
              <Mail size={28} className="mx-auto text-purple-400" />
              <p className="text-sm text-slate-200">{info}</p>
              <button onClick={close} className="text-xs text-slate-500 hover:text-slate-300">
                Close
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Inline Google "G" SVG — official brand colors. Avoids pulling in a brand-asset package.
function GoogleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.7-6.1 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.1 8 3l5.7-5.7C34 6.1 29.3 4 24 4 13 4 4 13 4 24s9 20 20 20 20-9 20-20c0-1.3-.1-2.3-.4-3.5z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 16 19 13 24 13c3.1 0 5.8 1.1 8 3l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"/>
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2c-2 1.4-4.6 2.3-7.2 2.3-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z"/>
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.3 4.3-4.1 5.6l6.2 5.2c-.4.4 6.6-4.8 6.6-14.8 0-1.3-.1-2.3-.4-3.5z"/>
    </svg>
  )
}
