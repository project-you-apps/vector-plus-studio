import { useEffect, useRef, useState } from 'react'
import { LogIn, LogOut, User as UserIcon } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

// Header element: "Sign In" button when signed out, user chip + dropdown when
// signed in. Render this inside Header.tsx anywhere on the right cluster.
export default function AuthChip() {
  const user = useAuthStore((s) => s.user)
  const initialized = useAuthStore((s) => s.initialized)
  const openSignIn = useAuthStore((s) => s.openSignInModal)
  const signOut = useAuthStore((s) => s.signOut)
  const [menuOpen, setMenuOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  // Close dropdown on outside click.
  useEffect(() => {
    if (!menuOpen) return
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    window.addEventListener('mousedown', handler)
    return () => window.removeEventListener('mousedown', handler)
  }, [menuOpen])

  // Render nothing until we know the auth state (avoids flash of "Sign In" for already-signed-in users).
  if (!initialized) {
    return <div className="w-20 h-8" aria-hidden="true" />
  }

  if (!user) {
    return (
      <button
        onClick={openSignIn}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 transition-colors text-xs font-medium"
        title="Sign in"
      >
        <LogIn size={14} />
        Sign in
      </button>
    )
  }

  // Avatar from user_metadata if OAuth provided one; otherwise initials.
  const avatarUrl = (user.user_metadata?.avatar_url as string | undefined) ?? null
  const label = user.email ?? user.user_metadata?.full_name ?? 'Account'
  const initial = (user.email?.[0] ?? '?').toUpperCase()

  return (
    <div ref={wrapRef} className="relative">
      <button
        onClick={() => setMenuOpen((v) => !v)}
        className="flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-slate-800/60 transition-colors"
        title={label}
      >
        {avatarUrl ? (
          <img src={avatarUrl} alt="" className="w-7 h-7 rounded-full" />
        ) : (
          <div className="w-7 h-7 rounded-full gradient-bg flex items-center justify-center text-xs font-semibold text-white">
            {initial}
          </div>
        )}
      </button>

      {menuOpen && (
        <div className="absolute right-0 top-full mt-2 w-56 rounded-lg border border-slate-700/50 bg-slate-900 shadow-xl py-1 z-40">
          <div className="px-3 py-2 border-b border-slate-700/40">
            <div className="text-xs text-slate-500">Signed in as</div>
            <div className="text-sm text-slate-200 truncate">{label}</div>
          </div>
          <button
            onClick={() => { setMenuOpen(false); /* TODO: profile screen */ }}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
          >
            <UserIcon size={14} />
            Profile
            <span className="ml-auto text-[10px] text-slate-600">soon</span>
          </button>
          <button
            onClick={async () => { setMenuOpen(false); await signOut() }}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:bg-slate-800/60 hover:text-red-300"
          >
            <LogOut size={14} />
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}
