import { useEffect, useState } from 'react'
import { User as UserIcon, Shield, Eye, MessageSquare, Pencil, Crown, RefreshCw, AlertTriangle } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../store/authStore'

/**
 * The User Page: identity, seat, and every cart this person can reach.
 *
 * TALKS TO SUPABASE DIRECTLY, NOT THROUGH OUR API — deliberately. Row-level security
 * (db/004_cart_grants.sql) means the database itself refuses to hand this browser another
 * user's row. If our API were the gatekeeper instead, every new endpoint would be a fresh
 * chance to forget a `WHERE user_id =`. Two fail-open permission paths were found in the
 * sidecar reader on 2026-08-01; the floor belongs somewhere we cannot typo.
 *
 * Consequence worth stating: this component has NO privileged path. Everything it can see,
 * the signed-in user is entitled to see.
 *
 * SEAT = auth user UUID. Not the email, not the handle — both change and are re-assignable,
 * and either would silently re-point a person's whole attention history at someone else.
 * The UUID is opaque, so the human label is resolved for DISPLAY only.
 */

type CartRow = {
  id: string
  cart_filename: string
  display_name: string | null
  size_bytes: number | null
  pattern_count: number | null
  user_id: string
}

type Profile = {
  id: string
  email?: string | null
  username?: string | null
  full_name?: string | null
  display_name?: string | null
  avatar_url?: string | null
  apps_list?: string[] | null
  created_at?: string | null
}

type AccessLevel = 'owner' | 'editor' | 'commenter' | 'viewer' | null

// Mirrors api/profiles.py. Kept deliberately small and total: any value that is not one of
// these is NO ACCESS, because absence of a grant is a denial rather than a default.
const LEVEL_META: Record<Exclude<AccessLevel, null>, {
  label: string; icon: typeof Eye; className: string; can: string[]
}> = {
  owner:     { label: 'Owner',     icon: Crown,         className: 'bg-amber-500/15 text-amber-300 border-amber-500/30',   can: ['read', 'annotate', 'write', 'share', 'sign'] },
  editor:    { label: 'Editor',    icon: Pencil,        className: 'bg-purple-500/15 text-purple-300 border-purple-500/30', can: ['read', 'annotate', 'write'] },
  commenter: { label: 'Commenter', icon: MessageSquare, className: 'bg-sky-500/15 text-sky-300 border-sky-500/30',          can: ['read', 'annotate'] },
  viewer:    { label: 'Viewer',    icon: Eye,           className: 'bg-slate-500/15 text-slate-300 border-slate-600/40',    can: ['read'] },
}

function displayNameFor(profile: Profile | null, email?: string | null): string {
  const candidates = [profile?.display_name, profile?.full_name, profile?.username]
  for (const c of candidates) if (c && c.trim()) return c.trim()
  const mail = profile?.email ?? email ?? ''
  if (mail.includes('@')) return mail.split('@')[0]
  return 'Unknown'
}

function formatBytes(n: number | null): string {
  if (!n || n <= 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n, i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

function AccessBadge({ level }: { level: AccessLevel }) {
  if (!level) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs bg-red-500/10 text-red-300 border-red-500/30">
        No access
      </span>
    )
  }
  const meta = LEVEL_META[level]
  const Icon = meta.icon
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs ${meta.className}`}
      title={`Can: ${meta.can.join(', ')}`}
    >
      <Icon size={12} />
      {meta.label}
    </span>
  )
}

export default function ProfileScreen() {
  const { user, session } = useAuthStore()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [carts, setCarts] = useState<Array<CartRow & { access: AccessLevel }>>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  async function load() {
    if (!user) return
    setLoading(true)
    setError(null)
    try {
      // Profile row. Absent is NORMAL for a brand-new user — not an error state.
      const { data: profileRows, error: profileErr } = await supabase
        .from('profiles').select('*').eq('id', user.id).limit(1)
      if (profileErr) throw profileErr
      setProfile(profileRows?.[0] ?? null)

      // RLS returns owned carts (existing policy) AND granted carts (added in 004), so
      // this is one query rather than a union we would have to keep correct by hand.
      const { data: cartRows, error: cartErr } = await supabase
        .from('user_carts')
        .select('id, cart_filename, display_name, size_bytes, pattern_count, user_id')
      if (cartErr) throw cartErr

      const { data: grantRows, error: grantErr } = await supabase
        .from('cart_grants').select('cart_id, access_level').eq('grantee_id', user.id)
      if (grantErr) throw grantErr

      const grants = new Map<string, AccessLevel>(
        (grantRows ?? []).map((g: any) => [g.cart_id, g.access_level as AccessLevel]),
      )

      const resolved = (cartRows ?? []).map((row: CartRow) => ({
        ...row,
        access: (row.user_id === user.id ? 'owner' : grants.get(row.id) ?? null) as AccessLevel,
      }))
      // A row we cannot justify showing is one we should not be returning. Dropping beats
      // rendering it greyed out — and whether an inaccessible thing is even acknowledged is
      // a per-cart revocation policy, not a list view's call.
      setCarts(resolved.filter((r) => r.access !== null))
    } catch (e: any) {
      setError(e?.message ?? String(e))
    } finally {
      setLoading(false)
      setLoaded(true)
    }
  }

  useEffect(() => { if (user) load() /* eslint-disable-next-line */ }, [user?.id])

  if (!user) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <UserIcon size={40} className="mx-auto text-slate-600 mb-3" />
          <h2 className="text-lg text-slate-300 mb-1">Not signed in</h2>
          <p className="text-sm text-slate-500">
            Sign in to see your profile, your seat, and the carts you can reach.
          </p>
        </div>
      </div>
    )
  }

  const name = displayNameFor(profile, user.email)
  const owned = carts.filter((c) => c.access === 'owner').length
  const shared = carts.length - owned

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* Identity */}
        <section className="flex items-start gap-4 pb-5 border-b border-slate-800">
          {profile?.avatar_url
            ? <img src={profile.avatar_url} alt="" className="w-16 h-16 rounded-full border border-slate-700" />
            : <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                <UserIcon size={26} className="text-slate-500" />
              </div>}
          <div className="min-w-0 flex-1">
            <h1 className="text-xl text-slate-200">{name}</h1>
            <p className="text-sm text-slate-500">{profile?.email ?? user.email ?? '—'}</p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {(profile?.apps_list ?? []).map((app) => (
                <span key={app} className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-xs text-slate-400">
                  {app}
                </span>
              ))}
              {(profile?.apps_list ?? []).length === 0 && (
                <span className="text-xs text-slate-600">no apps recorded yet</span>
              )}
            </div>
          </div>
          <button
            onClick={load}
            disabled={loading}
            title="Reload from Supabase"
            className="p-2 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 disabled:opacity-40"
          >
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          </button>
        </section>

        {/* Seat */}
        <section>
          <h2 className="text-xs uppercase tracking-wide text-slate-500 mb-2">Seat</h2>
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
            <code className="text-sm text-purple-300 break-all">{user.id}</code>
            <p className="text-xs text-slate-500 mt-2 leading-relaxed">
              Your seat identifier. Your attention — what you have read, marked, and returned
              to — is recorded against this, not against your email. Emails change; this does
              not. Two people searching the same shared cart build different hierarchies over
              it, and this is what keeps yours yours.
            </p>
          </div>
        </section>

        {/* Carts */}
        <section>
          <div className="flex items-baseline justify-between mb-2">
            <h2 className="text-xs uppercase tracking-wide text-slate-500">Carts you can reach</h2>
            {loaded && !error && (
              <span className="text-xs text-slate-600">
                {owned} owned{shared > 0 ? ` · ${shared} shared with you` : ''}
              </span>
            )}
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 flex gap-3">
              <AlertTriangle size={16} className="text-red-400 flex-shrink-0 mt-0.5" />
              <div className="min-w-0">
                <p className="text-sm text-red-300">Could not load carts</p>
                <p className="text-xs text-red-400/80 mt-1 break-words">{error}</p>
                <p className="text-xs text-slate-500 mt-2">
                  If this mentions <code>cart_grants</code>, migration 004 has not been run on
                  this project yet.
                </p>
              </div>
            </div>
          )}

          {!error && loaded && carts.length === 0 && (
            <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-6 text-center">
              <p className="text-sm text-slate-400">No carts yet.</p>
              <p className="text-xs text-slate-600 mt-1">
                Carts you upload appear here, along with any shared with you.
              </p>
            </div>
          )}

          {!error && carts.length > 0 && (
            <ul className="space-y-2">
              {carts.map((cart) => (
                <li
                  key={cart.id}
                  className="rounded-lg border border-slate-800 bg-slate-900/40 p-3 flex items-center gap-3"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-slate-300 truncate">
                      {cart.display_name || cart.cart_filename}
                    </p>
                    <p className="text-xs text-slate-600 truncate">
                      {cart.cart_filename}
                      {cart.pattern_count ? ` · ${cart.pattern_count.toLocaleString()} patterns` : ''}
                      {' · '}{formatBytes(cart.size_bytes)}
                    </p>
                  </div>
                  <AccessBadge level={cart.access} />
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Scope vs secrecy — say it here, where someone is looking at access badges */}
        <section className="rounded-lg border border-slate-800/70 bg-slate-900/20 p-4 flex gap-3">
          <Shield size={15} className="text-slate-600 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-slate-500 leading-relaxed">
            Access levels control what you can <em>do</em> with a cart you can reach.
            Carts you have no access to are not listed here at all — separation between
            carts, not a filter over one.
          </p>
        </section>

        {!session && (
          <p className="text-xs text-slate-600">
            Session not fully initialised — some data may be unavailable until sign-in completes.
          </p>
        )}
      </div>
    </div>
  )
}
