import { useEffect, useState } from 'react'
import { Cookie, X } from 'lucide-react'

// GDPR-style cookie consent banner.
//
// Shows on first visit; persists choice in localStorage. Three buttons:
//   - "Reject non-essential" — only essential (auth/session) cookies remain
//   - "Customize" — modal with per-category toggles
//   - "Accept all" — opt in to everything (analytics, marketing) for the future
//
// Categories:
//   - essential: always on, can't be disabled (the Supabase auth cookie lives here)
//   - analytics: off by default (Plausible / etc. when we ship it)
//   - marketing: off by default (not currently used)
//
// localStorage key is versioned so we can prompt re-acceptance if the policy
// changes meaningfully. Bump CONSENT_VERSION when that happens.
const CONSENT_VERSION = '1'
const CONSENT_KEY = `cookie_consent_v${CONSENT_VERSION}`

interface ConsentState {
  version: string
  essential: true            // always true, kept for clarity
  analytics: boolean
  marketing: boolean
  timestamp: string
}

function readConsent(): ConsentState | null {
  try {
    const raw = localStorage.getItem(CONSENT_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed?.version !== CONSENT_VERSION) return null
    return parsed
  } catch { return null }
}

function writeConsent(c: ConsentState): void {
  try { localStorage.setItem(CONSENT_KEY, JSON.stringify(c)) } catch { /* private mode */ }
}

export default function CookieBanner() {
  const [open, setOpen] = useState(false)
  const [customizing, setCustomizing] = useState(false)
  const [analytics, setAnalytics] = useState(false)
  const [marketing, setMarketing] = useState(false)

  useEffect(() => {
    if (readConsent() === null) setOpen(true)
  }, [])

  if (!open) return null

  const save = (a: boolean, m: boolean) => {
    writeConsent({
      version: CONSENT_VERSION,
      essential: true,
      analytics: a,
      marketing: m,
      timestamp: new Date().toISOString(),
    })
    setOpen(false)
  }

  const rejectNonEssential = () => save(false, false)
  const acceptAll = () => save(true, true)
  const saveCustom = () => save(analytics, marketing)

  return (
    <>
      {/* Main banner — fixed to bottom */}
      {!customizing && (
        <div className="fixed bottom-0 inset-x-0 z-50 px-4 pb-4 pointer-events-none">
          <div className="max-w-4xl mx-auto rounded-xl border border-slate-700/60 bg-slate-900/95 backdrop-blur shadow-2xl pointer-events-auto">
            <div className="p-4 md:p-5 flex items-start gap-3 md:gap-4">
              <Cookie size={20} className="shrink-0 text-purple-400 mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="text-base font-semibold text-slate-100 mb-1.5">Cookie Policy</div>
                <div className="text-xs text-slate-400 leading-relaxed">
                  We use one essential cookie to keep you signed in across the Waving Cat apps on{' '}
                  <code className="text-purple-300">.project-you.app</code>. Nothing tracks you yet.
                  If we add analytics later, those will be off by default until you opt in here.{' '}
                  <a href="/privacy.html" className="text-purple-400 hover:text-purple-300 underline">Privacy</a>
                  {' · '}
                  <a href="/terms.html" className="text-purple-400 hover:text-purple-300 underline">Terms</a>
                </div>
              </div>
              <div className="shrink-0 flex flex-col sm:flex-row gap-2">
                <button
                  onClick={rejectNonEssential}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors whitespace-nowrap"
                >
                  Reject non-essential
                </button>
                <button
                  onClick={() => setCustomizing(true)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors whitespace-nowrap"
                >
                  Customize
                </button>
                <button
                  onClick={acceptAll}
                  className="px-3 py-1.5 rounded-lg gradient-bg text-white text-xs font-medium hover:opacity-90 transition-opacity whitespace-nowrap"
                >
                  Accept all
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Customize modal */}
      {customizing && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) setCustomizing(false) }}
        >
          <div className="relative w-full max-w-md mx-4 rounded-2xl border border-slate-700/50 bg-slate-900 shadow-2xl">
            <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cookie size={16} className="text-purple-400" />
                <h2 className="text-sm font-medium text-slate-200">Customize cookies</h2>
              </div>
              <button
                onClick={() => setCustomizing(false)}
                className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
              >
                <X size={14} />
              </button>
            </div>

            <div className="px-5 py-4 space-y-3 text-sm">
              <CategoryRow
                title="Essential"
                description="Required for sign-in to work. Includes the Supabase session token cookie. Can't be disabled."
                checked={true}
                disabled={true}
                onChange={() => {}}
              />
              <CategoryRow
                title="Analytics"
                description="Aggregate, privacy-respecting site analytics so we can see what's used and what isn't. Not currently implemented; toggle has no effect until we add it."
                checked={analytics}
                disabled={false}
                onChange={setAnalytics}
              />
              <CategoryRow
                title="Marketing"
                description="Not currently used. Reserved for future ad-attribution or conversion tracking (which we'd announce before turning on)."
                checked={marketing}
                disabled={false}
                onChange={setMarketing}
              />

              <div className="pt-2 flex items-center justify-end gap-2">
                <button
                  onClick={() => setCustomizing(false)}
                  className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200"
                >
                  Cancel
                </button>
                <button
                  onClick={saveCustom}
                  className="px-3 py-1.5 rounded-lg gradient-bg text-white text-xs font-medium hover:opacity-90"
                >
                  Save my choices
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function CategoryRow({ title, description, checked, disabled, onChange }: {
  title: string
  description: string
  checked: boolean
  disabled: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/40 border border-slate-700/40">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 w-4 h-4 accent-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-slate-200">{title}</div>
        <div className="text-xs text-slate-500 leading-relaxed mt-0.5">{description}</div>
      </div>
    </div>
  )
}
