import { useState, useEffect } from 'react'
import { Sun, Moon, Settings as SettingsIcon, Info, ShieldAlert, Unlock, Database } from 'lucide-react'
import { useAppStore } from '../store/appStore'

// Settings screen — global preferences and search defaults. Read-write,
// persists to Zustand (in-memory) and localStorage (theme). Future:
// add advanced flags, embedder backend selection, droplet endpoint.

function ToggleRow({ label, description, checked, onChange }: {
  label: string
  description?: string
  checked: boolean
  onChange: (next: boolean) => void
}) {
  return (
    <label className="flex items-start gap-3 py-3 cursor-pointer hover:bg-slate-800/30 -mx-3 px-3 rounded transition-colors">
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`mt-0.5 relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${
          checked ? 'bg-purple-500/60' : 'bg-slate-700'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
            checked ? 'translate-x-4' : 'translate-x-0'
          }`}
        />
      </button>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-slate-200">{label}</div>
        {description && <div className="text-xs text-slate-500 mt-0.5">{description}</div>}
      </div>
    </label>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4">
      <h2 className="text-xs uppercase tracking-wider text-slate-500 mb-2">{title}</h2>
      <div className="space-y-1">{children}</div>
    </div>
  )
}

export default function SettingsScreen() {
  const {
    topK, setTopK,
    strictMode, setStrictMode,
    exactMatch, setExactMatch,
  } = useAppStore()
  // 2026-07-23 -- Permission UI MVP #5. Surface the mounted cart's
  // permission sidecar as a chip + owner line so users can check "what
  // access do I have on this thing?" without opening Edit Carts.
  const status = useAppStore((s) => s.status)
  const currentCartPermissions = useAppStore((s) => s.currentCartPermissions)
  const backendMounted = !!status?.mounted_cartridge
  const permissionWritable =
    currentCartPermissions == null ||
    (typeof currentCartPermissions.default === 'string' && currentCartPermissions.default.includes('w'))

  const [theme, setTheme] = useState(() =>
    document.documentElement.classList.contains('light') ? 'light' : 'dark'
  )

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light')
    localStorage.setItem('vps-theme', theme)
  }, [theme])

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="max-w-3xl mx-auto w-full space-y-6">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
            <SettingsIcon size={28} className="text-purple-300" />
            Settings
          </h1>
          <p className="text-sm text-slate-500">Global preferences, search defaults, appearance.</p>
        </div>

        {/* 2026-07-23 -- Permission UI MVP #5. Mounted Cart section. Only
            renders when a backend cart is mounted (LocalCarts have no
            sidecar and are always writable, so the chip would just say
            "Editable" every time -- noise). Chip mirrors the one in the
            CRUDScreen banner + OpenCartBanner so users learn one visual. */}
        {backendMounted && (
          <Section title="Mounted Cart">
            <div className="py-3 flex items-center gap-3">
              <Database size={16} className="text-purple-400 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-slate-200 truncate">
                  {status?.mounted_cartridge}
                </div>
                <div className="text-xs text-slate-500">
                  {status?.pattern_count?.toLocaleString() ?? 0} patterns
                </div>
              </div>
              {permissionWritable ? (
                <span
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-green-500/20 border border-green-500/40 text-green-300"
                  title={
                    currentCartPermissions == null
                      ? 'Legacy cart — no permissions sidecar. Treated as writable by default.'
                      : `Sidecar declares default="${currentCartPermissions.default}" — includes write access.`
                  }
                >
                  <Unlock size={11} /> Editable
                </span>
              ) : (
                <span
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-amber-500/20 border border-amber-500/40 text-amber-200"
                  title="Cart owner has set this cart to read-only. Only the owner can change this."
                >
                  <ShieldAlert size={11} /> Read-only
                </span>
              )}
            </div>
            {currentCartPermissions && (
              <div className="border-t border-slate-800 pt-3 pb-1 space-y-1.5 text-xs">
                {currentCartPermissions.owner && (
                  <div className="flex gap-3">
                    <span className="text-slate-500 w-20 shrink-0">Owner</span>
                    <span className="font-mono text-slate-300 truncate">{currentCartPermissions.owner}</span>
                  </div>
                )}
                {currentCartPermissions.description && (
                  <div className="flex gap-3">
                    <span className="text-slate-500 w-20 shrink-0">About</span>
                    <span className="text-slate-300 italic">{currentCartPermissions.description}</span>
                  </div>
                )}
                {!permissionWritable && (
                  <div className="mt-2 text-[11px] text-slate-500 leading-snug">
                    To change permissions, the owner can rewrite the cart's{' '}
                    <code className="font-mono text-slate-400">.permissions.json</code> sidecar
                    (or use <code className="font-mono text-slate-400">bin/set_cart_permissions.py</code>).
                  </div>
                )}
              </div>
            )}
          </Section>
        )}

        {/* Search Defaults */}
        <Section title="Search Defaults">
          <div className="py-3">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-sm font-medium text-slate-200">Top-K results</div>
                <div className="text-xs text-slate-500">How many results to return per search.</div>
              </div>
              <span className="text-sm font-mono text-purple-300 px-2 py-1 rounded bg-purple-500/10 border border-purple-500/30">
                {topK}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={50}
              step={1}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-purple-500 bg-slate-700"
            />
            <div className="flex justify-between text-[10px] text-slate-600 mt-1">
              <span>1</span>
              <span>10</span>
              <span>25</span>
              <span>50</span>
            </div>
          </div>

          <div className="border-t border-slate-800 pt-1">
            <ToggleRow
              label="Strict keyword filter"
              description="Drop results that don't contain at least one of the query's content words. Trades recall for precision."
              checked={strictMode}
              onChange={setStrictMode}
            />
          </div>
          <div className="border-t border-slate-800 pt-1">
            <ToggleRow
              label="Exact phrase match"
              description="Require the full query phrase to appear in the result. Strongest filter; small recall, very high precision."
              checked={exactMatch}
              onChange={setExactMatch}
            />
          </div>
        </Section>

        {/* Appearance */}
        <Section title="Appearance">
          <div className="py-3 flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-slate-200">Theme</div>
              <div className="text-xs text-slate-500">Dark or light. Same toggle also lives in the header.</div>
            </div>
            <div className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/40 p-0.5">
              <button
                onClick={() => setTheme('dark')}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center gap-1.5 ${
                  theme === 'dark'
                    ? 'bg-purple-500/20 text-purple-300'
                    : 'text-slate-500 hover:text-slate-200'
                }`}
              >
                <Moon size={12} /> Dark
              </button>
              <button
                onClick={() => setTheme('light')}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors flex items-center gap-1.5 ${
                  theme === 'light'
                    ? 'bg-purple-500/20 text-purple-300'
                    : 'text-slate-500 hover:text-slate-200'
                }`}
              >
                <Sun size={12} /> Light
              </button>
            </div>
          </div>
        </Section>

        {/* Backend */}
        <Section title="Backend">
          <div className="py-3 space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">API URL</span>
              <span className="font-mono text-xs text-slate-300">localhost:8000</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Frontend URL</span>
              <span className="font-mono text-xs text-slate-300">localhost:5173</span>
            </div>
          </div>
        </Section>

        {/* Future settings placeholder */}
        <div className="rounded-lg border border-dashed border-slate-700 bg-slate-800/10 p-4 text-center">
          <Info size={14} className="inline text-slate-600 mr-1.5 mb-0.5" />
          <span className="text-xs text-slate-600 italic">
            More coming: embedder backend selector, FAST/BALANCED/QUALITY profile, advanced search flags, droplet endpoint override, Membox auth token.
          </span>
        </div>
      </div>
    </main>
  )
}
