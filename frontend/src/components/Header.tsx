import { useState } from 'react'
import { Activity, Cpu, Lock, LockOpen, Moon, Save, Sun, Zap } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import AuthChip from './AuthChip'

export default function Header() {
  const status = useAppStore((s) => s.status)
  const saveCartridge = useAppStore((s) => s.saveCartridge)
  const toggleLock = useAppStore((s) => s.toggleLock)
  const memboxPanelOpen = useAppStore((s) => s.memboxPanelOpen)
  const toggleMemboxPanel = useAppStore((s) => s.toggleMemboxPanel)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState('')
  const [theme, setTheme] = useState(() =>
    document.documentElement.classList.contains('light') ? 'light' : 'dark'
  )

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    localStorage.setItem('vps-theme', next)
    document.documentElement.classList.toggle('light', next === 'light')
  }

  const handleSave = async () => {
    setSaving(true)
    setSaveMsg('')
    const resp = await saveCartridge()
    setSaveMsg(resp.message)
    setSaving(false)
    if (resp.success) {
      setTimeout(() => setSaveMsg(''), 3000)
    }
  }

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-[var(--chrome-bg)]">
      <div className="flex items-center gap-3">
        <div className="gradient-bg w-8 h-8 rounded-lg flex items-center justify-center">
          <Zap size={18} className="text-white" />
        </div>
        <h1 className="text-xl font-bold gradient-text">Vector+ Studio</h1>
        <span className="text-xs text-slate-500 mt-1">v1.1</span>
      </div>

      <div className="flex items-center gap-4 text-sm">
        {status?.mounted_cartridge && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/60">
            <span className="text-slate-400">Mounted:</span>
            <span className="font-medium text-slate-200">{status.mounted_cartridge}</span>
            <span className="text-slate-500">({status.pattern_count.toLocaleString()} patterns)</span>
          </div>
        )}

        {/* Save button -- visible when there are unsaved changes */}
        {status?.mounted_cartridge && status.dirty && (
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 transition-colors text-xs font-medium disabled:opacity-50"
            title="Save changes to disk"
          >
            <Save size={14} />
            {saving ? 'Saving...' : 'Unsaved changes'}
          </button>
        )}

        {/* Save confirmation message */}
        {saveMsg && !status?.dirty && (
          <span className="text-xs text-green-400">{saveMsg}</span>
        )}

        {/* Lock state — three flavors, in priority order:
              (1) Server-wide read-only (VPS_READ_ONLY env var) → "Public read-only"
              (2) Cart-declared read-only (Step 2a permissions sidecar) → "Cart read-only"
              (3) Per-session toggle (the legacy interactive lock) */}
        {status?.mounted_cartridge && status.read_only_mode && (
          <span
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/50 text-slate-400 cursor-default"
            title="This server is in public read-only mode. All writes are disabled."
          >
            <Lock size={14} />
            Public read-only
          </span>
        )}
        {status?.mounted_cartridge && !status.read_only_mode && status.cart_permissions
          && !String(status.cart_permissions.default).includes('w') && (
          <span
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/50 text-slate-400 cursor-default"
            title={`This cart's .permissions.json declares default="${status.cart_permissions.default}". Edit the sidecar to allow writes.`}
          >
            <Lock size={14} />
            Cart read-only
          </span>
        )}
        {status?.mounted_cartridge && !status.read_only_mode
          && (!status.cart_permissions || String(status.cart_permissions.default).includes('w')) && (
          <button
            onClick={toggleLock}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              status.read_only
                ? 'bg-red-500/15 text-red-400 hover:bg-red-500/25'
                : 'bg-green-500/15 text-green-400 hover:bg-green-500/25'
            }`}
            title={status.read_only ? 'Click to unlock for editing' : 'Click to lock (read-only)'}
          >
            {status.read_only ? <Lock size={14} /> : <LockOpen size={14} />}
            {status.read_only ? 'Read-only' : 'Unlocked'}
          </button>
        )}

        <div
          className="flex items-center gap-2"
          title={status?.gpu_available
            ? 'GPU active -- Lattice physics search enabled'
            : 'CPU mode -- Lattice physics search NOT enabled'}
        >
          <Cpu size={14} className={status?.gpu_available ? 'text-slate-400' : 'text-amber-400'} />
          <span className={`w-2 h-2 rounded-full ${status?.gpu_available ? 'bg-green-400 animate-pulse' : 'bg-amber-400'}`} />
          <span className={`text-xs font-medium ${status?.gpu_available ? 'text-slate-500' : 'text-amber-400'}`}>
            {status?.gpu_available ? 'GPU' : 'CPU'}
          </span>
        </div>

        <button
          onClick={toggleMemboxPanel}
          className={`p-2 rounded-lg transition-colors ${
            memboxPanelOpen
              ? 'bg-purple-500/20 text-purple-300'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
          title="Toggle Membox visualizer"
        >
          <Activity size={16} />
        </button>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-colors"
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        <AuthChip />
      </div>
    </header>
  )
}
