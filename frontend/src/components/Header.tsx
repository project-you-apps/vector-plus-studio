import { useState } from 'react'
import { Cpu, Moon, Save, Sun, Zap } from 'lucide-react'
import { useAppStore } from '../store/appStore'

export default function Header() {
  const status = useAppStore((s) => s.status)
  const saveCartridge = useAppStore((s) => s.saveCartridge)
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
        <span className="text-xs text-slate-500 mt-1">v1.0</span>
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
          onClick={toggleTheme}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 transition-colors"
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  )
}
