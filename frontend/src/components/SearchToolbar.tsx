import { useEffect, useRef, useState } from 'react'
import { Database, ChevronDown, FolderOpen, Loader2, Zap } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { SearchMode } from '../api/types'
import * as api from '../api/client'

// Search modes (lifted from Sidebar.tsx in 2026-05-03 reorg Phase 3).
// Disable/training/ready logic depends on status.gpu_available,
// status.physics_trained, status.training_active, status.signatures_loaded.
const MODES: { key: SearchMode; label: string; desc: string; tooltip: string }[] = [
  { key: 'hamming',    label: 'Hamming Blend', desc: '70% cosine + 30% Hamming',  tooltip: '70% cosine + 30% sign-zero Hamming with keyword reranking. Same as Membot production search. No GPU required.' },
  { key: 'smart',      label: 'Smart Search',  desc: 'Physics + cosine blend',     tooltip: 'Blends neural lattice physics with cosine similarity. Use the slider to control the mix. Best overall quality. Requires GPU mode.' },
  { key: 'pure_brain', label: 'Pure Brain',    desc: 'L2 signatures only',         tooltip: 'Searches using only the trained neural lattice — no embedding database needed. Finds associative relationships cosine misses.' },
  { key: 'fast',       label: 'Fast',          desc: 'Cosine only',                tooltip: 'Standard cosine similarity on embeddings. No GPU required. Fastest but misses physics-discovered associations.' },
  { key: 'associate',  label: 'Associate',     desc: 'Physics-driven association', tooltip: 'Settle the query through the trained lattice and rank by what the physics surfaces. Finds cross-domain associations (e.g. earthquakes → Poseidon). Requires GPU + trained cartridge.' },
]

// Top-of-screen toolbar for the Search screen. Owns the cart picker that
// previously lived in the Sidebar's top section. This is the per-screen
// toolbar pattern from the 2026-05-03 sidebar reorg — controls that operate
// on the *current screen's* state belong here, not in the global nav rail.
export default function SearchToolbar() {
  const {
    cartridges, status, mounting,
    searchMode, blendAlpha,
    fetchCartridges, mount, unmount,
    setSearchMode, setBlendAlpha,
  } = useAppStore()

  const [open, setOpen] = useState(false)
  const [modeOpen, setModeOpen] = useState(false)
  const [pathOpen, setPathOpen] = useState(false)
  const [pathInput, setPathInput] = useState('')
  const [pathLoading, setPathLoading] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const modeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchCartridges()
  }, [fetchCartridges])

  // Close dropdowns when clicking outside (one handler for both)
  useEffect(() => {
    if (!open && !modeOpen) return
    const handler = (e: MouseEvent) => {
      const target = e.target as Node
      if (open && ref.current && !ref.current.contains(target)) setOpen(false)
      if (modeOpen && modeRef.current && !modeRef.current.contains(target)) setModeOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open, modeOpen])

  const currentMode = MODES.find((m) => m.key === searchMode) ?? MODES[0]

  const mounted = status?.mounted_cartridge ?? null
  const mountedCart = mounted ? cartridges.find((c) => c.name === mounted) : null

  const handleBrowse = async () => {
    setPathLoading(true)
    try {
      const path = await api.browseForCartridge()
      if (path) {
        const res = await api.mountCartridge(path)
        if (!res.success) {
          alert(res.message || 'Mount failed')
          return
        }
        fetchCartridges()
        useAppStore.getState().fetchStatus()
        setOpen(false)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to open file')
    } finally {
      setPathLoading(false)
    }
  }

  const handlePaste = () => {
    if (!pathInput.trim()) return
    setPathLoading(true)
    api.mountCartridge(pathInput.trim())
      .then(() => {
        fetchCartridges()
        useAppStore.getState().fetchStatus()
      })
      .catch((err) => alert(err.message))
      .finally(() => {
        setPathLoading(false)
        setPathOpen(false)
        setPathInput('')
        setOpen(false)
      })
  }

  return (
    <div className="flex items-center gap-3 px-6 py-2.5 border-b border-slate-800 bg-[var(--chrome-bg)]/40 flex-shrink-0">
      {/* Cart picker dropdown */}
      <div className="relative" ref={ref}>
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border border-slate-700 bg-slate-800/40 hover:bg-slate-800/70 hover:border-purple-500/40 transition-colors min-w-[280px] cursor-pointer"
          title={mounted ? 'Click to change or unmount the current cartridge' : 'Click to mount a cartridge'}
        >
          <Database size={14} className={mounted ? 'text-purple-400' : 'text-slate-500'} />
          <span className="flex-1 text-left truncate">
            {mounted ? (
              <>
                <span className="text-slate-500 mr-1.5">Cartridge:</span>
                <span className="text-slate-100 font-medium">{mounted}</span>
                {mountedCart && (
                  <span className="text-slate-500 ml-1.5 text-xs">({mountedCart.size_mb.toFixed(1)} MB)</span>
                )}
              </>
            ) : (
              <span className="text-slate-500 italic">Click to mount a cartridge…</span>
            )}
          </span>
          <ChevronDown size={14} className={`text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>

        {open && (
          <div className="absolute top-full mt-1 left-0 w-96 max-h-[28rem] overflow-y-auto rounded-lg border border-slate-700 bg-[var(--chrome-bg)] shadow-2xl z-30">
            {/* Open from file system */}
            <button
              onClick={handleBrowse}
              disabled={pathLoading}
              className="w-full flex items-center gap-2 px-3 py-2.5 text-sm border-b border-slate-800 hover:bg-slate-800/50 disabled:opacity-50 transition-colors"
            >
              {pathLoading ? <Loader2 size={14} className="animate-spin" /> : <FolderOpen size={14} />}
              <span className="font-medium">{pathLoading ? 'Opening…' : 'Open Cartridge…'}</span>
            </button>

            {/* Paste path fallback */}
            <div className="border-b border-slate-800">
              <button
                onClick={() => setPathOpen(!pathOpen)}
                className="w-full text-left px-3 py-1.5 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
              >
                {pathOpen ? '▾ paste path' : '▸ or paste a path…'}
              </button>
              {pathOpen && (
                <div className="px-3 pb-2 space-y-1.5">
                  <input
                    type="text"
                    value={pathInput}
                    onChange={(e) => setPathInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handlePaste()}
                    autoFocus
                    placeholder="Paste full path to .pkl/.npz file"
                    className="w-full px-2 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded text-slate-200 placeholder-slate-600 focus:border-purple-500/50 focus:outline-none"
                  />
                  <button
                    onClick={handlePaste}
                    disabled={pathLoading || !pathInput.trim()}
                    className="w-full text-xs py-1.5 rounded gradient-bg text-white font-medium disabled:opacity-50"
                  >
                    {pathLoading ? 'Mounting…' : 'Mount'}
                  </button>
                </div>
              )}
            </div>

            {/* Available carts */}
            <div className="p-2 space-y-1">
              <div className="px-1 py-1 text-[10px] uppercase tracking-wider text-slate-600">
                Available ({cartridges.length})
              </div>
              {cartridges.length === 0 ? (
                <div className="px-2 py-3 text-xs text-slate-600 italic">No cartridges found</div>
              ) : (
                cartridges.map((c) => {
                  const isMounted = mounted === c.name
                  return (
                    <div
                      key={c.filename}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-all ${
                        isMounted
                          ? 'bg-purple-500/10 border border-purple-500/30'
                          : 'border border-transparent hover:bg-slate-800/40'
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate text-slate-200">{c.name}</div>
                        <div className="text-[10px] text-slate-500 flex gap-1.5">
                          <span>{c.size_mb} MB</span>
                          {c.has_brain && <span className="text-green-400">Brain</span>}
                          {c.has_signatures && <span className="text-blue-400">Sigs</span>}
                        </div>
                      </div>
                      {isMounted ? (
                        <button
                          onClick={() => {
                            unmount()
                            setOpen(false)
                          }}
                          className="text-xs px-2 py-1 rounded text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        >
                          Unmount
                        </button>
                      ) : (
                        <button
                          onClick={() => {
                            mount(c.filename)
                            setOpen(false)
                          }}
                          disabled={mounting}
                          className="text-xs px-2.5 py-1 rounded gradient-bg text-white font-medium disabled:opacity-50 hover:opacity-90 transition-opacity"
                        >
                          {mounting ? <Loader2 size={11} className="animate-spin" /> : 'Mount'}
                        </button>
                      )}
                    </div>
                  )
                })
              )}
            </div>
          </div>
        )}
      </div>

      {/* Status pill — mounted cart pattern count */}
      {status?.mounted_cartridge && (
        <span className="text-xs text-slate-500 font-mono">
          {status.pattern_count.toLocaleString()} patterns
        </span>
      )}

      {/* Search mode dropdown */}
      <div className="relative" ref={modeRef}>
        <button
          onClick={() => setModeOpen(!modeOpen)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border border-slate-700 bg-slate-800/40 hover:bg-slate-800/70 transition-colors min-w-[200px]"
          title={`Search mode: ${currentMode.label} — ${currentMode.tooltip}`}
        >
          <Zap size={14} className="text-purple-400" />
          <span className="flex-1 text-left text-slate-200 font-medium">{currentMode.label}</span>
          <ChevronDown size={14} className={`text-slate-500 transition-transform ${modeOpen ? 'rotate-180' : ''}`} />
        </button>

        {modeOpen && (
          <div className="absolute top-full mt-1 left-0 w-72 rounded-lg border border-slate-700 bg-[var(--chrome-bg)] shadow-2xl z-30 p-2 space-y-1">
            <div className="px-1 py-1 text-[10px] uppercase tracking-wider text-slate-600">Search mode</div>
            {MODES.map((m) => {
              const needsFullBrain = m.key === 'smart' || m.key === 'associate'
              const needsSigs = m.key === 'pure_brain'
              const stillTraining = !!status?.training_active
              const isDisabled =
                (needsFullBrain && !status?.gpu_available) ||
                (needsFullBrain && status?.gpu_available && (!status?.physics_trained || stillTraining)) ||
                (needsSigs && !status?.signatures_loaded)
              const isTraining =
                (needsFullBrain && status?.gpu_available && stillTraining) ||
                (needsSigs && !status?.signatures_loaded && stillTraining)
              const isReady =
                (needsFullBrain && status?.physics_trained && !stillTraining) ||
                (needsSigs && status?.signatures_loaded)
              let subtitle = m.desc
              if (needsFullBrain && !status?.gpu_available) subtitle = 'Requires GPU'
              else if (needsSigs && !status?.signatures_loaded && stillTraining) subtitle = 'Building signatures…'
              else if (needsSigs && !status?.signatures_loaded) subtitle = 'Signatures not available'
              else if (needsFullBrain && stillTraining) subtitle = 'Training — available soon'
              else if (needsFullBrain && status?.gpu_available && !status?.physics_trained && !stillTraining) subtitle = 'Mount a cartridge to enable'

              const active = searchMode === m.key
              return (
                <button
                  key={m.key}
                  onClick={() => {
                    if (isDisabled || isTraining) return
                    setSearchMode(m.key)
                    setModeOpen(false)
                  }}
                  title={isDisabled ? (needsSigs ? `${m.label} requires built signatures` : `${m.label} requires a GPU — currently running in CPU mode`) : m.tooltip}
                  className={`w-full text-left px-2.5 py-2 rounded text-sm transition-all ${
                    isDisabled || isTraining
                      ? 'opacity-40 cursor-not-allowed text-slate-500'
                      : active
                        ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                        : 'text-slate-300 hover:bg-slate-800/50 border border-transparent'
                  }`}
                >
                  <div className="font-medium flex items-center gap-2">
                    {m.label}
                    {isTraining && <Loader2 size={10} className="animate-spin text-amber-400" />}
                    {isReady && !active && <span className="w-1.5 h-1.5 rounded-full bg-green-400" />}
                  </div>
                  <div className="text-[10px] opacity-60 mt-0.5">{subtitle}</div>
                </button>
              )
            })}

            {/* Blend slider (only when smart mode active) */}
            {searchMode === 'smart' && (
              <div className="px-2 pt-2 mt-1 border-t border-slate-800">
                <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                  <span>Cosine</span>
                  <span className="font-mono">{blendAlpha.toFixed(2)}</span>
                  <span>Physics</span>
                </div>
                <input
                  type="range"
                  min={0} max={1} step={0.05}
                  value={blendAlpha}
                  onChange={(e) => setBlendAlpha(parseFloat(e.target.value))}
                  className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-purple-500 bg-slate-700"
                />
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex-1" />
    </div>
  )
}
