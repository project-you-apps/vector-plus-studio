import { useEffect, useState } from 'react'
import { Database, Upload, Plus, Trash2, RotateCcw, ChevronDown, ChevronRight, Loader2, FolderOpen } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { SearchMode } from '../api/types'
import * as api from '../api/client'

const MODES: { key: SearchMode; label: string; desc: string; tooltip: string }[] = [
  { key: 'smart', label: 'Smart Search', desc: 'Physics + cosine blend', tooltip: 'Blends neural lattice physics with cosine similarity. Use the slider to control the mix. Best overall quality. Requires GPU mode.' },
  { key: 'pure_brain', label: 'Pure Brain', desc: 'L2 signatures only', tooltip: 'Searches using only the trained neural lattice -- no embedding database needed. Finds associative relationships cosine misses.' },
  { key: 'fast', label: 'Fast', desc: 'Cosine only', tooltip: 'Standard cosine similarity on embeddings. No GPU required. Fastest but misses physics-discovered associations.' },
]

export default function Sidebar() {
  const {
    cartridges, status, mounting,
    searchMode, blendAlpha,
    deletedPatterns,
    fetchCartridges, mount, unmount,
    setSearchMode, setBlendAlpha,
    restoreResult, fetchDeleted,
  } = useAppStore()

  const [pathOpen, setPathOpen] = useState(false)
  const [pathInput, setPathInput] = useState('')
  const [pathLoading, setPathLoading] = useState(false)
  const [buildOpen, setBuildOpen] = useState(false)
  const [restoreOpen, setRestoreOpen] = useState(false)
  const [buildName, setBuildName] = useState('my_docs')
  const [buildFiles, setBuildFiles] = useState<File[]>([])
  const [building, setBuilding] = useState(false)
  const [buildMsg, setBuildMsg] = useState('')

  useEffect(() => {
    fetchCartridges()
  }, [fetchCartridges])

  useEffect(() => {
    if (restoreOpen && status?.mounted_cartridge) {
      fetchDeleted()
    }
  }, [restoreOpen, status?.mounted_cartridge, fetchDeleted])

  const handleBuild = async () => {
    if (!buildFiles.length || !buildName) return
    setBuilding(true)
    setBuildMsg('')
    try {
      const result = await api.forgeCartridge(buildName, buildFiles)
      setBuildMsg(result.message)
      setBuildFiles([])
      fetchCartridges()
    } catch (e: unknown) {
      setBuildMsg(e instanceof Error ? e.message : 'Build failed')
    } finally {
      setBuilding(false)
    }
  }

  return (
    <aside className="w-72 border-r border-slate-800 bg-[#131620] flex flex-col overflow-hidden">
      {/* Cartridge List */}
      <div className="p-4 max-h-72 overflow-y-auto shrink-0">
        <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
          <Database size={12} /> Cartridges
        </h2>

        {/* Open from file system */}
        <button
          onClick={async () => {
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
              }
            } catch (err) {
              alert(err instanceof Error ? err.message : 'Failed to open file')
            } finally {
              setPathLoading(false)
            }
          }}
          disabled={pathLoading}
          className="w-full flex items-center gap-2 px-3 py-2 mb-1 rounded-lg text-sm transition-all border bg-slate-800/40 border-slate-700/50 text-slate-300 hover:bg-slate-800/70 disabled:opacity-50"
        >
          {pathLoading ? <Loader2 size={14} className="animate-spin" /> : <FolderOpen size={14} />}
          <span className="font-medium">{pathLoading ? 'Opening...' : 'Open Cartridge...'}</span>
        </button>

        {/* Paste path fallback */}
        <button
          onClick={() => setPathOpen(!pathOpen)}
          className="text-[10px] text-slate-600 hover:text-slate-400 transition-colors mb-3 px-1"
        >
          {pathOpen ? 'hide' : 'or paste a path...'}
        </button>

        {pathOpen && (
          <div className="mb-3 space-y-1.5">
            <input
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && pathInput.trim()) {
                  setPathLoading(true)
                  api.mountCartridge(pathInput.trim())
                    .then(() => { fetchCartridges(); useAppStore.getState().fetchStatus() })
                    .catch((err) => alert(err.message))
                    .finally(() => { setPathLoading(false); setPathOpen(false); setPathInput('') })
                }
              }}
              autoFocus
              placeholder="Paste full path to .pkl file"
              className="w-full px-2 py-1.5 text-xs bg-slate-800 border border-slate-700 rounded text-slate-200 placeholder-slate-600 focus:border-purple-500/50 focus:outline-none"
            />
            <button
              onClick={() => {
                if (!pathInput.trim()) return
                setPathLoading(true)
                api.mountCartridge(pathInput.trim())
                  .then(() => { fetchCartridges(); useAppStore.getState().fetchStatus() })
                  .catch((err) => alert(err.message))
                  .finally(() => { setPathLoading(false); setPathOpen(false); setPathInput('') })
              }}
              disabled={pathLoading || !pathInput.trim()}
              className="w-full text-xs py-1.5 rounded gradient-bg text-white font-medium disabled:opacity-50 transition-opacity"
            >
              {pathLoading ? 'Mounting...' : 'Mount'}
            </button>
          </div>
        )}

        {cartridges.length === 0 ? (
          <p className="text-sm text-slate-600 italic">No cartridges found</p>
        ) : (
          <div className="space-y-2">
            {cartridges.map((c) => {
              const isMounted = status?.mounted_cartridge === c.name
              return (
                <div
                  key={c.filename}
                  className={`p-3 rounded-lg border transition-all ${
                    isMounted
                      ? 'border-purple-500/50 bg-purple-500/10'
                      : 'border-slate-700/50 bg-slate-800/30 hover:bg-slate-800/60'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm truncate">{c.name}</span>
                    <span className="text-xs text-slate-500">{c.size_mb} MB</span>
                  </div>

                  <div className="flex gap-1 mb-2">
                    {c.has_brain && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">Brain</span>
                    )}
                    {c.has_signatures && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">Sigs</span>
                    )}
                  </div>

                  {isMounted ? (
                    <button
                      onClick={unmount}
                      className="w-full text-xs py-1.5 rounded bg-slate-700/50 hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
                    >
                      Unmount
                    </button>
                  ) : (
                    <button
                      onClick={() => mount(c.filename)}
                      disabled={mounting}
                      className="w-full text-xs py-1.5 rounded gradient-bg text-white font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
                    >
                      {mounting ? (
                        <span className="flex items-center justify-center gap-1">
                          <Loader2 size={12} className="animate-spin" /> Mounting...
                        </span>
                      ) : (
                        'Mount'
                      )}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Search Mode */}
      <div className="px-4 py-3 border-t border-slate-800">
        <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Search Mode</h2>
        <div className="space-y-1">
          {MODES.map((m) => {
            const isSmartDisabled = m.key === 'smart' && !status?.gpu_available
            const isSmartTraining = m.key === 'smart' && status?.gpu_available && status?.training_active
            const isSmartReady = m.key === 'smart' && status?.gpu_available && status?.physics_trained && !status?.training_active
            let subtitle = m.desc
            if (isSmartDisabled) subtitle = 'Requires GPU'
            else if (isSmartTraining) subtitle = 'Training -- available soon'
            else if (m.key === 'smart' && status?.gpu_available && !status?.physics_trained && !status?.training_active) subtitle = 'Mount a cartridge to enable'

            return (
              <button
                key={m.key}
                onClick={() => !isSmartDisabled && setSearchMode(m.key)}
                title={isSmartDisabled ? 'Smart Search requires a GPU -- currently running in CPU mode' : m.tooltip}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                  isSmartDisabled
                    ? 'opacity-40 cursor-not-allowed text-slate-500 border border-transparent'
                    : searchMode === m.key
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                      : 'text-slate-400 hover:bg-slate-800/60 border border-transparent'
                }`}
              >
                <div className="font-medium flex items-center gap-2">
                  {m.label}
                  {isSmartTraining && <Loader2 size={10} className="animate-spin text-amber-400" />}
                  {isSmartReady && <span className="w-1.5 h-1.5 rounded-full bg-green-400" />}
                </div>
                <div className="text-[10px] opacity-60">{subtitle}</div>
              </button>
            )
          })}
        </div>

        {/* Blend slider */}
        {searchMode === 'smart' && (
          <div className="mt-3">
            <div className="flex justify-between text-xs text-slate-500 mb-1">
              <span>Cosine</span>
              <span>{blendAlpha.toFixed(2)}</span>
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

      {/* Build Cartridge */}
      <div className="border-t border-slate-800">
        <button
          onClick={() => setBuildOpen(!buildOpen)}
          title="Upload documents (.txt, .pdf, .docx) to create a new searchable cartridge"
          className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          {buildOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <Upload size={12} /> Build Cartridge
        </button>

        {buildOpen && (
          <div className="px-4 pb-3 space-y-2">
            <input
              type="text"
              value={buildName}
              onChange={(e) => setBuildName(e.target.value)}
              placeholder="Cartridge name"
              className="w-full px-2 py-1.5 text-sm bg-slate-800 border border-slate-700 rounded text-slate-200 placeholder-slate-600"
            />
            <input
              type="file"
              multiple
              accept=".txt,.pdf,.docx"
              onChange={(e) => setBuildFiles(Array.from(e.target.files || []))}
              className="w-full text-xs text-slate-400 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-slate-700 file:text-slate-300"
            />
            {buildFiles.length > 0 && (
              <p className="text-xs text-slate-500">{buildFiles.length} file(s) selected</p>
            )}
            <button
              onClick={handleBuild}
              disabled={building || !buildFiles.length}
              className="w-full text-xs py-1.5 rounded gradient-bg text-white font-medium disabled:opacity-50"
            >
              {building ? 'Building...' : 'Build'}
            </button>
            {buildMsg && <p className="text-xs text-green-400">{buildMsg}</p>}
          </div>
        )}
      </div>

      {/* Add Passage -- opens main-area editor */}
      {status?.mounted_cartridge && (
        <div className="border-t border-slate-800">
          <button
            onClick={() => useAppStore.getState().openEditor()}
            title="Open the editor to add a new passage to the mounted cartridge"
            className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            <Plus size={12} /> Add Passage
          </button>
        </div>
      )}

      {/* Restore */}
      {status?.mounted_cartridge && status.deleted_count > 0 && (
        <div className="border-t border-slate-800">
          <button
            onClick={() => setRestoreOpen(!restoreOpen)}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            {restoreOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <Trash2 size={12} /> Tombstoned ({status.deleted_count})
          </button>

          {restoreOpen && (
            <div className="px-4 pb-3 space-y-1 max-h-40 overflow-y-auto">
              {deletedPatterns.map((d) => (
                <div key={d.idx} className="flex items-center justify-between p-2 rounded bg-slate-800/40">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-300 truncate">{d.title}</p>
                  </div>
                  <button
                    onClick={() => restoreResult(d.idx)}
                    className="ml-2 p-1 rounded hover:bg-green-500/20 text-slate-500 hover:text-green-400 transition-colors"
                    title="Restore"
                  >
                    <RotateCcw size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Training progress */}
      {status?.training_active && (
        <div className="px-4 py-3 border-t border-slate-800">
          <p className="text-xs text-amber-400 mb-1">
            Training: {status.training_progress.toLocaleString()} / {status.training_total.toLocaleString()}
          </p>
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full gradient-bg transition-all duration-500"
              style={{ width: `${status.training_total > 0 ? (status.training_progress / status.training_total) * 100 : 0}%` }}
            />
          </div>
        </div>
      )}
    </aside>
  )
}
