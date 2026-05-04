import { useEffect, useState } from 'react'
import { Upload, Plus, Trash2, RotateCcw, ChevronDown, ChevronRight, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { SearchMode } from '../api/types'
import * as api from '../api/client'

const MODES: { key: SearchMode; label: string; desc: string; tooltip: string }[] = [
  { key: 'hamming', label: 'Hamming Blend', desc: '70% cosine + 30% Hamming', tooltip: '70% cosine + 30% sign-zero Hamming with keyword reranking. Same as Membot production search. No GPU required.' },
  { key: 'smart', label: 'Smart Search', desc: 'Physics + cosine blend', tooltip: 'Blends neural lattice physics with cosine similarity. Use the slider to control the mix. Best overall quality. Requires GPU mode.' },
  { key: 'pure_brain', label: 'Pure Brain', desc: 'L2 signatures only', tooltip: 'Searches using only the trained neural lattice -- no embedding database needed. Finds associative relationships cosine misses.' },
  { key: 'fast', label: 'Fast', desc: 'Cosine only', tooltip: 'Standard cosine similarity on embeddings. No GPU required. Fastest but misses physics-discovered associations.' },
  { key: 'associate', label: 'Associate', desc: 'Physics-driven association', tooltip: 'Settle the query through the trained lattice and rank by what the physics surfaces. Finds cross-domain associations (e.g. earthquakes → Poseidon). Requires GPU + trained cartridge.' },
]

export default function Sidebar() {
  // Cart picker moved to SearchToolbar (2026-05-03 reorg). This sidebar now
  // owns: search mode picker, build cartridge expandable, add passage button,
  // tombstoned restore, training progress. Future iterations migrate the
  // search mode picker to the SearchToolbar dropdown and Build Cartridge to
  // its own full screen.
  const {
    status,
    searchMode, blendAlpha,
    deletedPatterns,
    fetchCartridges,
    setSearchMode, setBlendAlpha,
    restoreResult, fetchDeleted,
  } = useAppStore()

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
    <aside className="w-72 border-r border-slate-800 bg-[var(--chrome-bg)] flex flex-col overflow-hidden">
      {/* Search Mode */}
      <div className="px-4 py-4">
        <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Search Mode</h2>
        <div className="space-y-1">
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
            else if (needsSigs && !status?.signatures_loaded && stillTraining) subtitle = 'Building signatures...'
            else if (needsSigs && !status?.signatures_loaded) subtitle = 'Signatures not available'
            else if (needsFullBrain && stillTraining) subtitle = 'Training -- available soon'
            else if (needsFullBrain && status?.gpu_available && !status?.physics_trained && !stillTraining) subtitle = 'Mount a cartridge to enable'

            return (
              <button
                key={m.key}
                onClick={() => !isDisabled && !isTraining && setSearchMode(m.key)}
                title={isDisabled ? (needsSigs ? `${m.label} requires built signatures` : `${m.label} requires a GPU -- currently running in CPU mode`) : m.tooltip}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all ${
                  isDisabled || isTraining
                    ? 'opacity-40 cursor-not-allowed text-slate-500 border border-transparent'
                    : searchMode === m.key
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                      : 'text-slate-400 hover:bg-slate-800/60 border border-transparent'
                }`}
              >
                <div className="font-medium flex items-center gap-2">
                  {m.label}
                  {isTraining && <Loader2 size={10} className="animate-spin text-amber-400" />}
                  {isReady && <span className="w-1.5 h-1.5 rounded-full bg-green-400" />}
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
