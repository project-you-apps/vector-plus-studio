import { useEffect, useState } from 'react'
import { Upload, Plus, Trash2, RotateCcw, ChevronDown, ChevronRight } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import * as api from '../api/client'

export default function Sidebar() {
  // 2026-05-03 reorg: cart picker AND search mode picker have moved to
  // SearchToolbar. This sidebar now owns: Build Cartridge expandable, Add
  // Passage button, Tombstoned restore, Training progress. Next-pass moves
  // remove Build Cartridge into its own full screen.
  const {
    status,
    deletedPatterns,
    fetchCartridges,
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
      {/* Build Cartridge */}
      <div>
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
