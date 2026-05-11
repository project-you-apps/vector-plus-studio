import { useEffect } from 'react'
import { Folder, X, ChevronRight, ChevronUp, Check, HardDrive } from 'lucide-react'
import { useCartBuilderStore } from '../store/cartBuilderStore'

// FolderPickerModal — server-side path browser. Used to add saved-folder
// roots for the cart browser, and to pick ingestion source dirs. Walks
// the actual server filesystem via /api/cartbuilder/browse.

export default function FolderPickerModal() {
  const {
    pickerOpen, pickerPath, pickerDirs, pickerParent,
    closeFolderPicker, navigateFolderPicker, addBrowserFolder,
    openFolderPicker, pickerOnConfirm,
  } = useCartBuilderStore()

  useEffect(() => {
    if (!pickerOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeFolderPicker()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [pickerOpen, closeFolderPicker])

  if (!pickerOpen) return null

  const isRoot = !pickerPath
  const pathDisplay = pickerPath || '(drive roots)'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) closeFolderPicker() }}
    >
      <div className="relative w-full max-w-2xl max-h-[80vh] mx-4 flex flex-col rounded-2xl border border-slate-700/50 bg-slate-900 shadow-2xl">
        {/* Header */}
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Folder size={16} className="text-purple-400 shrink-0" />
            <h2 className="text-sm font-medium text-slate-200">Pick a folder</h2>
            <span className="text-xs text-slate-500 font-mono truncate" title={pathDisplay}>
              {pathDisplay}
            </span>
          </div>
          <button
            onClick={closeFolderPicker}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
          >
            <X size={14} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="px-5 py-2 border-b border-slate-700/40 flex items-center gap-2">
          {pickerParent && !isRoot && (
            <button
              onClick={() => navigateFolderPicker(pickerParent)}
              className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-700/40 flex items-center gap-1"
            >
              <ChevronUp size={11} /> Up
            </button>
          )}
          <button
            onClick={() => openFolderPicker({ path: '' })}
            className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-700/40"
          >
            Drives / root
          </button>
          {!isRoot && (
            <button
              onClick={() => {
                // If a caller registered a custom onConfirm (e.g., the New
                // Cart destination flow), use it. Otherwise fall back to
                // the legacy behavior of adding the picked folder to the
                // user's saved-cart-folders bookmark list.
                if (pickerOnConfirm) {
                  pickerOnConfirm(pickerPath)
                } else {
                  addBrowserFolder(pickerPath)
                }
                closeFolderPicker()
              }}
              className="ml-auto text-xs px-3 py-1 rounded bg-purple-500/20 border border-purple-500/40 text-purple-300 hover:bg-purple-500/30 flex items-center gap-1"
            >
              <Check size={11} /> Use this folder
            </button>
          )}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto">
          {pickerDirs.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-600 italic">
              {isRoot ? 'No drives detected.' : 'Empty folder.'}
            </div>
          ) : (
            <div className="divide-y divide-slate-800/60">
              {pickerDirs.map((name) => {
                const sep = pickerPath.includes('\\') ? '\\' : '/'
                const child = isRoot
                  ? name  // drive letters on Windows already include trailing separator
                  : `${pickerPath.replace(/[/\\]+$/, '')}${sep}${name}`
                return (
                  <button
                    key={child}
                    onClick={() => navigateFolderPicker(child)}
                    className="w-full text-left px-5 py-2 flex items-center gap-3 hover:bg-slate-800/40 text-sm"
                  >
                    {isRoot
                      ? <HardDrive size={13} className="text-slate-500 shrink-0" />
                      : <Folder size={13} className="text-amber-400/70 shrink-0" />}
                    <span className="flex-1 truncate text-slate-300 font-mono text-xs">{name}</span>
                    <ChevronRight size={11} className="text-slate-600" />
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div className="px-5 py-2 border-t border-slate-700/40 text-[10px] text-slate-600 italic">
          Click a folder to navigate in. "Use this folder" adds the current path to saved cart folders.
        </div>
      </div>
    </div>
  )
}
