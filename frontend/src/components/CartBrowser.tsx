import { useEffect, useState } from 'react'
import {
  Folder, Database, FileText, ChevronRight, FolderPlus, Trash2,
  ChevronUp, Plus, RefreshCw,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import type { CartBuilderListedCart, CartBuilderDoc } from '../api/cartbuilder'

// CartBrowser — shared "My Carts" panel embedded in both Cart Builder and
// Edit Carts (Andy 2026-05-05 IA Q2). Lists carts, ingestable docs, and
// subdirectories with breadcrumb navigation. Folder management (add/remove
// saved root folders) lives here too.
//
// Click handlers are passed in by the parent so each screen can decide what
// "click on cart" means (Cart Builder = load into workspace; Edit Carts =
// mount for passage-level CRUD).

interface Props {
  headerLabel?: string
  onCartClick?: (cart: CartBuilderListedCart) => void
  onDocClick?: (doc: CartBuilderDoc) => void  // for Cart Builder ingest-by-path
}

export default function CartBrowser({
  headerLabel = 'My Carts',
  onCartClick,
  onDocClick,
}: Props) {
  const {
    browserCarts, browserSubdirs, browserDocs,
    browserFolders, browserCurrentPath,
    refreshBrowser, removeBrowserFolder,
    ingestPath, openFolderPicker,
  } = useCartBuilderStore()

  // In read-only mode (public droplet), filesystem-walking endpoints are 403'd
  // by the backend. Hide the affordances that would only produce errors:
  //   • Folders button + folder management (would need /browse and /carts?path=…)
  //   • Subdir drill-down (would need /carts?path=…)
  //   • Up button (same)
  // The saved-folders root list still works (no path) so users can see the
  // operator-curated cart catalog.
  const readOnlyMode = useAppStore((s) => s.status?.read_only_mode ?? false)

  const [showFolderManager, setShowFolderManager] = useState(false)

  useEffect(() => {
    refreshBrowser('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const isRoot = !browserCurrentPath
  const goUp = () => {
    if (!browserCurrentPath) return
    // Navigate to parent — use browse-folder semantics
    const parent = browserCurrentPath.replace(/[/\\][^/\\]+[/\\]?$/, '') || '/'
    refreshBrowser(parent === browserCurrentPath ? '' : parent)
  }

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/30">
      <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between flex-wrap gap-2">
        <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
          <Folder size={12} />
          {headerLabel}
          {browserCurrentPath && (
            <span className="text-[10px] normal-case text-slate-600 font-mono truncate max-w-md" title={browserCurrentPath}>
              · {browserCurrentPath}
            </span>
          )}
        </h2>
        <div className="flex items-center gap-1">
          {!readOnlyMode && !isRoot && (
            <button
              onClick={goUp}
              className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-1 rounded hover:bg-slate-700/40 flex items-center gap-1"
              title="Up one folder"
            >
              <ChevronUp size={11} /> Up
            </button>
          )}
          <button
            onClick={() => refreshBrowser(readOnlyMode ? '' : browserCurrentPath)}
            className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-1 rounded hover:bg-slate-700/40 flex items-center gap-1"
            title="Refresh"
          >
            <RefreshCw size={10} />
          </button>
          {!readOnlyMode && (
            <>
              <button
                onClick={() => refreshBrowser('')}
                className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-1 rounded hover:bg-slate-700/40"
                title="Back to saved-folders root"
              >
                Root
              </button>
              <button
                onClick={() => setShowFolderManager(!showFolderManager)}
                className={`text-[10px] px-2 py-1 rounded flex items-center gap-1 ${
                  showFolderManager
                    ? 'bg-purple-500/20 text-purple-300'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700/40'
                }`}
                title="Manage saved cart folders"
              >
                <FolderPlus size={10} />
                Folders ({browserFolders.length})
              </button>
            </>
          )}
        </div>
      </div>

      {/* Folder manager — collapsible */}
      {showFolderManager && (
        <div className="px-4 py-3 border-b border-slate-700 bg-slate-900/40">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Saved cart folders</div>
          <div className="space-y-1.5">
            {browserFolders.map((f) => (
              <div key={f} className="flex items-center gap-2 text-xs font-mono">
                <Folder size={11} className="text-slate-500" />
                <span className="flex-1 truncate text-slate-400">{f}</span>
                <button
                  onClick={() => refreshBrowser(f)}
                  className="text-[10px] text-slate-500 hover:text-slate-300 px-1 py-0.5 rounded hover:bg-slate-700/40"
                  title="Browse this folder"
                >
                  Browse
                </button>
                <button
                  onClick={() => {
                    if (browserFolders.length <= 1) {
                      alert('Keep at least one saved folder.')
                      return
                    }
                    if (confirm(`Remove ${f} from saved folders?`)) removeBrowserFolder(f)
                  }}
                  className="text-rose-400/60 hover:text-rose-400 p-0.5 rounded"
                  title="Remove from saved folders"
                >
                  <Trash2 size={10} />
                </button>
              </div>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-2">
            <button
              onClick={async () => {
                await openFolderPicker(browserCurrentPath)
              }}
              className="text-xs px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 flex items-center gap-1"
            >
              <Plus size={11} /> Pick folder via picker…
            </button>
            <span className="text-[10px] text-slate-600 italic">
              (folder picker also creates the folder for browsing)
            </span>
          </div>
        </div>
      )}

      {/* Subdirs — hidden in read-only mode (drilling into them needs /carts?path
          which 403s on the public droplet). Operator can still organize via the
          saved-folders root list. */}
      {!readOnlyMode && browserSubdirs.length > 0 && (
        <div className="divide-y divide-slate-800/60">
          {browserSubdirs.map((d) => (
            <button
              key={d.path}
              onClick={() => refreshBrowser(d.path)}
              className="w-full text-left px-4 py-1.5 flex items-center gap-3 hover:bg-slate-800/40 text-sm"
            >
              <Folder size={13} className="text-amber-400/70 shrink-0" />
              <span className="flex-1 truncate text-slate-300">{d.name}</span>
              <ChevronRight size={11} className="text-slate-600" />
            </button>
          ))}
        </div>
      )}

      {/* Carts */}
      {browserCarts.length > 0 && (
        <div className="divide-y divide-slate-800/60">
          {browserCarts.map((c) => (
            <button
              key={c.path}
              onClick={() => onCartClick?.(c)}
              className="w-full text-left px-4 py-2 flex items-center gap-3 hover:bg-slate-800/40 text-sm group"
              title={`Open: ${c.path}`}
            >
              <Database size={13} className="text-purple-400 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-slate-200 truncate">{c.name}</div>
                <div className="text-[10px] text-slate-500 font-mono">
                  {c.passages} passages · {c.size_mb.toFixed(1)} MB · {c.modified}
                </div>
              </div>
              <span className="text-[10px] text-slate-600 group-hover:text-purple-400 transition-colors">Open →</span>
            </button>
          ))}
        </div>
      )}

      {/* Ingestable docs (only shown when browsing into a non-root folder) */}
      {browserDocs.length > 0 && (
        <div className="border-t border-slate-700/60">
          <div className="px-4 py-1.5 text-[10px] uppercase tracking-wider text-slate-500 bg-slate-900/40 flex items-center gap-2">
            <FileText size={10} />
            Ingestable documents in this folder
          </div>
          <div className="divide-y divide-slate-800/60 max-h-48 overflow-y-auto">
            {browserDocs.map((d) => (
              <button
                key={d.path}
                onClick={() => {
                  if (onDocClick) onDocClick(d)
                  else ingestPath(d.path)
                }}
                className="w-full text-left px-4 py-1.5 flex items-center gap-3 hover:bg-slate-800/40 text-xs"
              >
                <FileText size={11} className="text-slate-500 shrink-0" />
                <span className="flex-1 truncate text-slate-400 font-mono">{d.name}</span>
                <span className="text-slate-600 uppercase">{d.type}</span>
                <span className="text-slate-600 font-mono">{(d.size / 1024).toFixed(1)} KB</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {browserCarts.length === 0 && browserSubdirs.length === 0 && browserDocs.length === 0 && (
        <div className="p-6 text-center text-xs text-slate-600 italic">
          {isRoot ? 'No carts in saved folders.' : 'Empty folder.'}
        </div>
      )}
    </div>
  )
}
