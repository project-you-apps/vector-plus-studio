import { useEffect, useRef, useState } from 'react'
import { Database, ChevronDown, FolderOpen, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import * as api from '../api/client'

// Top-of-screen toolbar for the Search screen. Owns the cart picker that
// previously lived in the Sidebar's top section. This is the per-screen
// toolbar pattern from the 2026-05-03 sidebar reorg — controls that operate
// on the *current screen's* state belong here, not in the global nav rail.
export default function SearchToolbar() {
  const { cartridges, status, mounting, fetchCartridges, mount, unmount } = useAppStore()
  const [open, setOpen] = useState(false)
  const [pathOpen, setPathOpen] = useState(false)
  const [pathInput, setPathInput] = useState('')
  const [pathLoading, setPathLoading] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchCartridges()
  }, [fetchCartridges])

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

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
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border border-slate-700 bg-slate-800/40 hover:bg-slate-800/70 transition-colors min-w-[240px]"
          title={mounted ? `Mounted: ${mounted}` : 'Click to mount a cartridge'}
        >
          <Database size={14} className={mounted ? 'text-purple-400' : 'text-slate-500'} />
          <span className={`flex-1 text-left truncate ${mounted ? 'text-slate-200 font-medium' : 'text-slate-500 italic'}`}>
            {mounted
              ? mountedCart
                ? `${mounted} (${mountedCart.size_mb.toFixed(1)} MB)`
                : mounted
              : 'Mount a cartridge…'}
          </span>
          <ChevronDown size={14} className={`text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
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

      {/* Spacer for future toolbar additions (search mode picker, filters, etc.) */}
      <div className="flex-1" />
    </div>
  )
}
