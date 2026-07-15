import { useEffect, useRef, useState } from 'react'
import { BookOpen, ChevronDown, Database, FolderOpen, Loader2, Trash2, Upload, Zap, Footprints, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useCartBuilderStore } from '../store/cartBuilderStore'
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
    webgpuStatus, webgpuBrainLoading, webgpuBrainProgress,
    walkTrail, clearWalk, restoreTrailStep,
    localCarts, activeLocalCart, localCartLoading,
    mountLocalCart, unmountLocalCart, selectLocalCart,
    showTocPanel, setShowTocPanel,
    pattern0DrillActive, triggerPattern0BackToTop,
  } = useAppStore()
  const localFileInputRef = useRef<HTMLInputElement>(null)
  const [walkTrailOpen, setWalkTrailOpen] = useState(false)
  const walkTrailRef = useRef<HTMLDivElement>(null)
  const currentWalkStep = walkTrail.length > 1 ? walkTrail[walkTrail.length - 1] : null

  // WebGPU Associate gating (Phase 2e). When the server has no CUDA but the
  // browser has WebGPU and the mounted cart has a brain on disk, Associate
  // is available via browser-side physics.
  const mountedName = status?.mounted_cartridge ?? null
  const mountedCartHasBrain = !!cartridges.find((c) => c.name === mountedName)?.has_brain
  const webgpuAssociateAvailable =
    webgpuStatus === 'available' && !status?.gpu_available && !!mountedName && mountedCartHasBrain

  const [open, setOpen] = useState(false)
  const [modeOpen, setModeOpen] = useState(false)
  const [pathOpen, setPathOpen] = useState(false)
  const [pathInput, setPathInput] = useState('')
  const [pathLoading, setPathLoading] = useState(false)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [ejecting, setEjecting] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const modeRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchCartridges()
  }, [fetchCartridges])

  // Close dropdowns when clicking outside (one handler for both)
  useEffect(() => {
    if (!open && !modeOpen && !walkTrailOpen) return
    const handler = (e: MouseEvent) => {
      const target = e.target as Node
      if (open && ref.current && !ref.current.contains(target)) setOpen(false)
      if (modeOpen && modeRef.current && !modeRef.current.contains(target)) setModeOpen(false)
      if (walkTrailOpen && walkTrailRef.current && !walkTrailRef.current.contains(target)) setWalkTrailOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open, modeOpen, walkTrailOpen])

  const currentMode = MODES.find((m) => m.key === searchMode) ?? MODES[0]

  const mounted = status?.mounted_cartridge ?? null
  const mountedCart = mounted ? cartridges.find((c) => c.name === mounted) : null
  const isSandboxed = !!status?.mounted_is_sandboxed
  const sandboxPath = status?.mounted_path ?? null

  const pushToast = useCartBuilderStore((s) => s.pushToast)

  const handleEject = async () => {
    if (!sandboxPath || ejecting) return
    if (!confirm('Delete this uploaded cart from the sandbox now? Unmounts and erases the file immediately.')) {
      return
    }
    setEjecting(true)
    try {
      // Unmount first — eject endpoint refuses if currently mounted.
      if (mounted) {
        await unmount()
      }
      const res = await api.ejectCartridge(sandboxPath)
      if (res.success) {
        pushToast('success', 'Cart ejected — file deleted from sandbox.', 4000)
        fetchCartridges()
        useAppStore.getState().fetchStatus()
      }
    } catch (err) {
      pushToast('error', `Eject failed: ${err instanceof Error ? err.message : 'unknown'}`, 6000)
    } finally {
      setEjecting(false)
    }
  }

  // handleBrowse used to invoke the backend's PowerShell file dialog; removed
  // when the browser-side "Open from My Computer" replaced it.

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

  // Client-side upload — works in both local and read-only-mode (droplet) deploys.
  // The backend writes to a sandbox dir with TTL eviction (1h default), forces a
  // read-only permissions sidecar, then we mount via the standard path.
  const handleUpload = async (file: File) => {
    setUploadLoading(true)
    const pushToast = useCartBuilderStore.getState().pushToast
    try {
      const resp = await api.uploadCartridge(file)
      pushToast('info', `Uploaded ${resp.size_mb} MB — mounting…`, 3000)
      const mres = await api.mountCartridge(resp.cart_path)
      if (!mres.success) {
        pushToast('error', `Mount failed: ${mres.message}`, 8000)
        return
      }
      fetchCartridges()
      useAppStore.getState().fetchStatus()
      // Andy 2026-07-06 AM: fire per-pattern-meta fetch on sandbox upload
      // so images render. This wasn't happening because handleUpload calls
      // api.mountCartridge directly instead of the store's mount() action.
      useAppStore.getState().fetchSandboxPerPatternMeta()
      pushToast('success', `Mounted ${mres.name} (${mres.pattern_count} patterns) — sandbox cart, expires in ${Math.round(resp.ttl_sec / 60)} min`, 6000)
      setOpen(false)
    } catch (err) {
      pushToast('error', `Upload failed: ${err instanceof Error ? err.message : 'unknown'}`, 8000)
    } finally {
      setUploadLoading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
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
          {activeLocalCart ? (
            <FolderOpen size={14} className="text-cyan-400" />
          ) : (
            <Database size={14} className={mounted ? 'text-purple-400' : 'text-slate-500'} />
          )}
          <span className="flex-1 text-left truncate">
            {activeLocalCart ? (
              <>
                <span className="text-cyan-500 mr-1.5">Local:</span>
                <span className="text-slate-100 font-medium">{activeLocalCart}</span>
                {(() => {
                  const lc = localCarts.get(activeLocalCart)
                  return lc ? <span className="text-slate-500 ml-1.5 text-xs">({(lc.sizeBytes / 1048576).toFixed(1)} MB)</span> : null
                })()}
              </>
            ) : mounted ? (
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
            {/* The old backend-PowerShell-dialog "Open Cartridge…" button was
                removed 2026-06-02. The browser-side "Open from My Computer…"
                below is strictly better: works on local dev AND droplet, never
                round-trips to backend, never hangs on a server dialog.
                Paste-path is still available below for power users. */}

            {/* Paste path fallback — also hidden in read-only mode. */}
            {!status?.read_only_mode && (
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
            )}

            {/* Upload Cartridge — works in both local and droplet deploys.
                Backend sandboxes the upload (1h TTL) and forces r-only perms.
                On the droplet this is the ONLY way to bring your own data. */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadLoading}
              className="w-full flex items-center gap-2 px-3 py-2.5 text-sm border-b border-slate-800 hover:bg-slate-800/50 disabled:opacity-50 transition-colors"
              title="Upload a .cart.npz from your machine. Sandboxed, read-only, evicted after 1 hour."
            >
              {uploadLoading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
              <span className="font-medium">{uploadLoading ? 'Uploading…' : 'Upload Cartridge…'}</span>
              <span className="ml-auto text-[10px] text-slate-500">
                {status?.read_only_mode ? 'sandbox · 1h TTL' : 'sandbox'}
              </span>
            </button>
            {status?.read_only_mode && (
              <div className="px-3 py-1.5 border-b border-slate-800 text-[10px] text-slate-500 italic flex items-center gap-2">
                <FolderOpen size={10} className="text-slate-600 flex-shrink-0" />
                Public demo — uploads are sandboxed and read-only. Or pick from the list below.
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".npz,.cart.npz"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleUpload(f)
              }}
            />

            {/* Open from My Computer — F1 client-side. Browser parses the
                cart, nothing uploads. Works in both local dev and on the
                public droplet. Currently cosine-only (no brain/sigs locally). */}
            <button
              onClick={() => localFileInputRef.current?.click()}
              disabled={localCartLoading}
              className="w-full flex items-center gap-2 px-3 py-2.5 text-sm border-b border-slate-800 hover:bg-slate-800/50 disabled:opacity-50 transition-colors"
              title="Open a .cart.npz from your own disk. Parsed in your browser — never uploaded. Read-only, cosine search only (no Associate without brain)."
            >
              {localCartLoading ? <Loader2 size={14} className="animate-spin" /> : <FolderOpen size={14} className="text-cyan-400" />}
              <span className="font-medium">{localCartLoading ? 'Parsing…' : 'Open from My Computer…'}</span>
              <span className="ml-auto text-[10px] text-cyan-500">never uploaded</span>
            </button>
            <input
              ref={localFileInputRef}
              type="file"
              accept=".npz,.cart.npz"
              className="hidden"
              onChange={async (e) => {
                const f = e.target.files?.[0]
                if (!f) return
                const res = await mountLocalCart(f)
                if (!res.success) alert(res.message)
                setOpen(false)
                // Reset the input so re-picking the same file fires onChange again.
                if (localFileInputRef.current) localFileInputRef.current.value = ''
              }}
            />

            {/* Local carts list — anything the user has opened from disk
                this session. Distinct from "Available" (server-side carts). */}
            {localCarts.size > 0 && (
              <div className="p-2 space-y-1 border-b border-slate-800">
                <div className="px-1 py-1 text-[10px] uppercase tracking-wider text-cyan-600">
                  On Your Computer ({localCarts.size})
                </div>
                {Array.from(localCarts.values()).map((c) => {
                  const isActive = activeLocalCart === c.name
                  return (
                    <div
                      key={c.name}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-all cursor-pointer ${
                        isActive
                          ? 'bg-cyan-500/10 border border-cyan-500/30'
                          : 'border border-transparent hover:bg-slate-800/40'
                      }`}
                      onClick={() => { selectLocalCart(c.name); setOpen(false) }}
                    >
                      <FolderOpen size={12} className={isActive ? 'text-cyan-400' : 'text-slate-500'} />
                      <span className="flex-1 truncate text-slate-200">{c.name}</span>
                      <span className="text-[10px] text-slate-500">{(c.sizeBytes / 1048576).toFixed(1)} MB</span>
                      {isActive && (
                        <button
                          onClick={async (e) => {
                            // Fix #13 (2026-06-30) — X was already wired to
                            // unmountLocalCart, but defensively also drop any
                            // backend mount so the pill in Header resets and
                            // every tab sees a clean "nothing mounted" state.
                            // Same shared unmount path as the Header pill (Fix #1).
                            e.stopPropagation()
                            unmountLocalCart()
                            if (status?.mounted_cartridge) await unmount()
                            setOpen(false)
                          }}
                          className="p-0.5 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white"
                          title="Unmount this local cart"
                        >
                          <X size={12} />
                        </button>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

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

      {/* Eject button — only shown when the mounted cart is from the upload
          sandbox. Privacy/control feature: lets users delete their uploaded
          cart immediately instead of waiting for the 1h TTL. */}
      {isSandboxed && sandboxPath && (
        <button
          onClick={handleEject}
          disabled={ejecting}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs border border-rose-500/40 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20 disabled:opacity-50 transition-colors"
          title="Immediately delete this uploaded cart from the sandbox (otherwise auto-deletes after 1 hour)"
        >
          {ejecting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
          <span className="hidden md:inline">{ejecting ? 'Ejecting…' : 'Eject upload'}</span>
        </button>
      )}

      {/* Status pill — mounted cart pattern count */}
      {status?.mounted_cartridge && (
        <span className="text-xs text-slate-500 font-mono">
          {status.pattern_count.toLocaleString()} patterns
        </span>
      )}

      {/* Pattern-0 button — return the Search tab to the TOC view after a
          search. Visible only when a cart is mounted (server or local). When
          the TOC is already visible AND no per-file drill is open, dims to
          signal "already here." When the TOC is visible but the user has
          drilled into a file, stays live and acts as back-to-top (collapses
          the drill via a store signal that Pattern0TocPanel watches). */}
      {(status?.mounted_cartridge || activeLocalCart) && (() => {
        const atTop = showTocPanel && !pattern0DrillActive
        return (
          <button
            onClick={() => {
              setShowTocPanel(true)
              triggerPattern0BackToTop()
            }}
            disabled={atTop}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm border transition-colors ${
              atTop
                ? 'border-slate-800 bg-slate-800/20 text-slate-500 cursor-default'
                : 'border-slate-700 bg-slate-800/40 hover:bg-slate-800/70 hover:border-purple-500/40 text-slate-200'
            }`}
            title={
              atTop
                ? 'Pattern-0 TOC is already visible'
                : pattern0DrillActive
                  ? 'Back to the top of the Pattern-0 table of contents'
                  : 'Return to the Pattern-0 table of contents for this cart'
            }
          >
            <BookOpen size={14} className={atTop ? 'text-slate-500' : 'text-purple-400'} />
            <span className="font-medium">Pattern-0</span>
          </button>
        )
      })()}

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
              // Associate has a second path: browser-side WebGPU when the
              // server lacks CUDA. Smart Search still needs server CUDA
              // because the blend ranges are server-only for now.
              const webgpuViable = m.key === 'associate' && webgpuAssociateAvailable
              const isDisabled =
                (needsFullBrain && !status?.gpu_available && !webgpuViable) ||
                (needsFullBrain && status?.gpu_available && (!status?.physics_trained || stillTraining)) ||
                (needsSigs && !status?.signatures_loaded)
              const isTraining =
                (needsFullBrain && status?.gpu_available && stillTraining) ||
                (needsSigs && !status?.signatures_loaded && stillTraining)
              const isReady =
                (needsFullBrain && status?.physics_trained && !stillTraining) ||
                (needsSigs && status?.signatures_loaded) ||
                webgpuViable
              let subtitle = m.desc
              if (webgpuViable) subtitle = 'Browser GPU (WebGPU)'
              else if (needsFullBrain && !status?.gpu_available) subtitle = 'Requires GPU'
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

      {currentWalkStep && (
        <div className="relative" ref={walkTrailRef}>
          <div className="flex items-center gap-2 px-3 py-1 rounded-lg border border-cyan-500/40 bg-cyan-500/5 text-xs">
            <button
              onClick={() => setWalkTrailOpen(!walkTrailOpen)}
              className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
              title="Show walk trail"
            >
              <Footprints size={12} className="text-cyan-400 shrink-0" />
              <span className="text-cyan-200 uppercase tracking-wider text-[10px]">Walk · {walkTrail.length - 1} step{walkTrail.length === 2 ? '' : 's'}</span>
              <span className="text-cyan-100 truncate max-w-[280px]" title={currentWalkStep.label}>
                {currentWalkStep.label}
              </span>
              <ChevronDown size={11} className={`text-cyan-300 transition-transform ${walkTrailOpen ? 'rotate-180' : ''}`} />
            </button>
            <button
              onClick={() => { clearWalk(); setWalkTrailOpen(false) }}
              className="p-0.5 rounded hover:bg-slate-700/50 text-cyan-300 hover:text-white transition-colors"
              title="Clear walk and return to the original query"
            >
              <X size={12} />
            </button>
          </div>

          {walkTrailOpen && (
            <div className="absolute top-full mt-1 left-0 min-w-[320px] max-w-[480px] max-h-[60vh] overflow-y-auto rounded-lg border border-slate-700 bg-[var(--chrome-bg)] shadow-2xl z-30 p-2">
              <div className="px-1 py-1 text-[10px] uppercase tracking-wider text-slate-500 sticky top-0 bg-[var(--chrome-bg)]">Walk trail · click to jump back</div>
              {walkTrail.map((step, i) => {
                const isCurrent = i === walkTrail.length - 1
                return (
                  <button
                    key={i}
                    onClick={() => { restoreTrailStep(i); setWalkTrailOpen(false) }}
                    className={`w-full text-left px-2.5 py-1.5 rounded text-xs transition-all ${
                      isCurrent
                        ? 'bg-cyan-500/15 text-cyan-200 border border-cyan-500/30'
                        : 'text-slate-300 hover:bg-slate-800/50 border border-transparent cursor-pointer'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-500 font-mono shrink-0">{i}.</span>
                      <span className="text-[10px] uppercase tracking-wider text-slate-500 shrink-0">
                        {step.kind === 'query' ? 'Query' : 'Walk'}
                      </span>
                      <span className="truncate">{step.label || '(empty)'}</span>
                      {isCurrent && <span className="text-[10px] text-cyan-400 shrink-0 ml-auto">here</span>}
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      )}

      {webgpuBrainLoading && webgpuBrainProgress && (
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg border border-cyan-500/40 bg-cyan-500/5 text-xs">
          <Loader2 size={12} className="text-cyan-400 animate-spin shrink-0" />
          <span className="text-cyan-200 uppercase tracking-wider text-[10px]">{webgpuBrainProgress.stage}</span>
          {webgpuBrainProgress.total > 0 && (
            <>
              <div className="h-1 w-24 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
                  style={{ width: `${(webgpuBrainProgress.loaded / webgpuBrainProgress.total) * 100}%` }}
                />
              </div>
              <span className="text-cyan-300 font-mono shrink-0">
                {(webgpuBrainProgress.loaded / 1048576).toFixed(0)} / {(webgpuBrainProgress.total / 1048576).toFixed(0)} MB
              </span>
            </>
          )}
        </div>
      )}

      <div className="flex-1" />
    </div>
  )
}
