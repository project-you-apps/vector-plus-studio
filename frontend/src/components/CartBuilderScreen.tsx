import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import {
  Hammer, Upload, FileText, Save, Loader2, Info, AlertCircle, Lock,
  Folder, Tag, User, RefreshCw,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import CartBrowser from './CartBrowser'
import FolderPickerModal from './FolderPickerModal'
import type { CartBuilderFile } from '../api/cartbuilder'

// MDEditor lazy-loaded so the ~315KB gzip cost is only paid when a user
// actually opens a metadata panel (which most won't on first visit).
const MDEditor = lazy(() => import('@uiw/react-md-editor'))

// BrowserCartBuilder lazy-loaded — pulls in transformers.js (~600KB gzip),
// pdfjs, mammoth, xlsx, npyjs, jszip, and the cart-builder-v2 pipeline.
// Users on the Search screen never pay this cost; only Cart Builder visitors
// trigger the ~1.5MB additional download.
const BrowserCartBuilder = lazy(() => import('./BrowserCartBuilder'))

// Heuristic: dragenter event has actual files in dataTransfer.types.
// Without this check, dragging text selections inside the page fires
// the overlay too. We only want the OS-level file drag.
function _hasFiles(dt: DataTransfer): boolean {
  return Array.from(dt.types).includes('Files')
}

// Cart Builder — full Phase 2 port. Drag-drop file ingestion → metadata
// editing → Pattern 0 preview → Build with sticky bottom progress bar.
//
// Layout (Andy 2026-05-05 IA decisions):
//   • CartBrowser cross-cuts: lives at the bottom of this screen AND in
//     Edit Carts. Same component, same store.
//   • "Open existing cart" → Cart Builder (this screen). Edit Carts is a
//     manual nav from the rail when the user wants passage-level edits.
//   • Build progress = sticky bottom bar (matches Edit Carts save bar).

export default function CartBuilderScreen() {
  const {
    files, uploading,
    pattern0, build, buildPolling,
    cartName, setCartName,
    uploadFiles, refreshFiles, refreshPattern0,
    startBuild, stopBuildPolling,
    clearWorkspace, refreshBrowser,
  } = useCartBuilderStore()

  const readOnlyMode = useAppStore((s) => s.status?.read_only_mode ?? false)

  const [dragOver, setDragOver] = useState(false)
  const [windowDrag, setWindowDrag] = useState(false)
  const dragCounter = useRef(0)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    refreshFiles()
    refreshPattern0()
    refreshBrowser()
    return () => stopBuildPolling()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Screen-wide drag detection — show full-screen overlay when the user
  // drags files anywhere onto Cart Builder (not just the inner drop zone).
  // dragCounter handles enter/leave on nested children correctly.
  useEffect(() => {
    const onDragEnter = (e: DragEvent) => {
      if (!e.dataTransfer || !_hasFiles(e.dataTransfer)) return
      e.preventDefault()
      dragCounter.current += 1
      setWindowDrag(true)
    }
    const onDragOver = (e: DragEvent) => {
      if (!e.dataTransfer || !_hasFiles(e.dataTransfer)) return
      e.preventDefault()  // required to allow the drop
    }
    const onDragLeave = (e: DragEvent) => {
      if (!e.dataTransfer || !_hasFiles(e.dataTransfer)) return
      dragCounter.current -= 1
      if (dragCounter.current <= 0) {
        dragCounter.current = 0
        setWindowDrag(false)
      }
    }
    const onDrop = (e: DragEvent) => {
      if (!e.dataTransfer || !_hasFiles(e.dataTransfer)) return
      e.preventDefault()
      dragCounter.current = 0
      setWindowDrag(false)
      handleFiles(e.dataTransfer.files)
    }
    window.addEventListener('dragenter', onDragEnter)
    window.addEventListener('dragover', onDragOver)
    window.addEventListener('dragleave', onDragLeave)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragenter', onDragEnter)
      window.removeEventListener('dragover', onDragOver)
      window.removeEventListener('dragleave', onDragLeave)
      window.removeEventListener('drop', onDrop)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-render Pattern 0 when cartName changes
  useEffect(() => {
    refreshPattern0()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cartName])

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    uploadFiles(Array.from(fileList))
  }

  const isBuilding = build.status === 'building' || buildPolling
  const buildPct = build.chunks_total
    ? Math.min(100, Math.round((build.chunks_done ?? 0) / build.chunks_total * 100))
    : Math.round((build.progress ?? 0) * 100)

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto pb-24">
      <div className="max-w-6xl mx-auto w-full space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
              <Hammer size={28} className="text-purple-300" />
              Cart Builder
            </h1>
            <p className="text-sm text-slate-500">
              Drag-and-drop documents to build a Membot brain cartridge.
            </p>
          </div>
          {files.length > 0 && !readOnlyMode && (
            <button
              onClick={() => {
                if (confirm('Clear workspace? Uploaded files will be removed.')) {
                  clearWorkspace()
                }
              }}
              className="text-xs text-slate-500 hover:text-slate-300 px-3 py-1.5 rounded-lg hover:bg-slate-800/40 transition-colors"
            >
              Clear workspace
            </button>
          )}
        </div>

        {/* Browser-side cart builder (Block 4 of WebGPU pivot, 2026-05-08).
            Self-contained: parses files, embeds via transformers.js (WebGPU
            with WASM fallback), packages an NPZ + manifest + permissions.
            Works in BOTH read-only and writable modes — privacy pitch is
            "your data never leaves your machine." Lazy-loaded so the
            transformers.js bundle only ships when a user opens this screen. */}
        <Suspense
          fallback={
            <div className="rounded-xl border border-slate-700 bg-slate-800/30 p-5 flex items-center gap-3 text-sm text-slate-400">
              <Loader2 size={16} className="animate-spin text-purple-300" />
              Loading browser-side cart builder…
            </div>
          }
        >
          <BrowserCartBuilder />
        </Suspense>

        {/* Read-only-mode banner — only shown when the server-side flow is
            disabled. Browser-side builder above stays functional regardless. */}
        {readOnlyMode && (
          <div className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 p-3 flex items-start gap-2 text-xs">
            <Lock size={14} className="text-cyan-400 flex-shrink-0 mt-0.5" />
            <div className="text-slate-400 leading-relaxed">
              <span className="text-cyan-200 font-medium">Server-side cart Builder is read-only on this demo.</span>{' '}
              The browser-side builder above runs end-to-end in your browser — use it to package your own carts.
              For full server-side workflow, install Vector+ Studio locally
              (<code className="font-mono text-slate-300 text-[11px]">git clone …/vector-plus-studio &amp;&amp; uvicorn api.main:app --port 8000</code>).
            </div>
          </div>
        )}

        {/* Upload errors surface as toasts now (bottom-right). The store still
            keeps `uploadError` for any inline retry UI we might add later. */}

        {/* Drop zone — hidden in read-only mode (uploads 403). */}
        {!readOnlyMode && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            handleFiles(e.dataTransfer.files)
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`rounded-xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer ${
            dragOver
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-slate-700 bg-slate-800/20 hover:border-slate-600'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.xlsx,.txt,.md,.rtf"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          {uploading ? (
            <Loader2 size={36} className="mx-auto mb-3 text-purple-400 animate-spin" />
          ) : (
            <Upload size={36} className={`mx-auto mb-3 ${dragOver ? 'text-purple-400' : 'text-slate-600'}`} />
          )}
          <div className="text-sm font-medium text-slate-300 mb-1">
            {uploading ? 'Uploading…' : 'Drop documents here or click to browse'}
          </div>
          <div className="text-xs text-slate-500">
            PDF, DOCX, XLSX, TXT, MD, RTF
          </div>
        </div>
        )}

        {/* Workspace file list — hidden in read-only mode (no uploads possible). */}
        {files.length > 0 && !readOnlyMode && (
          <div className="rounded-lg border border-slate-700 bg-slate-800/30">
            <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between">
              <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
                <FileText size={12} />
                Workspace ({files.length} {files.length === 1 ? 'file' : 'files'})
              </h2>
              <button
                onClick={refreshFiles}
                className="text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-1"
                title="Refresh from server"
              >
                <RefreshCw size={10} />
                Refresh
              </button>
            </div>
            <div className="divide-y divide-slate-800">
              {files.map((f) => <FileCard key={f.id} file={f} />)}
            </div>
          </div>
        )}

        {/* Pattern 0 preview + cart name — hidden in read-only mode (Pattern 0
            and build are both write-side; nothing to preview if you can't build). */}
        {!readOnlyMode && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4 space-y-3">
            <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Folder size={12} />
              Cart Settings
            </h2>
            <div>
              <label className="text-xs text-slate-500 mb-1 block">Cart name</label>
              <input
                type="text"
                value={cartName}
                onChange={(e) => setCartName(e.target.value)}
                placeholder="my-cart"
                className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
              />
              <div className="text-[10px] text-slate-600 mt-1 italic">
                alphanumeric, dashes, underscores only — backend sanitizes
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4 space-y-2">
            <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Info size={12} />
              Pattern 0 Preview
            </h2>
            {pattern0 ? (
              <div className="text-sm space-y-1.5">
                <div className="flex justify-between"><span className="text-slate-500">Cart:</span><span className="font-mono text-slate-300">{pattern0.cart_name}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Files:</span><span className="font-mono text-slate-300">{pattern0.file_count}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Total chunks:</span><span className="font-mono text-purple-300 font-semibold">{pattern0.total_chunks}</span></div>
                <div className="flex justify-between"><span className="text-slate-500">Created:</span><span className="font-mono text-slate-400 text-xs">{pattern0.created}</span></div>
              </div>
            ) : (
              <div className="text-xs text-slate-600 italic">
                Pattern 0 unavailable (backend cart-builder modules not loaded on this server).
              </div>
            )}
          </div>
        </div>
        )}

        {/* Cart browser embedded — also lives in Edit Carts.
            In read-only mode the click handler shows a toast instead of trying
            to call /load_cart (which 403s). The user still gets to BROWSE the
            catalog; they just can't open carts into the Cart Builder workspace. */}
        <CartBrowser
          headerLabel="My Carts"
          onCartClick={(cart) => {
            if (readOnlyMode) {
              useCartBuilderStore.getState().pushToast(
                'info',
                'Cart Builder is read-only on the public demo. To search this cart, mount it from the Search screen.',
                6000,
              )
              return
            }
            // Q1 = (c): one button → Cart Builder. Open loads the cart's
            // contents into THIS workspace for re-editing. To passage-edit,
            // user navigates manually to Edit Carts.
            useCartBuilderStore.getState().loadCart(cart.path)
          }}
        />
      </div>

      <FolderPickerModal />

      {/* Full-screen drag-drop overlay — fires when files are dragged anywhere
          on the Cart Builder screen, not just over the inner drop zone. The
          window-level listeners in useEffect drive `windowDrag`. */}
      {windowDrag && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-purple-500/15 backdrop-blur-sm border-4 border-dashed border-purple-400 pointer-events-none animate-dragpulse">
          <div className="rounded-2xl bg-slate-900/90 px-8 py-6 border border-purple-400/60 shadow-2xl flex flex-col items-center gap-3">
            <Upload size={48} className="text-purple-300" />
            <div className="text-lg font-medium text-purple-200">Drop to upload</div>
            <div className="text-xs text-slate-400">PDF, DOCX, XLSX, TXT, MD, RTF</div>
          </div>
        </div>
      )}

      {/* Sticky build bar — hidden in read-only mode (build endpoint 403s). */}
      {!readOnlyMode && (
      <div className="fixed bottom-0 left-0 right-0 lg:left-48 px-6 py-3 border-t border-slate-700 bg-slate-900/95 backdrop-blur flex items-center gap-4 z-40">
        <div className="flex-1 min-w-0">
          {isBuilding ? (
            <div className="flex items-center gap-3">
              <Loader2 size={16} className="text-amber-400 animate-spin shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] uppercase tracking-wider text-amber-400 font-semibold">Building</span>
                  <span className="text-xs text-slate-400 font-mono truncate">
                    {build.chunks_done ?? 0} / {build.chunks_total ?? 0} chunks · {build.message || build.status}
                  </span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full gradient-bg transition-all duration-500"
                    style={{ width: `${buildPct}%` }}
                  />
                </div>
              </div>
              <span className="text-xs text-amber-300 font-mono shrink-0">{buildPct}%</span>
            </div>
          ) : build.status === 'done' && build.cart_path ? (
            <div className="flex items-center gap-2 text-sm text-green-400">
              <Save size={14} />
              <span>Built: <code className="font-mono text-xs text-slate-300">{build.cart_path}</code></span>
            </div>
          ) : build.status === 'error' ? (
            <div className="flex items-center gap-2 text-sm text-rose-400">
              <AlertCircle size={14} />
              <span>Build failed: {build.error || 'unknown error'}</span>
            </div>
          ) : (
            <span className="text-xs text-slate-500">
              {files.length === 0 ? 'Add files to start building' : `${files.length} files queued · ready to build`}
            </span>
          )}
        </div>

        <button
          onClick={startBuild}
          disabled={isBuilding || files.length === 0}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            !isBuilding && files.length > 0
              ? 'bg-purple-500/30 border border-purple-500/50 text-purple-200 hover:bg-purple-500/40'
              : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
          }`}
        >
          <Hammer size={14} />
          {isBuilding ? 'Building…' : 'Build Cart'}
        </button>
      </div>
      )}
    </main>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// FileCard — per-file workspace card with inline metadata editor.
// ─────────────────────────────────────────────────────────────────────────────

function FileCard({ file }: { file: CartBuilderFile }) {
  const setMetadata = useCartBuilderStore((s) => s.setMetadata)
  const [editing, setEditing] = useState(false)
  const [owner, setOwner] = useState(file.owner)
  const [description, setDescription] = useState(file.description)
  const [tagsText, setTagsText] = useState((file.tags || []).join(', '))

  const saveMeta = () => {
    const tags = tagsText.split(',').map(t => t.trim()).filter(Boolean)
    setMetadata(file.id, { owner, description, tags })
    setEditing(false)
  }

  return (
    <div className="px-4 py-3">
      <div className="flex items-center gap-3">
        <FileText size={14} className="text-slate-500 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-sm text-slate-200 font-medium truncate">{file.name}</div>
          <div className="text-[11px] text-slate-500 font-mono">
            {file.type.toUpperCase()} · {file.chunks} chunks · {(file.size / 1024).toFixed(1)} KB · {file.chars.toLocaleString()} chars
          </div>
        </div>
        <button
          onClick={() => setEditing(!editing)}
          className="text-[10px] uppercase tracking-wider text-slate-500 hover:text-slate-300 px-2 py-1 rounded hover:bg-slate-700/40"
        >
          {editing ? 'Close' : 'Metadata'}
        </button>
      </div>

      {file.preview && !editing && (
        <div className="mt-2 ml-6 text-[11px] text-slate-500 italic line-clamp-2">{file.preview}</div>
      )}

      {(file.owner || file.description || (file.tags && file.tags.length > 0)) && !editing && (
        <div className="mt-2 ml-6 flex flex-wrap items-center gap-3 text-[11px] text-slate-500">
          {file.owner && <span className="flex items-center gap-1"><User size={10} /> {file.owner}</span>}
          {file.description && <span className="flex items-center gap-1 truncate max-w-md"><Info size={10} /> {file.description}</span>}
          {(file.tags || []).map(t => (
            <span key={t} className="px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400 text-[10px] flex items-center gap-1">
              <Tag size={9} />{t}
            </span>
          ))}
        </div>
      )}

      {editing && (
        <div className="mt-3 ml-6 space-y-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">Owner</label>
              <input
                value={owner}
                onChange={(e) => setOwner(e.target.value)}
                placeholder="who owns this doc"
                className="w-full rounded bg-slate-950/60 border border-slate-800 px-2 py-1 text-xs text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">Tags (comma-sep)</label>
              <input
                value={tagsText}
                onChange={(e) => setTagsText(e.target.value)}
                placeholder="tag1, tag2"
                className="w-full rounded bg-slate-950/60 border border-slate-800 px-2 py-1 text-xs text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
              />
            </div>
          </div>
          <div data-color-mode="dark">
            <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
              Description
              <span className="ml-2 text-slate-600 normal-case">· markdown supported</span>
            </label>
            <Suspense fallback={
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="short note about this doc"
                rows={4}
                className="w-full rounded bg-slate-950/60 border border-slate-800 px-2 py-1 text-xs text-slate-200 font-mono resize-none focus:outline-none focus:border-purple-500/60"
              />
            }>
              <MDEditor
                value={description}
                onChange={(v) => setDescription(v ?? '')}
                height={150}
                preview="edit"
                visibleDragbar={false}
                textareaProps={{ placeholder: 'short note about this doc — **markdown** _works_' }}
              />
            </Suspense>
          </div>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => { setEditing(false); setOwner(file.owner); setDescription(file.description); setTagsText((file.tags || []).join(', ')) }}
              className="px-2 py-1 text-[11px] text-slate-500 hover:text-slate-300"
            >
              Cancel
            </button>
            <button
              onClick={saveMeta}
              className="px-3 py-1 text-[11px] rounded bg-purple-500/20 border border-purple-500/40 text-purple-300 hover:bg-purple-500/30"
            >
              Save
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
