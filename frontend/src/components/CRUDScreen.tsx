import { useEffect, useRef, useState } from 'react'
import {
  Pencil, Plus, Save, Trash2, RotateCcw, Lock, Unlock, FilePlus2,
  FolderOpen, Folder, Database, AlertCircle, CheckCircle2, Loader2,
  Hammer, ChevronRight, X, List, ChevronLeft, Search,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import CartBrowser from './CartBrowser'
import ConfirmDialog, { type ConfirmState } from './ConfirmDialog'
import * as api from '../api/client'
import type { PatternListItem } from '../api/client'
import {
  buildCartFromPassages,
  chunkSections,
  parseFile,
  type BuiltCart,
  type PipelineProgress,
} from '../cart-builder-v2'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import * as cb from '../api/cartbuilder'

// CRUDScreen — first mockup for Andy to react to.
//
// Architecture (Andy 2026-05-04 spec, mockup 2026-05-05):
//   • Two MODES at the top:
//       1. "Open Cart"  — operate on an existing mounted cart (rwx)
//       2. "New Cart"   — start with an empty cart and add passages
//   • Three OPS panels: Add / Update / Delete
//   • Tombstoned-passages panel with one-click Restore
//   • Activity log of recent ops
//
// NOT WIRED YET. This is a UI scaffold to align on layout before plumbing.
// Each interactive button below has either:
//   (a) A real handler that calls the existing store action (Add, Delete,
//       Restore, Lock/Unlock), or
//   (b) An inline TODO comment for backend gaps Andy needs to confirm:
//        - "New empty cart" route (no /api/cartridges/new yet)
//        - True in-place Update (PUT /api/patterns/{idx}) — current
//          appStore saveEditor() simulates Update via add-new + tombstone-old
//
// Future polish (after Andy approves the structure):
//   • Replace plain <textarea> with react-mde markdown editor
//   • Wire "Open from existing cart" picker to the same picker as Search
//   • Add diff-view for Update preview
//   • Activity log persistence (currently in-component only)

type Mode = 'open' | 'new'
// Activity log kinds. Andy 2026-05-06: mount / unmount / save / create
// belong in the log too — any cart-state-changing action gets a row.
// 'open' is reserved for screens like Cart Builder where loading a cart
// into a workspace (load_cart) isn't the same as mounting it for search.
type OpKind = 'add' | 'update' | 'delete' | 'restore' | 'mount' | 'unmount' | 'save' | 'create' | 'open'

interface ActivityEntry {
  ts: string
  kind: OpKind
  detail: string
  ok: boolean
}

export default function CRUDScreen() {
  const status = useAppStore((s) => s.status)
  const deletedPatterns = useAppStore((s) => s.deletedPatterns)
  const fetchCartridges = useAppStore((s) => s.fetchCartridges)
  const fetchStatus = useAppStore((s) => s.fetchStatus)
  const fetchDeleted = useAppStore((s) => s.fetchDeleted)
  const unmount = useAppStore((s) => s.unmount)
  const toggleLock = useAppStore((s) => s.toggleLock)
  const saveCartridge = useAppStore((s) => s.saveCartridge)
  const addPassage = useAppStore((s) => s.addPassage)
  const deleteResult = useAppStore((s) => s.deleteResult)
  const restoreResult = useAppStore((s) => s.restoreResult)

  // LocalCart edit support (Andy 2026-06-16 PM): when a browser-mounted cart
  // is active and the backend has no cart mounted, Edit Carts operates on the
  // LocalCart in-memory state. Operations route through new store actions
  // (localCartTombstone, localCartTombstoneBySource, localCartRestore,
  // localCartSave) instead of the backend API.
  const activeLocalCartName = useAppStore((s) => s.activeLocalCart)
  const localCarts = useAppStore((s) => s.localCarts)
  const unmountLocalCart = useAppStore((s) => s.unmountLocalCart)
  const localCartTombstone = useAppStore((s) => s.localCartTombstone)
  const localCartRestore = useAppStore((s) => s.localCartRestore)
  const localCartTombstoneBySource = useAppStore((s) => s.localCartTombstoneBySource)
  const localCartListSources = useAppStore((s) => s.localCartListSources)
  const localCartSave = useAppStore((s) => s.localCartSave)

  const activeLocalCart = activeLocalCartName ? localCarts.get(activeLocalCartName) ?? null : null
  const isLocalMount = !!activeLocalCart && !status?.mounted_cartridge
  const isBackendMount = !!status?.mounted_cartridge

  const [mode, setMode] = useState<Mode>('open')
  const [activity, setActivity] = useState<ActivityEntry[]>([])
  const [confirm, setConfirm] = useState<ConfirmState | null>(null)

  // Add-panel state. For LocalCart, addSource is the source label to attach
  // to the new passage (becomes its sourcePath entry). Defaults to '<inline>'
  // if user leaves blank.
  const [addText, setAddText] = useState('')
  const [addSource, setAddSource] = useState('')
  const localCartAddPassage = useAppStore((s) => s.localCartAddPassage)
  // File-picker Add (LocalCart only, Andy 2026-06-17 PM "and we're good to
  // go"). Pick files → parse via cart-builder-v2 → chunk → embed each chunk
  // → localCartAddPassage with source=file.name. Progress shown inline.
  const [addFilesProcessing, setAddFilesProcessing] = useState(false)
  const [addFilesStatus, setAddFilesStatus] = useState<string | null>(null)
  const addFilesInputRef = useRef<HTMLInputElement | null>(null)

  // Update-panel state
  const [updateIdx, setUpdateIdx] = useState('')
  const [updateText, setUpdateText] = useState('')
  const [updateLoading, setUpdateLoading] = useState(false)

  // Delete-panel state
  const [deleteIdx, setDeleteIdx] = useState('')
  // LocalCart delete-target mode (Andy 2026-06-17 PM): user picks whether the
  // typed IDX refers to a row in Source Files (delete a whole file) or a
  // passage IDX from search results (delete one passage). Defaults to 'file'
  // because file-level delete is the primary use case for user-built carts.
  const [deleteMode, setDeleteMode] = useState<'file' | 'passage'>('file')

  // New-cart state
  const [newCartName, setNewCartName] = useState('')

  useEffect(() => {
    fetchCartridges()
    fetchDeleted()
  }, [fetchCartridges, fetchDeleted])

  // Unified mount state: a cart is "mounted" if either the backend has one
  // (status.mounted_cartridge) OR a browser-side LocalCart is active. Backend
  // takes precedence for legacy compatibility. Read-only for local carts is
  // always false (the user owns the file they picked from their disk).
  // Dirty tracks unsaved tombstones/adds for local carts; backend tracks its
  // own dirty flag via status.dirty.
  const mounted = isBackendMount || isLocalMount
  const readOnly = isBackendMount ? !!status?.read_only : false
  const dirty = isBackendMount ? !!status?.dirty : (activeLocalCart?.dirty ?? false)
  const patternCount = isBackendMount
    ? (status?.pattern_count ?? 0)
    : (activeLocalCart ? activeLocalCart.passages.length - activeLocalCart.tombstones.size : 0)

  const log = (kind: OpKind, detail: string, ok: boolean) => {
    setActivity((prev) => [
      { ts: new Date().toLocaleTimeString(), kind, detail, ok },
      ...prev.slice(0, 19),
    ])
  }

  // ── Add ──
  // Routes through localCartAddPassage for LocalCart-mounted carts (embeds
  // via /api/embed, appends to in-memory state) or addPassage (backend API)
  // for backend-mounted carts.
  const handleAdd = async () => {
    const text = addText.trim()
    if (!text) return
    if (isLocalMount) {
      const source = addSource.trim() || '<inline>'
      const resp = await localCartAddPassage(text, source)
      log('add', resp.message, resp.success)
      if (resp.success) {
        setAddText('')
        setAddSource('')
      }
    } else {
      const resp = await addPassage(text)
      log('add', resp.message, resp.success)
      if (resp.success) {
        setAddText('')
        fetchStatus()
      }
    }
  }

  // ── Add files (LocalCart only) ──
  // Reuses cart-builder-v2's parseFile + chunkSections so the file→passage
  // pipeline matches Cart Builder's behavior exactly (300-word chunks, 50
  // overlap, same parsers for PDF/DOCX/XLSX/MD/HTML/RTF/TXT). Each chunk
  // becomes one passage with source = file.name so Source Files panel
  // groups them by their origin file.
  const handleAddFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    if (!isLocalMount) return
    setAddFilesProcessing(true)
    try {
      let totalAdded = 0
      let totalFailed = 0
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        setAddFilesStatus(`(${i + 1}/${files.length}) Parsing ${file.name}…`)
        try {
          const parsed = await parseFile(file)
          if (parsed.sections.length === 0) {
            log('add', `${file.name}: no text extracted`, false)
            totalFailed++
            continue
          }
          const chunks = chunkSections(parsed.sections)
          setAddFilesStatus(`(${i + 1}/${files.length}) Embedding ${chunks.length} chunk${chunks.length === 1 ? '' : 's'} from ${file.name}…`)
          let fileAdded = 0
          for (let c = 0; c < chunks.length; c++) {
            setAddFilesStatus(`(${i + 1}/${files.length}) ${file.name}: embedding ${c + 1}/${chunks.length}…`)
            const resp = await localCartAddPassage(chunks[c].text, file.name)
            if (resp.success) fileAdded++
          }
          log('add', `Added ${fileAdded}/${chunks.length} chunk${chunks.length === 1 ? '' : 's'} from ${file.name}`, fileAdded > 0)
          totalAdded += fileAdded
        } catch (e) {
          const msg = e instanceof Error ? e.message : 'Unknown parser error'
          log('add', `${file.name}: ${msg}`, false)
          totalFailed++
        }
      }
      setAddFilesStatus(
        totalFailed > 0
          ? `Done — added ${totalAdded} passages, ${totalFailed} file${totalFailed === 1 ? '' : 's'} failed`
          : `Done — added ${totalAdded} passages from ${files.length} file${files.length === 1 ? '' : 's'}`
      )
    } finally {
      setAddFilesProcessing(false)
      // Clear status after a few seconds so the panel returns to clean state
      setTimeout(() => setAddFilesStatus(null), 6000)
      // Reset the file input so picking the same file again re-fires onChange
      if (addFilesInputRef.current) addFilesInputRef.current.value = ''
    }
  }

  // ── Update (current backend doesn't have PUT /api/patterns/{idx}) ──
  // Simulated via add-new + tombstone-old, mirroring appStore.saveEditor()
  // behavior. Flag for Andy: should we add a real in-place update route?
  const handleUpdateLoad = async () => {
    const idx = parseInt(updateIdx, 10)
    if (isNaN(idx)) return
    setUpdateLoading(true)
    try {
      const apiBase = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'
      const res = await fetch(`${apiBase}/patterns/${idx}`)
      if (!res.ok) throw new Error(`Pattern #${idx} not found`)
      const p = await res.json()
      setUpdateText(p.full_text || '')
      log('update', `Loaded pattern #${idx} for editing`, true)
    } catch (e) {
      log('update', e instanceof Error ? e.message : 'Load failed', false)
    } finally {
      setUpdateLoading(false)
    }
  }

  const handleUpdateSave = () => {
    const idx = parseInt(updateIdx, 10)
    const text = updateText.trim()
    if (isNaN(idx) || !text) return
    setConfirm({
      title: `Update pattern #${idx}?`,
      body: (
        <>
          <p className="mb-2">
            This adds a NEW passage with your edited text and tombstones the original
            <code className="font-mono text-rose-300 mx-1">#{idx}</code>.
            (True in-place update — <code className="font-mono">PUT /api/patterns/{'{idx}'}</code> — is a TODO.)
          </p>
          <p className="text-xs text-slate-500">
            The new passage gets a new index assigned at the end of the cart.
            The old idx becomes a tombstone you can Restore from below.
          </p>
        </>
      ),
      confirmLabel: 'Update',
      destructive: true,
      onConfirm: async () => {
        const addResp = await addPassage(text)
        if (!addResp.success) {
          log('update', `Add-new failed: ${addResp.message}`, false)
          return
        }
        await deleteResult(idx)
        log('update', `Updated pattern #${idx} (new entry + tombstoned old)`, true)
        setUpdateText('')
        setUpdateIdx('')
      },
    })
  }

  // ── Delete ──
  // Standard practice (Andy 2026-05-06): destructive actions get a
  // confirmation modal. The modal does the actual call on confirm.
  // For LocalCart-mounted carts, routes through localCartTombstone for the
  // passage-idx variant, or resolves the file row to its sourcePath and
  // routes through handleDeleteSource for the file-idx variant.
  // Andy 2026-06-17 PM: the file/passage distinction is the user's call —
  // deleteMode toggle in the panel says which kind of IDX they typed.
  const handleDelete = () => {
    const idx = parseInt(deleteIdx, 10)
    if (isNaN(idx)) return

    // LocalCart + file mode: look up the file at row #idx in the sorted
    // Source Files list (same sort as the panel below so the visible row
    // numbers match) and tombstone every passage from that sourcePath.
    if (isLocalMount && deleteMode === 'file') {
      const sources = localCartListSources()
      const sorted = [...sources].sort((a, b) => {
        if (b.activeCount !== a.activeCount) return b.activeCount - a.activeCount
        return a.sourcePath.localeCompare(b.sourcePath)
      })
      const row = sorted[idx - 1]
      if (!row) {
        log('delete', `File #${idx} not found (only ${sorted.length} files)`, false)
        return
      }
      // Delegate to the same handler the Source Files row buttons use —
      // single confirm path, consistent UX.
      handleDeleteSource(row.sourcePath, row.activeCount)
      setDeleteIdx('')
      return
    }

    // Passage mode (LocalCart or backend) — standard per-passage tombstone.
    setConfirm({
      title: `Tombstone passage #${idx}?`,
      body: (
        <>
          <p className="mb-2">
            This marks passage <code className="font-mono text-rose-300">#{idx}</code> as deleted.
            It stops appearing in search results and can be restored from the tombstoned list below.
          </p>
          <p className="text-xs text-slate-500">
            {isLocalMount
              ? 'In-memory tombstone on your local cart. Click Save Cart to persist (re-downloads the modified .cart.npz to your disk).'
              : 'Permanent removal happens when you Save the cart (the on-disk data is overwritten). Until then, Restore brings it back.'}
          </p>
        </>
      ),
      confirmLabel: 'Tombstone',
      destructive: true,
      onConfirm: async () => {
        if (isLocalMount) {
          localCartTombstone(idx)
          log('delete', `Tombstoned local passage #${idx}`, true)
        } else {
          await deleteResult(idx)
          log('delete', `Tombstoned passage #${idx}`, true)
        }
        setDeleteIdx('')
      },
    })
  }

  // Delete an entire source file from a LocalCart — tombstones every passage
  // whose sourcePath matches. Demo step 6's "delete the specific single file"
  // flows through here. Confirmation modal explains how many passages will
  // disappear.
  const handleDeleteSource = (sourcePath: string, activeCount: number) => {
    if (!isLocalMount) return
    setConfirm({
      title: `Delete file "${sourcePath}"?`,
      body: (
        <>
          <p className="mb-2">
            This tombstones <strong className="text-rose-300">{activeCount}</strong> passages
            sourced from <code className="font-mono text-rose-300">{sourcePath}</code>.
          </p>
          <p className="text-xs text-slate-500">
            They'll disappear from search results immediately. Click Save Cart afterward to
            persist the change to a new .cart.npz on your disk.
          </p>
        </>
      ),
      confirmLabel: `Delete ${activeCount} passages`,
      destructive: true,
      onConfirm: async () => {
        const added = localCartTombstoneBySource(sourcePath)
        // Andy 2026-06-17 PM: include the actual passage indices that got
        // tombstoned so it's visible at a glance whether the file→idx
        // resolution is doing the right thing (vs the "is it tombstoning the
        // typed idx instead?" suspicion).
        const idxList = added.length === 0
          ? '(none — no matching sourcePath, possibly legacy cart)'
          : added.length <= 6
            ? `idx ${added.join(', ')}`
            : `idx ${added.slice(0, 6).join(', ')}…(+${added.length - 6} more)`
        log('delete', `Tombstoned ${added.length} from "${sourcePath}" — ${idxList}`, added.length > 0)
      },
    })
  }

  const handleRestore = async (idx: number) => {
    if (isLocalMount) {
      localCartRestore(idx)
      log('restore', `Restored local pattern #${idx}`, true)
    } else {
      await restoreResult(idx)
      log('restore', `Restored pattern #${idx}`, true)
    }
  }

  // New Cart is composed entirely in the browser via the cart-builder-v2
  // pipeline (parse skipped, chunker → embedder → writer). No backend route
  // needed for the build itself; user saves the resulting .cart.npz via the
  // File System Access API (or Downloads fallback) and mounts it from the
  // Search screen. See NewCartPanel below.

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full space-y-5">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
            <Pencil size={28} className="text-purple-300" />
            Edit Carts
          </h1>
          <p className="text-sm text-slate-500">
            Add, update, delete passages on the mounted cart. Tombstone first; hard-delete on save.
          </p>
        </div>

        {/* Mode tabs */}
        <div className="flex gap-1 bg-slate-800/40 rounded-lg p-1 w-fit">
          <ModeTab active={mode === 'open'} onClick={() => setMode('open')} icon={FolderOpen} label="Open Cart" />
          <ModeTab active={mode === 'new'} onClick={() => setMode('new')} icon={FilePlus2} label="New Cart" />
        </div>

        {/* Cart-state banner */}
        {mode === 'open' ? <OpenCartBanner /> : <NewCartBanner />}

        {/* Operations grid — only shown when there's a mounted cart */}
        {mode === 'open' && mounted && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Add panel. For LocalCart adds an additional 'source label'
                  input so the new passage gets a sourcePath entry (used by
                  the Source-files panel for organization + delete-by-source). */}
              <OpPanel
                icon={Plus}
                title="Add"
                accent="green"
                disabled={readOnly}
                disabledReason="Cart is read-only — unlock first"
              >
                {isLocalMount && (
                  <input
                    className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-green-500/60"
                    placeholder="Source label (e.g. 'my-notes.md', defaults to <inline>)"
                    value={addSource}
                    onChange={(e) => setAddSource(e.target.value)}
                    disabled={addFilesProcessing}
                  />
                )}
                <textarea
                  className="w-full h-32 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 resize-none font-mono focus:outline-none focus:border-purple-500/60"
                  placeholder={isLocalMount ? 'Passage text to add to your local cart…' : 'New passage text…'}
                  value={addText}
                  onChange={(e) => setAddText(e.target.value)}
                  disabled={readOnly || addFilesProcessing}
                />
                <button
                  className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    addText.trim() && !readOnly && !addFilesProcessing
                      ? 'bg-green-500/20 border border-green-500/40 text-green-300 hover:bg-green-500/30'
                      : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                  }`}
                  disabled={!addText.trim() || readOnly || addFilesProcessing}
                  onClick={handleAdd}
                >
                  {isLocalMount ? 'Add to LocalCart' : 'Add Passage'}
                </button>

                {/* File-picker Add — LocalCart only. Pick one or more files
                    (PDF, DOCX, XLSX, MD, HTML, RTF, TXT supported via the
                    cart-builder-v2 parsers); each becomes N passages (one per
                    300-word chunk) with source = file.name so they show up
                    grouped in the Source Files panel below. */}
                {isLocalMount && (
                  <>
                    <div className="flex items-center gap-2 my-1">
                      <div className="flex-1 h-px bg-slate-700/40" />
                      <span className="text-[10px] uppercase tracking-wider text-slate-500">or</span>
                      <div className="flex-1 h-px bg-slate-700/40" />
                    </div>
                    <input
                      ref={addFilesInputRef}
                      type="file"
                      multiple
                      accept=".pdf,.docx,.xlsx,.md,.markdown,.html,.htm,.rtf,.txt"
                      className="hidden"
                      onChange={(e) => handleAddFiles(e.target.files)}
                    />
                    <button
                      className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                        addFilesProcessing
                          ? 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                          : 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30'
                      }`}
                      disabled={addFilesProcessing}
                      onClick={() => addFilesInputRef.current?.click()}
                      title="Pick one or more files — each becomes a set of chunked passages with source = filename"
                    >
                      {addFilesProcessing
                        ? <><Loader2 size={14} className="animate-spin" /> Processing…</>
                        : <><FilePlus2 size={14} /> Pick Files…</>}
                    </button>
                    {addFilesStatus && (
                      <div className="text-[10px] text-slate-400 italic px-1 leading-snug">
                        {addFilesStatus}
                      </div>
                    )}
                    <div className="text-[10px] text-slate-500 leading-snug px-1">
                      Supports PDF, DOCX, XLSX, MD, HTML, RTF, TXT — chunked at 300 words, 50-overlap.
                    </div>
                  </>
                )}
              </OpPanel>

              {/* Update panel. Backend-mounted carts: full editable flow.
                  LocalCart: rendered as a disabled "COMING SOON" preview so
                  users can see the capability is on the roadmap without it
                  being functional. Andy 2026-06-17 PM: file-level / passage-
                  level update on LocalCart needs UX design (which IDX system
                  does the user mean? do file-level updates roundtrip back to
                  the source file on disk?). Add + Delete handle the
                  "edit-by-replace" pattern in the meantime. */}
              <OpPanel
                icon={Save}
                title="Update"
                accent="cyan"
                disabled={readOnly || isLocalMount}
                disabledReason={isLocalMount ? undefined : 'Cart is read-only — unlock first'}
              >
                {isLocalMount && (
                  <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 flex items-center gap-2">
                    <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-500/25 border border-amber-500/50 text-amber-100 font-mono shrink-0">
                      Coming Soon
                    </span>
                    <span className="leading-snug">
                      For now, edit by deleting + re-adding (Delete panel + Add panel).
                    </span>
                  </div>
                )}
                <div className="flex gap-2">
                  <input
                    className="w-24 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60 disabled:opacity-40"
                    placeholder="idx"
                    value={updateIdx}
                    onChange={(e) => setUpdateIdx(e.target.value)}
                    disabled={readOnly || isLocalMount}
                  />
                  <button
                    className="flex-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 disabled:opacity-50"
                    disabled={!updateIdx || readOnly || isLocalMount || updateLoading}
                    onClick={handleUpdateLoad}
                  >
                    {updateLoading ? <Loader2 size={12} className="animate-spin inline" /> : 'Load'}
                  </button>
                </div>
                <textarea
                  className="w-full h-24 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 resize-none font-mono focus:outline-none focus:border-purple-500/60 disabled:opacity-40"
                  placeholder={isLocalMount ? '(coming soon)' : '(load a passage by idx, or paste new text)'}
                  value={updateText}
                  onChange={(e) => setUpdateText(e.target.value)}
                  disabled={readOnly || isLocalMount}
                />
                <button
                  className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    updateIdx && updateText.trim() && !readOnly && !isLocalMount
                      ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30'
                      : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                  }`}
                  disabled={!updateIdx || !updateText.trim() || readOnly || isLocalMount}
                  onClick={handleUpdateSave}
                  title={isLocalMount ? 'Coming soon — for now, delete + re-add' : 'Saves new text + tombstones the old idx (until backend gets a true PUT route)'}
                >
                  {isLocalMount ? 'Coming Soon' : 'Update Passage'}
                </button>
              </OpPanel>

              {/* Delete panel. Backend-mounted carts: passage-IDX only (no
                  file-level provenance on legacy server-built carts). LocalCart:
                  user picks File # (from Source Files list below) or Passage #
                  via a toggle so it's explicit which IDX system they're typing.
                  Andy 2026-06-17 PM: re-enabled for LocalCart after the IDX
                  column on Source Files made file-level delete unambiguous. */}
              <OpPanel
                icon={Trash2}
                title="Delete"
                accent="rose"
                disabled={readOnly}
                disabledReason="Cart is read-only — unlock first"
              >
                {isLocalMount && (
                  // Match the Mode-tabs pattern at top of Edit Carts (slate-800/40
                  // outer + X-500/30 active) — both have html.light overrides so
                  // they read in light + dark mode.
                  <div className="flex gap-1 bg-slate-800/40 rounded-lg p-1">
                    <button
                      onClick={() => setDeleteMode('file')}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                        deleteMode === 'file'
                          ? 'bg-rose-500/30 text-rose-200'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                      }`}
                      title="Type a file # from the Source Files list below — tombstones every passage from that file"
                    >
                      File #
                    </button>
                    <button
                      onClick={() => setDeleteMode('passage')}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                        deleteMode === 'passage'
                          ? 'bg-rose-500/30 text-rose-200'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                      }`}
                      title="Type a passage # (the IDX shown next to a search result) — tombstones just that one passage"
                    >
                      Passage #
                    </button>
                  </div>
                )}
                <div className="text-xs text-slate-500 leading-relaxed">
                  {isLocalMount && deleteMode === 'file' ? (
                    <>Tombstone every passage from one source file. Type its row # from the Source Files list below.</>
                  ) : isLocalMount && deleteMode === 'passage' ? (
                    <>Tombstone a single passage by its IDX (the # shown next to a search result). Restore from the list below to undo.</>
                  ) : (
                    <>Tombstone a passage by index. The pattern stays on disk until you Save the cart; Restore from the list below to undo.</>
                  )}
                </div>
                <input
                  className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
                  placeholder={
                    isLocalMount && deleteMode === 'file'
                      ? 'File # (from Source Files below)'
                      : isLocalMount && deleteMode === 'passage'
                        ? 'Passage # (from search results)'
                        : 'Pattern idx'
                  }
                  value={deleteIdx}
                  onChange={(e) => setDeleteIdx(e.target.value)}
                  disabled={readOnly}
                />
                <button
                  className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    deleteIdx && !readOnly
                      ? 'bg-rose-500/20 border border-rose-500/40 text-rose-300 hover:bg-rose-500/30'
                      : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                  }`}
                  disabled={!deleteIdx || readOnly}
                  onClick={handleDelete}
                >
                  {isLocalMount && deleteMode === 'file'
                    ? 'Tombstone File'
                    : isLocalMount && deleteMode === 'passage'
                      ? 'Tombstone Passage'
                      : 'Tombstone Pattern'}
                </button>
              </OpPanel>
            </div>

            {/* Source-files panel — LocalCart only. Demo step 6 ("delete the
                specific single file") flows through here: lists every distinct
                sourcePath in the cart with its active+total passage counts and
                a Delete-file button that tombstones every passage from that
                source. Clicking a row drills into a per-file passages view
                (Andy 2026-06-28) so individual passages can be edited or
                tombstoned without leaving Edit Carts. */}
            {isLocalMount && activeLocalCart && (
              <SourceFilesPanel
                sources={localCartListSources()}
                onDeleteSource={(sourcePath, activeCount) => handleDeleteSource(sourcePath, activeCount)}
              />
            )}

            {/* Passage browser — backend-mounted carts only. Paginated 25-per-page
                with filter, click row to populate Update + Delete IDX fields.
                Hidden for LocalCart because the API calls it makes (api.listPatterns)
                go to the backend which doesn't know about LocalCart state. The
                LocalCart equivalent is browsable in Search (which uses
                cosineSearchLocal directly). */}
            {!isLocalMount && (
              <PassageBrowser
                patternCount={patternCount}
                mountedKey={status?.mounted_cartridge ?? ''}
                onPickIdx={(idx) => {
                  setUpdateIdx(String(idx))
                  setDeleteIdx(String(idx))
                }}
              />
            )}

            {/* Tombstoned list — backend-mounted reads from deletedPatterns
                (fetched from /api/deleted_patterns), LocalCart-mounted reads
                from activeLocalCart.tombstones in-memory state. For LocalCart
                we also surface the sourcePath alongside the idx — Andy 6/17 PM:
                lets the user verify at a glance which actual passage got
                tombstoned, regardless of whether they came from File mode
                (file # in Delete panel) or Passage mode (passage idx). */}
            {(() => {
              const localTombstoneList = isLocalMount && activeLocalCart
                ? Array.from(activeLocalCart.tombstones).sort((a, b) => a - b).map((idx) => {
                    const passage = activeLocalCart.passages[idx] ?? ''
                    const firstNewline = passage.indexOf('\n')
                    const title = (firstNewline > 0 ? passage.slice(0, firstNewline) : passage.slice(0, 80)).trim()
                    const sourcePath = activeLocalCart.sourcePaths?.[idx] ?? null
                    return { idx, title, sourcePath }
                  })
                : []
              const items: Array<{ idx: number; title: string; sourcePath?: string | null }> = isLocalMount ? localTombstoneList : deletedPatterns
              return (
                <div className="rounded-lg border border-slate-700 bg-slate-800/30">
                  <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between">
                    <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
                      <Trash2 size={12} />
                      Tombstoned ({items.length})
                    </h2>
                    <span className="text-[10px] text-slate-600">
                      {isLocalMount
                        ? 'Click Save Cart to persist by re-downloading a modified .cart.npz'
                        : 'Save the cart to GC permanently'}
                    </span>
                  </div>
                  {items.length === 0 ? (
                    <div className="p-4 text-center text-xs text-slate-600 italic">
                      No tombstoned passages.
                    </div>
                  ) : (
                    <div className="divide-y divide-slate-800">
                      {items.map((p) => (
                        <div key={p.idx} className="px-4 py-2 flex items-center gap-3 text-sm">
                          <span className="text-xs text-slate-500 font-mono w-12 shrink-0">#{p.idx}</span>
                          <div className="flex-1 min-w-0">
                            <div className="truncate text-slate-400">{p.title}</div>
                            {isLocalMount && p.sourcePath && (
                              <div className="text-[10px] font-mono text-cyan-400/70 truncate" title={`Source: ${p.sourcePath}`}>
                                from {p.sourcePath}
                              </div>
                            )}
                          </div>
                          <button
                            onClick={() => handleRestore(p.idx)}
                            className="px-2 py-1 rounded text-xs bg-slate-700/50 hover:bg-slate-600 text-slate-300 hover:text-slate-100 flex items-center gap-1 shrink-0"
                          >
                            <RotateCcw size={11} /> Restore
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })()}
          </>
        )}

        {/* New-cart mode body */}
        {mode === 'new' && (
          <NewCartPanel
            cartName={newCartName}
            setCartName={setNewCartName}
            log={log}
            setMode={setMode}
          />
        )}

        {/* Cart browser — only relevant on Open Cart mode where the user is
            picking an existing cart to mount + edit. Hidden on New Cart mode
            since the user is composing from scratch; the destination folder
            for the new cart is chosen inline in NewCartPanel. */}
        {mode === 'open' && (
          <CartBrowser
            headerLabel="Carts available to edit"
            onCartClick={(cart) => {
              // Mount via the existing /api/cartridges/mount path (absolute path).
              // Log as 'mount' (Andy 2026-05-06 — mount/unmount belong in the log).
              useAppStore.getState().mount(cart.path)
              log('mount', `Mounted ${cart.name} for editing`, true)
            }}
          />
        )}

        {/* Activity log */}
        <ActivityLog entries={activity} />

        {/* Footer save bar — shown when dirty. Routes to localCartSave for
            LocalCart (re-downloads via showSaveFilePicker) or saveCartridge
            for backend mounts. */}
        {dirty && mounted && (
          <div className="sticky bottom-0 -mx-6 px-6 py-3 border-t border-purple-500/30 bg-slate-900/95 backdrop-blur flex items-center justify-between">
            <div className="text-sm text-amber-300 flex items-center gap-2">
              <AlertCircle size={14} />
              {isLocalMount
                ? 'Unsaved local edits — click Save Cart to download a modified .cart.npz to your disk.'
                : 'Unsaved changes — tombstones and adds aren\'t on disk yet.'}
            </div>
            <button
              onClick={async () => {
                if (isLocalMount) {
                  const r = await localCartSave()
                  log('save', r.message, r.success)
                } else {
                  const r = await saveCartridge()
                  log('save', r.message, r.success)
                }
              }}
              className="px-4 py-1.5 rounded-lg text-sm font-medium bg-purple-500/30 border border-purple-500/50 text-purple-200 hover:bg-purple-500/40"
            >
              <Save size={14} className="inline mr-1.5" />
              Save Cart
            </button>
          </div>
        )}
      </div>

      {/* Destructive-action confirmation modal — used for tombstone + update */}
      <ConfirmDialog
        state={confirm}
        onCancel={() => setConfirm(null)}
        onConfirm={async () => {
          await confirm?.onConfirm()
          setConfirm(null)
        }}
      />
    </main>
  )

  // ----- Helper components below (closures so they can read store state) -----

  function OpenCartBanner() {
    const readOnlyMode = !!status?.read_only_mode
    if (!mounted) {
      return (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 flex items-start gap-3">
          <AlertCircle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <div className="text-amber-200 font-medium mb-1">No cart mounted</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Edit Carts is the destructive screen — passages get tombstoned, updated, and deleted here.{' '}
              {readOnlyMode ? (
                <>
                  On the public VPS, mount a cart via{' '}
                  <strong className="text-slate-200">Search → "Open from My Computer"</strong>{' '}
                  (uses your browser's File System Access; bytes never leave your machine)
                  or via <strong className="text-slate-200">"Upload Cartridge"</strong>{' '}
                  (1-hour sandboxed for trying public carts).{' '}
                  <strong className="text-slate-200">Browser-built carts saved to your local disk via Cart Builder</strong>{' '}
                  need to be re-mounted via "Open from My Computer" before Edit Carts can see them — the public VPS
                  cannot reach into your filesystem directly (that's the browser sandbox doing its job, not a bug).
                </>
              ) : (
                <>
                  Mount a cart from the <strong className="text-slate-200">Search</strong> screen first
                  (the toolbar there has the file picker, upload, and the cart list), then come back to edit.
                  Or pick from <strong className="text-slate-200">My Carts</strong> below if you already
                  know which one you want.
                </>
              )}
            </div>
          </div>
        </div>
      )
    }
    // Display name + filename are different for local vs backend mounts.
    // Backend: name is the cartridge filename on the server.
    // Local: show the cart's friendly name + filename ("from disk" hint).
    const displayName = isLocalMount && activeLocalCart
      ? activeLocalCart.name
      : status?.mounted_cartridge
    const tombCount = activeLocalCart?.tombstones.size ?? 0
    const subDetail = isLocalMount && activeLocalCart
      ? `${patternCount.toLocaleString()} active · ${tombCount > 0 ? `${tombCount} tombstoned · ` : ''}${dirty ? 'unsaved changes' : 'clean'} · from ${activeLocalCart.filename}`
      : `${patternCount.toLocaleString()} patterns · ${dirty ? 'unsaved changes' : 'clean'}`
    return (
      <div className={`rounded-lg border p-3 flex items-center gap-3 ${
        isLocalMount
          ? 'border-cyan-500/30 bg-cyan-500/5'
          : 'border-purple-500/30 bg-purple-500/5'
      }`}>
        <Database size={16} className={isLocalMount ? 'text-cyan-400 shrink-0' : 'text-purple-400 shrink-0'} />
        <div className="flex-1 min-w-0">
          <div className="text-sm text-slate-200 font-medium flex items-center gap-2">
            <span className="truncate">{displayName}</span>
            {isLocalMount && (
              <span
                className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 font-mono shrink-0"
                title="This cart is mounted from your local disk via File System Access API. Edits are in-memory; Save downloads a modified copy."
              >
                LOCAL
              </span>
            )}
          </div>
          <div className="text-[11px] text-slate-500">{subDetail}</div>
        </div>
        {/* Lock toggle only meaningful for backend carts. Local carts are
            always editable (user owns the file they picked from their disk). */}
        {!isLocalMount && (
          <button
            onClick={toggleLock}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              readOnly
                ? 'bg-rose-500/20 border border-rose-500/40 text-rose-300 hover:bg-rose-500/30'
                : 'bg-green-500/20 border border-green-500/40 text-green-300 hover:bg-green-500/30'
            }`}
            title={readOnly ? 'Click to unlock for editing' : 'Click to lock (read-only)'}
          >
            {readOnly ? <><Lock size={11} /> Read-only</> : <><Unlock size={11} /> Editable</>}
          </button>
        )}
        {isLocalMount && (
          <span
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-green-500/20 border border-green-500/40 text-green-300"
            title="LocalCart is always editable — you picked this file from your own disk."
          >
            <Unlock size={11} /> Editable
          </span>
        )}
        <button
          onClick={async () => {
            const name = displayName || 'cart'
            if (isLocalMount) {
              unmountLocalCart()
            } else {
              await unmount()
            }
            log('unmount', `Unmounted ${name}`, true)
          }}
          className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800"
        >
          Unmount
        </button>
      </div>
    )
  }

  function NewCartBanner() {
    return (
      <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-3 flex items-start gap-3">
        <FilePlus2 size={16} className="text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-slate-300 leading-relaxed">
          New cart mode: start blank, add passages one at a time. The cart is saved to disk only
          when you click "Save Cart" at the bottom of the screen.
        </div>
      </div>
    )
  }
}

// ----- Standalone subcomponents -----

function ModeTab({
  active, onClick, icon: Icon, label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ComponentType<{ size?: number; className?: string }>
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors ${
        active
          ? 'bg-purple-500/30 text-purple-200'
          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
      }`}
    >
      <Icon size={14} />
      {label}
    </button>
  )
}

type Accent = 'green' | 'cyan' | 'rose' | 'purple'

function OpPanel({
  icon: Icon, title, accent, disabled, disabledReason, children,
}: {
  icon: React.ComponentType<{ size?: number; className?: string }>
  title: string
  accent: Accent
  disabled?: boolean
  disabledReason?: string
  children: React.ReactNode
}) {
  const accentColor: Record<Accent, string> = {
    green: 'text-green-400',
    cyan: 'text-cyan-400',
    rose: 'text-rose-400',
    purple: 'text-purple-400',
  }
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Icon size={16} className={accentColor[accent]} />
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        {disabled && disabledReason && (
          <span className="ml-auto text-[10px] text-amber-400/70 italic" title={disabledReason}>
            disabled
          </span>
        )}
      </div>
      {children}
    </div>
  )
}

function ActivityLog({ entries }: { entries: ActivityEntry[] }) {
  if (entries.length === 0) {
    return (
      <div className="rounded-lg border border-slate-700/50 bg-slate-800/20 p-4 text-center text-xs text-slate-600 italic">
        No activity yet — Add, Update, Delete, or Restore to see entries here.
      </div>
    )
  }
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/30">
      <div className="px-4 py-2 border-b border-slate-700">
        <h2 className="text-xs uppercase tracking-wider text-slate-500">Activity</h2>
      </div>
      <div className="divide-y divide-slate-800 max-h-48 overflow-y-auto">
        {entries.map((e, i) => (
          <div key={i} className="px-4 py-1.5 flex items-center gap-3 text-xs">
            <span className="text-slate-600 font-mono w-20 shrink-0">{e.ts}</span>
            {e.ok
              ? <CheckCircle2 size={12} className="text-green-400 shrink-0" />
              : <AlertCircle size={12} className="text-rose-400 shrink-0" />}
            <span className={`uppercase tracking-wider w-20 shrink-0 ${
              e.kind === 'add'      ? 'text-green-400'
                : e.kind === 'update'  ? 'text-cyan-400'
                : e.kind === 'delete'  ? 'text-rose-400'
                : e.kind === 'restore' ? 'text-amber-400'
                : e.kind === 'mount'   ? 'text-purple-400'
                : e.kind === 'unmount' ? 'text-purple-300/70'
                : e.kind === 'save'    ? 'text-emerald-400'
                : e.kind === 'create'  ? 'text-cyan-300'
                : e.kind === 'open'    ? 'text-blue-400'
                : 'text-slate-400'
            }`}>{e.kind}</span>
            <span className="flex-1 truncate text-slate-400">{e.detail}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SourceFilesPanel — for LocalCart-mounted carts only.
//
// Two view modes (Andy 2026-06-28 spec):
//   • list — every unique source file with active/total counts + Delete-file
//   • drill — one file's passages, paginated 25/page with per-row Edit +
//             Tombstone/Restore. Other files hide while drilled.
//
// Drill state is component-local so browsing back and forth doesn't affect
// the rest of the screen. Pagination block is lifted verbatim from the
// backend PassageBrowser (PASSAGE_PAGE_SIZE, Prev/Next/jump-to-page) so the
// muscle memory carries over.
// ─────────────────────────────────────────────────────────────────────────────

function SourceFilesPanel({
  sources,
  onDeleteSource,
}: {
  sources: Array<{ sourcePath: string; count: number; activeCount: number }>
  onDeleteSource: (sourcePath: string, activeCount: number) => void
}) {
  const [drillPath, setDrillPath] = useState<string | null>(null)
  const [drillOffset, setDrillOffset] = useState(0)
  const [drillPageJump, setDrillPageJump] = useState('')

  const listPassagesBySource = useAppStore((s) => s.localCartListPassagesBySource)
  const openEditor = useAppStore((s) => s.openEditor)
  const localCartTombstone = useAppStore((s) => s.localCartTombstone)
  const localCartRestore = useAppStore((s) => s.localCartRestore)
  // Trigger re-render when the active cart's tombstones/passages mutate. The
  // listPassagesBySource selector reads from the store internally, but React
  // only re-renders this panel if the selector's *identity* or a subscribed
  // value changes — subscribing to activeLocalCart keeps the drill view in
  // sync with edits.
  useAppStore((s) => s.activeLocalCart)
  useAppStore((s) => s.localCarts)

  // Sort by activeCount descending (most-impactful files first), with all-
  // tombstoned files at the bottom and same-counts alphabetically.
  const sorted = [...sources].sort((a, b) => {
    if (b.activeCount !== a.activeCount) return b.activeCount - a.activeCount
    return a.sourcePath.localeCompare(b.sourcePath)
  })

  // ── Drill view ─────────────────────────────────────────────────────────
  if (drillPath !== null) {
    const window = listPassagesBySource(drillPath, drillOffset, PASSAGE_PAGE_SIZE)
    const total = window.total
    const totalPages = Math.max(1, Math.ceil(total / PASSAGE_PAGE_SIZE))
    const currentPage = Math.floor(drillOffset / PASSAGE_PAGE_SIZE) + 1
    const goPrev = () => setDrillOffset(Math.max(0, drillOffset - PASSAGE_PAGE_SIZE))
    const goNext = () =>
      setDrillOffset(Math.min((totalPages - 1) * PASSAGE_PAGE_SIZE, drillOffset + PASSAGE_PAGE_SIZE))
    const goPage = (page: number) => {
      const clamped = Math.max(1, Math.min(totalPages, page))
      setDrillOffset((clamped - 1) * PASSAGE_PAGE_SIZE)
    }
    const exitDrill = () => {
      setDrillPath(null)
      setDrillOffset(0)
      setDrillPageJump('')
    }
    return (
      <div className="rounded-lg border border-cyan-500/40 bg-cyan-500/5">
        {/* Drill header — back-arrow · SOURCE FILE label · filename · PASSAGES
            counts. Cyan accent (matches parent list) and light-mode-safe
            text-slate-100 on the filename per 6/28 contrast lesson. */}
        <div className="px-4 py-2 border-b border-cyan-500/40 flex items-center gap-3">
          <button
            onClick={exitDrill}
            className="flex items-center gap-1 px-2 py-1 rounded text-[11px] uppercase tracking-wider
                       bg-cyan-500/10 border border-cyan-500/30 text-cyan-200
                       hover:bg-cyan-500/20 hover:text-cyan-100 transition-colors"
            title="Back to Source files list"
          >
            <ChevronLeft size={11} />
            Back
          </button>
          <span className="text-[10px] uppercase tracking-wider text-cyan-400 shrink-0">
            Source file
          </span>
          <span
            className="text-xs font-mono text-slate-100 truncate flex-1"
            title={drillPath}
          >
            {drillPath}
          </span>
          <span className="text-[10px] uppercase tracking-wider text-cyan-400 shrink-0">
            Passages
          </span>
          <span className="text-[10px] font-mono text-slate-400 shrink-0">
            ({window.activeCount} active / {total} total)
          </span>
        </div>

        {/* Per-passage rows */}
        <div className="max-h-[320px] overflow-y-auto divide-y divide-cyan-500/20">
          {total === 0 ? (
            <div className="p-4 text-center text-xs text-slate-500 italic">
              No passages found for this source file.
            </div>
          ) : (
            window.passages.map((p) => (
              <div
                key={p.idx}
                className={`px-4 py-2 flex items-start gap-3 text-xs ${
                  p.tombstoned ? 'opacity-60' : ''
                }`}
              >
                <span className="font-mono text-[11px] text-slate-500 w-12 shrink-0 mt-0.5">
                  #{p.idx}
                </span>
                <div className="flex-1 min-w-0">
                  <div
                    className={`text-xs truncate ${
                      p.tombstoned ? 'text-slate-500 line-through' : 'text-slate-200'
                    }`}
                    title={p.title}
                  >
                    {p.title || '[empty]'}
                  </div>
                  {p.preview && (
                    <div className="text-[10px] text-slate-500 truncate mt-0.5">
                      {p.preview}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    onClick={() => {
                      // Reuse the existing PassageEditor + editorOriginalIdx
                      // machinery. saveEditor() will add-then-tombstone the old
                      // idx for backend carts; for LocalCart it flows through
                      // localCartAddPassage + tombstone via the same path.
                      const state = useAppStore.getState()
                      const activeName = state.activeLocalCart
                      const cart = activeName ? state.localCarts.get(activeName) : null
                      const fullText = cart?.passages[p.idx] ?? ''
                      openEditor(fullText, p.idx)
                    }}
                    disabled={p.tombstoned}
                    className="px-2 py-1 rounded text-[11px] bg-cyan-500/15 border border-cyan-500/30 text-cyan-200
                               hover:bg-cyan-500/25 disabled:opacity-30 disabled:cursor-not-allowed
                               flex items-center gap-1"
                    title={p.tombstoned ? 'Restore before editing' : 'Edit this passage'}
                  >
                    <Pencil size={10} />
                    Edit
                  </button>
                  {p.tombstoned ? (
                    <button
                      onClick={() => localCartRestore(p.idx)}
                      className="px-2 py-1 rounded text-[11px] bg-amber-500/15 border border-amber-500/30 text-amber-200
                                 hover:bg-amber-500/25 flex items-center gap-1"
                      title="Restore this passage"
                    >
                      <RotateCcw size={10} />
                      Restore
                    </button>
                  ) : (
                    <button
                      onClick={() => localCartTombstone(p.idx)}
                      className="px-2 py-1 rounded text-[11px] bg-rose-500/15 border border-rose-500/30 text-rose-300
                                 hover:bg-rose-500/25 flex items-center gap-1"
                      title="Tombstone just this passage"
                    >
                      <Trash2 size={10} />
                      Tombstone
                    </button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Pagination — same shape as PassageBrowser (Prev/Next + jump-to-page) */}
        {total > PASSAGE_PAGE_SIZE && (
          <div className="px-4 py-2 border-t border-cyan-500/20 flex items-center justify-between gap-2 text-xs text-slate-500">
            <button
              onClick={goPrev}
              disabled={drillOffset === 0}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={11} />
              Prev
            </button>
            <div className="flex items-center gap-2">
              <span className="font-mono">
                Page {currentPage.toLocaleString()} of {totalPages.toLocaleString()}
              </span>
              <span className="text-slate-600">·</span>
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  const n = parseInt(drillPageJump, 10)
                  if (!isNaN(n)) goPage(n)
                  setDrillPageJump('')
                }}
                className="flex items-center gap-1"
              >
                <span className="text-[10px] uppercase tracking-wider">jump</span>
                <input
                  type="text"
                  value={drillPageJump}
                  onChange={(e) => setDrillPageJump(e.target.value)}
                  placeholder={String(currentPage)}
                  className="w-12 rounded bg-slate-950/60 border border-slate-800 px-1.5 py-0.5 text-[11px] text-slate-200 font-mono focus:outline-none focus:border-cyan-500/60"
                />
              </form>
            </div>
            <button
              onClick={goNext}
              disabled={drillOffset + PASSAGE_PAGE_SIZE >= total}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
              <ChevronRight size={11} />
            </button>
          </div>
        )}
      </div>
    )
  }

  // ── List view ─────────────────────────────────────────────────────────
  return (
    <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/5">
      <div className="px-4 py-2 border-b border-cyan-500/30 flex items-center justify-between gap-3">
        <h2 className="text-xs uppercase tracking-wider text-cyan-300 flex items-center gap-2">
          <Folder size={12} />
          Source files ({sorted.length})
        </h2>
        <span className="text-[10px] text-slate-500">
          click a filename to drill in; Delete file tombstones the whole source
        </span>
      </div>
      {sorted.length === 0 ? (
        <div className="p-4 text-center text-xs text-slate-600 italic">
          No source files indexed in this cart. Provenance v1 sidecar may be missing
          (legacy server-built cart). Use Delete by IDX above for per-passage tombstoning.
        </div>
      ) : (
        <div className="max-h-[280px] overflow-y-auto divide-y divide-cyan-500/20">
          {sorted.map((s, i) => {
            // File-level IDX (1-based row number in the sorted list). Andy 6/17 PM:
            // users need an identifier on each row so they can reference files
            // by number when discussing or deleting. Sequential row index works
            // for the visible-list use case; doesn't need to be stable across
            // sessions (sort order changes when tombstones shift activeCount).
            const fileIdx = i + 1
            const allDeleted = s.activeCount === 0
            return (
              <div
                key={s.sourcePath}
                className={`px-4 py-2 flex items-center gap-3 text-xs ${
                  allDeleted ? 'opacity-50' : ''
                }`}
              >
                <span className="text-[10px] text-slate-500 font-mono w-8 shrink-0 text-right">
                  #{fileIdx}
                </span>
                <Folder size={11} className="text-cyan-400 shrink-0" />
                <button
                  type="button"
                  onClick={() => {
                    setDrillPath(s.sourcePath)
                    setDrillOffset(0)
                    setDrillPageJump('')
                  }}
                  className="flex-1 truncate font-mono text-slate-300 hover:text-cyan-200 text-left transition-colors"
                  title={`Drill into ${s.sourcePath}`}
                >
                  {s.sourcePath}
                </button>
                <span className="text-[10px] text-slate-500 font-mono shrink-0">
                  {allDeleted ? (
                    <span className="text-rose-400">all tombstoned</span>
                  ) : (
                    <>
                      {s.activeCount} active
                      {s.activeCount !== s.count && (
                        <span className="text-slate-600"> / {s.count} total</span>
                      )}
                    </>
                  )}
                </span>
                <button
                  onClick={() => onDeleteSource(s.sourcePath, s.activeCount)}
                  disabled={allDeleted}
                  className="px-2 py-1 rounded text-xs bg-rose-500/15 border border-rose-500/30 text-rose-300 hover:bg-rose-500/25 disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1 shrink-0"
                  title={allDeleted ? 'All passages from this file are already tombstoned' : `Tombstone all ${s.activeCount} passages from this file`}
                >
                  <Trash2 size={10} /> Delete file
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// NewCartPanel — compose a cart by typing passages directly.
//
// Routes the typed passages through the cart-builder-v2 pipeline (skip parse,
// run chunker → embedder → writer). The output BuiltCart is saved via the
// File System Access API (or Downloads fallback). After save the user mounts
// the cart from Search — auto-mount lands with v1.2 alongside CLOUD mode and
// per-user data stores.
// ─────────────────────────────────────────────────────────────────────────────

interface DraftPassage {
  id: string
  text: string
}

function NewCartPanel({
  cartName,
  setCartName,
  log,
  setMode,
}: {
  cartName: string
  setCartName: (v: string) => void
  log: (kind: OpKind, detail: string, ok: boolean) => void
  setMode: (mode: Mode) => void
}) {
  const [passages, setPassages] = useState<DraftPassage[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [draftText, setDraftText] = useState('')
  const [building, setBuilding] = useState(false)
  const [progress, setProgress] = useState<PipelineProgress | null>(null)
  const [result, setResult] = useState<BuiltCart | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [destFolder, setDestFolder] = useState<string>('')
  const [destFolderCarts, setDestFolderCarts] = useState<{ name: string; filename: string; size_mb: number; passages: number | string }[]>([])
  const [destFolderLoading, setDestFolderLoading] = useState(false)
  const [destFolderError, setDestFolderError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const openFolderPicker = useCartBuilderStore((s) => s.openFolderPicker)
  const refreshBrowser = useCartBuilderStore((s) => s.refreshBrowser)
  const mountCart = useAppStore((s) => s.mount)
  const toggleLock = useAppStore((s) => s.toggleLock)

  // Pre-flight: when a destination folder is picked, list any existing
  // .cart.npz files there so the user sees collisions before committing
  // the build. Andy 2026-05-10: "if there are carts already in there
  // they can see if they chose a name that's already taken before even
  // committing to the process."
  useEffect(() => {
    if (!destFolder) {
      setDestFolderCarts([])
      setDestFolderError(null)
      return
    }
    let cancelled = false
    setDestFolderLoading(true)
    setDestFolderError(null)
    cb.listCarts(destFolder)
      .then((resp) => {
        if (cancelled) return
        setDestFolderCarts(resp.carts.map((c) => ({
          name: c.name,
          filename: c.filename,
          size_mb: c.size_mb,
          passages: c.passages,
        })))
      })
      .catch((e) => {
        if (cancelled) return
        setDestFolderError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setDestFolderLoading(false)
      })
    return () => { cancelled = true }
  }, [destFolder])

  // Sanitized cart filename for collision detection — must match server-side
  // sanitization in api/cartbuilder.py:build_to_folder so the highlight is
  // accurate before the actual write attempt.
  const sanitizedFilename = (() => {
    const safe = cartName.trim().replace(/[^A-Za-z0-9_-]/g, '_')
    return `${safe || 'new-cart'}.cart.npz`
  })()
  const collisionDetected = destFolderCarts.some((c) => c.filename === sanitizedFilename)

  // Prerequisite gating — cart name + destination folder must be set before
  // any passage editing. Without these, the user is adding passages to
  // "the infinite void" (Andy 2026-05-10).
  const prerequisiteIssue: string | null = (() => {
    if (!cartName.trim()) return 'Enter a cart name above before adding passages.'
    if (!destFolder) return 'Choose a destination folder before adding passages.'
    return null
  })()

  const isEditing = activeId !== null
  const canAddOrUpdate = !prerequisiteIssue && draftText.trim().length > 0
  const canBuild = !building && passages.length > 0 && !prerequisiteIssue
  const buildPct = (() => {
    if (!progress) return 0
    if (progress.stage === 'done') return 100
    if (progress.stage === 'embedding' && progress.embeddingsTotal) {
      const done = progress.embeddingsCompleted ?? 0
      return Math.min(95, 30 + Math.round((done / progress.embeddingsTotal) * 60))
    }
    if (progress.stage === 'parsing') return 10
    if (progress.stage === 'chunking') return 20
    if (progress.stage === 'writing') return 96
    return 5
  })()

  const handleAddOrUpdate = () => {
    const text = draftText.trim()
    if (!text) return
    if (isEditing) {
      setPassages((prev) =>
        prev.map((p) => (p.id === activeId ? { ...p, text } : p)),
      )
      setActiveId(null)
    } else {
      const id = crypto.randomUUID()
      setPassages((prev) => [...prev, { id, text }])
    }
    setDraftText('')
    textareaRef.current?.focus()
  }

  const handleCancelEdit = () => {
    setActiveId(null)
    setDraftText('')
    textareaRef.current?.focus()
  }

  const handleSelectForEdit = (p: DraftPassage) => {
    setActiveId(p.id)
    setDraftText(p.text)
    textareaRef.current?.focus()
  }

  const handleDelete = (id: string) => {
    setPassages((prev) => prev.filter((p) => p.id !== id))
    if (activeId === id) {
      setActiveId(null)
      setDraftText('')
    }
  }

  const handleBuild = async () => {
    setBuilding(true)
    setProgress(null)
    setResult(null)
    setError(null)
    try {
      const sanitizedName = cartName.trim().replace(/[^A-Za-z0-9_-]/g, '_') || 'new-cart'
      const built = await buildCartFromPassages(
        passages.map((p) => p.text),
        {
          cartName: sanitizedName,
          onProgress: setProgress,
        },
      )
      setResult(built.cart)
      log('create', `Built ${built.cart.cartFilename} (${built.chunkCount} chunks)`, true)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setError(message)
      log('create', `Build failed: ${message}`, false)
    } finally {
      setBuilding(false)
    }
  }

  const handlePickDestination = () => {
    // Open the server-side folder picker. The picker registers our callback
    // so when the user clicks "Use this folder," we receive the absolute
    // server-resolved path back instead of the default add-to-saved-folders
    // behavior.
    void openFolderPicker({
      path: destFolder || undefined,
      onConfirm: (picked: string) => {
        setDestFolder(picked)
      },
    })
  }

  const handleSaveAndMount = async () => {
    if (!result) return
    const sanitizedName = cartName.trim().replace(/[^A-Za-z0-9_-]/g, '_') || 'new-cart'

    // Inner saver — calls buildToFolder + auto-mount + mode-switch on
    // success. Used twice: first attempt (replace=false), then retry
    // (replace=true) if the user confirms an overwrite.
    const doSave = async (replace: boolean) => {
      const resp = await cb.buildToFolder({
        cartBlob: result.cartBlob,
        manifestBlob: result.manifestBlob,
        permissionsBlob: result.permissionsBlob,
        folder: destFolder,
        cartName: sanitizedName,
        replace,
      })
      log('save', `${replace ? 'Replaced' : 'Saved'} ${resp.mounted_filename} to ${resp.folder}`, true)
      try {
        await mountCart(resp.cart_path)
        // User-just-built carts default RW (Andy 6/15 PM): the cart's owner is
        // mounting it for editing one second after building it. Mount route
        // defaults to read-only for safety on other-people's-carts; flip
        // automatically here so the build-then-edit flow lands editable.
        const postMountStatus = useAppStore.getState().status
        if (postMountStatus?.read_only) {
          await toggleLock()
        }
        log('mount', `Mounted ${resp.mounted_filename} for editing (RW)`, true)
        await refreshBrowser()
        setMode('open')
      } catch (mountErr) {
        const m = mountErr instanceof Error ? mountErr.message : String(mountErr)
        log('mount', `Cart saved but auto-mount failed: ${m}`, false)
      }
    }

    try {
      await doSave(false)
    } catch (err) {
      if (err instanceof cb.CartExistsError) {
        // Server rejected because a cart with this name already exists in
        // the chosen folder. Prompt the user; on confirm, retry with the
        // replace flag set so the server overwrites this time.
        const ok = confirm(
          `A cart named "${sanitizedName}.cart.npz" already exists in:\n\n${destFolder}\n\nReplace it? (The existing cart's three files will be overwritten.)`,
        )
        if (!ok) {
          log('save', 'Save aborted — existing cart not replaced', false)
          return
        }
        try {
          await doSave(true)
        } catch (retryErr) {
          const message = retryErr instanceof Error ? retryErr.message : String(retryErr)
          log('save', `Replace failed: ${message}`, false)
          setError(message)
        }
        return
      }
      const message = err instanceof Error ? err.message : String(err)
      log('save', `Save failed: ${message}`, false)
      setError(message)
    }
  }

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium text-slate-200 flex items-center gap-2">
            <FilePlus2 size={16} className="text-purple-400" />
            New Cart
          </h2>
          <p className="text-xs text-slate-500 leading-relaxed mt-0.5">
            Compose passages directly — each will become one searchable chunk in the resulting cart.
            For document-based ingestion (PDFs, Word, spreadsheets), use Cart Builder instead.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
            Cart name
          </label>
          <input
            type="text"
            value={cartName}
            onChange={(e) => setCartName(e.target.value)}
            placeholder="Enter cart name…"
            disabled={building}
            className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
          />
          <div className="text-[10px] text-slate-600 mt-1 italic">
            alphanumeric, dashes, underscores only — sanitized at build time
          </div>
        </div>

        <div>
          <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
            Destination folder
          </label>
          <button
            onClick={handlePickDestination}
            disabled={building}
            className={`w-full rounded-lg border px-3 py-1.5 text-sm flex items-center gap-2 transition-colors disabled:opacity-50 ${
              destFolder
                ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-200 hover:bg-cyan-500/15'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:bg-slate-900/60'
            }`}
            title={destFolder ? `Cart will save to "${destFolder}". Click to change.` : 'Pick the folder where this cart will be saved.'}
          >
            {destFolder ? <Folder size={14} className="text-cyan-400" /> : <FolderOpen size={14} className="text-slate-500" />}
            <span className="flex-1 text-left truncate font-mono text-xs">
              {destFolder || 'Choose destination folder…'}
            </span>
            {destFolder && <span className="text-[10px] uppercase tracking-wider text-cyan-400/70">change</span>}
          </button>
          <div className="text-[10px] text-slate-600 mt-1 italic">
            {destFolder
              ? 'Cart will save here + auto-mount when built.'
              : 'Required — pick the folder where the cart bundle will live.'}
          </div>
        </div>
      </div>

      {/* Pre-flight preview of existing carts in the destination folder.
          Surfaces name collisions BEFORE the user commits to a build, so
          they can rename or pick a different folder ahead of time. The
          would-collide entry highlights amber. */}
      {destFolder && (
        <div className="rounded-lg border border-slate-700 bg-slate-900/30 px-3 py-2">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
            <Folder size={11} className="text-slate-500" />
            Existing carts in this folder
            <span className="font-mono normal-case tracking-normal text-slate-400">
              ({destFolderLoading ? '…' : destFolderCarts.length})
            </span>
            {collisionDetected && (
              <span className="ml-auto flex items-center gap-1 text-amber-400 normal-case tracking-normal font-medium">
                <AlertCircle size={11} />
                {sanitizedFilename} already exists — Save will prompt to replace
              </span>
            )}
          </div>
          {destFolderError ? (
            <div className="text-xs text-rose-300 italic flex items-center gap-1">
              <AlertCircle size={11} className="text-rose-400 shrink-0" />
              {destFolderError}
            </div>
          ) : destFolderLoading ? (
            <div className="text-xs text-slate-500 italic flex items-center gap-1">
              <Loader2 size={11} className="animate-spin" />
              Reading folder…
            </div>
          ) : destFolderCarts.length === 0 ? (
            <div className="text-xs text-slate-600 italic">
              Empty — no name collisions possible.
            </div>
          ) : (
            <div className="max-h-[120px] overflow-y-auto -mx-1 divide-y divide-slate-800/60">
              {destFolderCarts.map((c) => {
                const isCollision = c.filename === sanitizedFilename
                return (
                  <div
                    key={c.filename}
                    className={`px-2 py-1 flex items-center gap-2 text-xs font-mono ${
                      isCollision ? 'bg-amber-500/10 text-amber-100' : 'text-slate-400'
                    }`}
                  >
                    <Folder size={10} className={isCollision ? 'text-amber-400 shrink-0' : 'text-slate-600 shrink-0'} />
                    <span className="flex-1 truncate">{c.filename}</span>
                    <span className="text-[10px] text-slate-600 shrink-0">
                      {c.passages} passages
                    </span>
                    <span className="text-[10px] text-slate-600 shrink-0">
                      {c.size_mb.toFixed(1)} MB
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Prerequisite hint banner — explicit feedback when the user needs to
          complete cart name + destination folder before composing. */}
      {prerequisiteIssue && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-2.5 text-xs text-amber-100 flex items-start gap-2">
          <AlertCircle size={12} className="text-amber-400 shrink-0 mt-0.5" />
          <span>{prerequisiteIssue}</span>
        </div>
      )}

      <div>
        <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
          {isEditing ? 'Update passage' : 'Add a passage'}
        </label>
        <textarea
          ref={textareaRef}
          value={draftText}
          onChange={(e) => setDraftText(e.target.value)}
          placeholder={isEditing ? 'Edit this passage…' : 'Type a passage — a single thought, paragraph, or chunk of text the cart should be searchable on…'}
          rows={5}
          disabled={building}
          className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-purple-500/60 disabled:opacity-50 resize-y"
        />
        <div className="flex items-center gap-2 mt-2">
          <button
            onClick={handleAddOrUpdate}
            disabled={!canAddOrUpdate || building}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              canAddOrUpdate && !building
                ? isEditing
                  ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-200 hover:bg-cyan-500/30'
                  : 'bg-purple-500/20 border border-purple-500/40 text-purple-200 hover:bg-purple-500/30'
                : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
            }`}
          >
            {isEditing ? <ChevronRight size={12} /> : <Plus size={12} />}
            {isEditing ? 'Update passage' : 'Add passage'}
          </button>
          {isEditing && (
            <button
              onClick={handleCancelEdit}
              disabled={building}
              className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-700/40 flex items-center gap-1 transition-colors"
            >
              <X size={11} />
              Cancel edit
            </button>
          )}
          <span className="ml-auto text-[10px] text-slate-600 font-mono">
            {draftText.trim().split(/\s+/).filter(Boolean).length} word{draftText.trim().split(/\s+/).filter(Boolean).length === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      {/* Passages list */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">
            Passages ({passages.length})
          </span>
          {passages.length > 0 && !building && (
            <button
              onClick={() => {
                if (confirm('Clear all draft passages?')) {
                  setPassages([])
                  setActiveId(null)
                  setDraftText('')
                }
              }}
              className="text-[10px] text-slate-500 hover:text-slate-300"
            >
              Clear all
            </button>
          )}
        </div>
        {passages.length === 0 ? (
          <div className="text-xs text-slate-600 italic py-2 px-3 rounded border border-dashed border-slate-700 bg-slate-900/30">
            No passages yet. Type one above and click <strong className="font-medium text-slate-500">Add passage</strong> to start composing your cart.
          </div>
        ) : (
          <div className="rounded-lg border border-slate-700 divide-y divide-slate-800 bg-slate-900/30">
            {passages.map((p, i) => {
              const isActive = activeId === p.id
              const preview = p.text.length > 140 ? p.text.slice(0, 137) + '…' : p.text
              return (
                <div
                  key={p.id}
                  className={`flex items-start gap-2 px-3 py-2 group ${
                    isActive ? 'bg-cyan-500/10' : 'hover:bg-slate-800/40'
                  }`}
                >
                  <span className="text-[10px] font-mono text-slate-600 w-6 shrink-0 mt-0.5">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <button
                    onClick={() => handleSelectForEdit(p)}
                    disabled={building}
                    className="flex-1 text-left text-xs text-slate-300 hover:text-slate-100 leading-relaxed disabled:opacity-50"
                    title="Click to edit this passage"
                  >
                    {preview}
                  </button>
                  <button
                    onClick={() => handleDelete(p.id)}
                    disabled={building}
                    className="text-rose-400/60 hover:text-rose-400 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30"
                    title="Remove this passage"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Build progress */}
      {building && progress && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Loader2 size={12} className="animate-spin text-amber-400 shrink-0" />
            <span className="text-amber-200 font-medium uppercase tracking-wider text-[10px]">
              {progress.stage}
            </span>
            <span className="text-slate-400 truncate">{progress.message}</span>
            <span className="ml-auto text-amber-300 font-mono text-[10px]">{buildPct}%</span>
          </div>
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-purple-500 transition-all duration-500"
              style={{ width: `${buildPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Build error */}
      {error && !building && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 flex items-start gap-2 text-xs">
          <AlertCircle size={14} className="text-rose-400 shrink-0 mt-0.5" />
          <div className="text-rose-200">{error}</div>
        </div>
      )}

      {/* Build result */}
      {result && !building && (
        <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 space-y-2">
          <div className="flex items-center gap-2 text-sm flex-wrap">
            <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
            <span className="text-emerald-200 font-medium">
              Built {result.cartFilename}
            </span>
            <span className="text-slate-400 font-mono text-[10px] ml-auto">
              {result.manifest.count} chunks · fingerprint {result.manifest.fingerprint}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => { void handleSaveAndMount() }}
              className="px-3 py-1.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 text-xs font-medium hover:bg-emerald-500/30 flex items-center gap-1.5 transition-colors"
              title={`Save to ${destFolder} and mount as Editable (RW) for immediate editing.`}
            >
              <Save size={12} />
              Save + mount for editing
            </button>
            {/* RW state pill (Andy 6/15 PM): users couldn't tell what permission state */}
            {/* their just-built cart would mount in. The Save+mount path auto-flips to */}
            {/* Editable on user-built carts (see handleSaveAndMount auto-unlock). */}
            <span
              className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-green-500/20 border border-green-500/40 text-green-300 font-mono flex items-center gap-1"
              title="Newly-built carts mount in Editable (RW) mode automatically — you own the cart you just built."
            >
              <Unlock size={9} />
              Will mount: Editable
            </span>
            <span className="text-[10px] text-slate-500 truncate">
              → {destFolder || '(no folder selected)'}
            </span>
          </div>
        </div>
      )}

      {/* Build button — sticks at the bottom */}
      <div className="pt-2 border-t border-slate-800">
        <button
          onClick={() => { void handleBuild() }}
          disabled={!canBuild}
          className={`w-full px-4 py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
            canBuild
              ? 'bg-purple-500/30 border border-purple-500/50 text-purple-100 hover:bg-purple-500/40'
              : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
          }`}
        >
          <Hammer size={14} />
          {building ? 'Building…' : `Build cart (${passages.length} passage${passages.length === 1 ? '' : 's'})`}
        </button>
        {!canBuild && !building && (
          <div className="text-[10px] text-slate-600 italic mt-1.5 text-center">
            {prerequisiteIssue
              ? prerequisiteIssue
              : passages.length === 0
                ? 'Add at least one passage above to build the cart.'
                : null}
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PassageBrowser — paginated, filterable list of active passages in the
// mounted cart. Click a row to populate Update + Delete IDX fields above so
// the user doesn't have to memorize pattern indices.
//
// Page size 25, viewport ~6 rows visible at a time (overflow-y-auto inside
// max-h). Filter is a case-insensitive substring search on passage text,
// executed server-side so it works against the full cart not just the
// current page. Pagination by Prev/Next arrows + jump-to-page input.
//
// Refresh triggers: filter change (300ms debounce), offset change, mount
// switch (mountedKey changes), and any external action that changes the
// active passage count (patternCount changes — fires on add/delete/restore).
// ─────────────────────────────────────────────────────────────────────────────

const PASSAGE_PAGE_SIZE = 25

function PassageBrowser({
  patternCount,
  mountedKey,
  onPickIdx,
}: {
  patternCount: number
  mountedKey: string
  onPickIdx: (idx: number) => void
}) {
  const [items, setItems] = useState<PatternListItem[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [filterDraft, setFilterDraft] = useState('')
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pageJump, setPageJump] = useState('')

  const totalPages = Math.max(1, Math.ceil(total / PASSAGE_PAGE_SIZE))
  const currentPage = Math.floor(offset / PASSAGE_PAGE_SIZE) + 1

  // Debounce the filter input so we don't refetch on every keystroke.
  useEffect(() => {
    const handle = setTimeout(() => {
      setFilter(filterDraft.trim())
      setOffset(0)
    }, 300)
    return () => clearTimeout(handle)
  }, [filterDraft])

  // Reset to page 1 whenever the mounted cart changes.
  useEffect(() => {
    setOffset(0)
    setFilterDraft('')
    setFilter('')
  }, [mountedKey])

  // Fetch when offset / filter / mount / patternCount changes.
  useEffect(() => {
    if (!mountedKey) {
      setItems([])
      setTotal(0)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    api
      .listPatterns(offset, PASSAGE_PAGE_SIZE, filter || undefined)
      .then((resp) => {
        if (cancelled) return
        setItems(resp.passages)
        setTotal(resp.total)
        // If we paged past the end (e.g., filter changed to a shorter list),
        // bounce back to page 1.
        if (resp.passages.length === 0 && resp.total > 0) {
          setOffset(0)
        }
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [mountedKey, offset, filter, patternCount])

  const goPrev = () => setOffset(Math.max(0, offset - PASSAGE_PAGE_SIZE))
  const goNext = () => setOffset(Math.min((totalPages - 1) * PASSAGE_PAGE_SIZE, offset + PASSAGE_PAGE_SIZE))
  const goPage = (page: number) => {
    const clamped = Math.max(1, Math.min(totalPages, page))
    setOffset((clamped - 1) * PASSAGE_PAGE_SIZE)
  }

  if (!mountedKey) return null

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/30">
      <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between gap-3">
        <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
          <List size={12} />
          Passages
          <span className="text-slate-400 font-mono normal-case tracking-normal">
            ({total.toLocaleString()}{filter ? ` matching "${filter}"` : ''})
          </span>
        </h2>
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          {loading && <Loader2 size={11} className="animate-spin text-slate-400" />}
          <span>click a row to load its IDX into Update + Delete</span>
        </div>
      </div>

      {/* Filter input */}
      <div className="px-4 py-2 border-b border-slate-800 flex items-center gap-2">
        <Search size={11} className="text-slate-500 shrink-0" />
        <input
          type="text"
          value={filterDraft}
          onChange={(e) => setFilterDraft(e.target.value)}
          placeholder="Filter by substring (case-insensitive)…"
          className="flex-1 bg-transparent text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none"
        />
        {filterDraft && (
          <button
            onClick={() => setFilterDraft('')}
            className="text-slate-500 hover:text-slate-300 p-0.5"
            title="Clear filter"
          >
            <X size={11} />
          </button>
        )}
      </div>

      {/* List viewport — fixed max-height with overflow-y-auto so ~6 rows
          show at a time and the user scrolls within the page. */}
      <div className="max-h-[260px] overflow-y-auto divide-y divide-slate-800">
        {error ? (
          <div className="px-4 py-3 text-xs text-rose-300 flex items-center gap-2">
            <AlertCircle size={12} className="text-rose-400 shrink-0" />
            {error}
          </div>
        ) : items.length === 0 && !loading ? (
          <div className="px-4 py-6 text-xs text-slate-500 italic text-center">
            {filter
              ? `No passages match "${filter}".`
              : patternCount > 0
                ? 'Page is empty — try resetting the filter.'
                : 'This cart has no passages yet. Add one via the panel above.'}
          </div>
        ) : (
          items.map((p) => (
            <button
              key={p.idx}
              onClick={() => onPickIdx(p.idx)}
              className="w-full px-4 py-2 flex items-start gap-3 hover:bg-slate-700/30 text-left transition-colors group"
              title={`Click to load #${p.idx} into Update + Delete IDX fields above`}
            >
              <span className="font-mono text-[11px] text-slate-500 w-12 shrink-0 mt-0.5">
                #{p.idx}
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-xs text-slate-200 truncate">
                  {p.title || '[empty]'}
                </div>
                {p.preview && (
                  <div className="text-[10px] text-slate-500 truncate mt-0.5">
                    {p.preview}
                  </div>
                )}
              </div>
              <span className="text-[10px] text-slate-600 font-mono shrink-0 mt-0.5">
                {p.word_count}w
              </span>
            </button>
          ))
        )}
      </div>

      {/* Pagination controls */}
      {total > PASSAGE_PAGE_SIZE && (
        <div className="px-4 py-2 border-t border-slate-800 flex items-center justify-between gap-2 text-xs text-slate-500">
          <button
            onClick={goPrev}
            disabled={offset === 0}
            className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft size={11} />
            Prev
          </button>
          <div className="flex items-center gap-2">
            <span className="font-mono">
              Page {currentPage.toLocaleString()} of {totalPages.toLocaleString()}
            </span>
            <span className="text-slate-600">·</span>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                const n = parseInt(pageJump, 10)
                if (!isNaN(n)) goPage(n)
                setPageJump('')
              }}
              className="flex items-center gap-1"
            >
              <span className="text-[10px] uppercase tracking-wider">jump</span>
              <input
                type="text"
                value={pageJump}
                onChange={(e) => setPageJump(e.target.value)}
                placeholder={String(currentPage)}
                className="w-12 rounded bg-slate-950/60 border border-slate-800 px-1.5 py-0.5 text-[11px] text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
              />
            </form>
          </div>
          <button
            onClick={goNext}
            disabled={offset + PASSAGE_PAGE_SIZE >= total}
            className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-slate-700/40 hover:text-slate-200 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Next
            <ChevronRight size={11} />
          </button>
        </div>
      )}
    </div>
  )
}
