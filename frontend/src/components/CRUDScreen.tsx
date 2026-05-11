import { useEffect, useRef, useState } from 'react'
import {
  Pencil, Plus, Save, Trash2, RotateCcw, Lock, Unlock, FilePlus2,
  FolderOpen, Folder, Database, AlertCircle, CheckCircle2, Loader2, Info,
  Hammer, ChevronRight, X,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import CartBrowser from './CartBrowser'
import ConfirmDialog, { type ConfirmState } from './ConfirmDialog'
import {
  buildCartFromPassages,
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

  const [mode, setMode] = useState<Mode>('open')
  const [activity, setActivity] = useState<ActivityEntry[]>([])
  const [confirm, setConfirm] = useState<ConfirmState | null>(null)

  // Add-panel state
  const [addText, setAddText] = useState('')

  // Update-panel state
  const [updateIdx, setUpdateIdx] = useState('')
  const [updateText, setUpdateText] = useState('')
  const [updateLoading, setUpdateLoading] = useState(false)

  // Delete-panel state
  const [deleteIdx, setDeleteIdx] = useState('')

  // New-cart state
  const [newCartName, setNewCartName] = useState('')

  useEffect(() => {
    fetchCartridges()
    fetchDeleted()
  }, [fetchCartridges, fetchDeleted])

  const mounted = !!status?.mounted_cartridge
  const readOnly = !!status?.read_only
  const dirty = !!status?.dirty
  const patternCount = status?.pattern_count ?? 0

  const log = (kind: OpKind, detail: string, ok: boolean) => {
    setActivity((prev) => [
      { ts: new Date().toLocaleTimeString(), kind, detail, ok },
      ...prev.slice(0, 19),
    ])
  }

  // ── Add ──
  const handleAdd = async () => {
    const text = addText.trim()
    if (!text) return
    const resp = await addPassage(text)
    log('add', resp.message, resp.success)
    if (resp.success) {
      setAddText('')
      fetchStatus()
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
  const handleDelete = () => {
    const idx = parseInt(deleteIdx, 10)
    if (isNaN(idx)) return
    setConfirm({
      title: `Tombstone pattern #${idx}?`,
      body: (
        <>
          <p className="mb-2">
            This marks pattern <code className="font-mono text-rose-300">#{idx}</code> as deleted.
            It stops appearing in search results and can be restored from the tombstoned list below.
          </p>
          <p className="text-xs text-slate-500">
            Permanent removal happens when you Save the cart (the on-disk data is overwritten).
            Until then, Restore brings it back.
          </p>
        </>
      ),
      confirmLabel: 'Tombstone',
      destructive: true,
      onConfirm: async () => {
        await deleteResult(idx)
        log('delete', `Tombstoned pattern #${idx}`, true)
        setDeleteIdx('')
      },
    })
  }

  const handleRestore = async (idx: number) => {
    await restoreResult(idx)
    log('restore', `Restored pattern #${idx}`, true)
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
              {/* Add panel */}
              <OpPanel
                icon={Plus}
                title="Add"
                accent="green"
                disabled={readOnly}
                disabledReason="Cart is read-only — unlock first"
              >
                <textarea
                  className="w-full h-32 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 resize-none font-mono focus:outline-none focus:border-purple-500/60"
                  placeholder="New passage text…"
                  value={addText}
                  onChange={(e) => setAddText(e.target.value)}
                  disabled={readOnly}
                />
                <button
                  className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    addText.trim() && !readOnly
                      ? 'bg-green-500/20 border border-green-500/40 text-green-300 hover:bg-green-500/30'
                      : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                  }`}
                  disabled={!addText.trim() || readOnly}
                  onClick={handleAdd}
                >
                  Add Passage
                </button>
              </OpPanel>

              {/* Update panel */}
              <OpPanel
                icon={Save}
                title="Update"
                accent="cyan"
                disabled={readOnly}
                disabledReason="Cart is read-only — unlock first"
              >
                <div className="flex gap-2">
                  <input
                    className="w-24 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
                    placeholder="idx"
                    value={updateIdx}
                    onChange={(e) => setUpdateIdx(e.target.value)}
                    disabled={readOnly}
                  />
                  <button
                    className="flex-1 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 disabled:opacity-50"
                    disabled={!updateIdx || readOnly || updateLoading}
                    onClick={handleUpdateLoad}
                  >
                    {updateLoading ? <Loader2 size={12} className="animate-spin inline" /> : 'Load'}
                  </button>
                </div>
                <textarea
                  className="w-full h-24 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 resize-none font-mono focus:outline-none focus:border-purple-500/60"
                  placeholder="(load a passage by idx, or paste new text)"
                  value={updateText}
                  onChange={(e) => setUpdateText(e.target.value)}
                  disabled={readOnly}
                />
                <button
                  className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    updateIdx && updateText.trim() && !readOnly
                      ? 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30'
                      : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                  }`}
                  disabled={!updateIdx || !updateText.trim() || readOnly}
                  onClick={handleUpdateSave}
                  title="Saves new text + tombstones the old idx (until backend gets a true PUT route)"
                >
                  Update Passage
                </button>
              </OpPanel>

              {/* Delete panel */}
              <OpPanel
                icon={Trash2}
                title="Delete"
                accent="rose"
                disabled={readOnly}
                disabledReason="Cart is read-only — unlock first"
              >
                <div className="text-xs text-slate-500 leading-relaxed">
                  Tombstone a passage by index. The pattern stays on disk until you Save the cart;
                  Restore from the list below to undo.
                </div>
                <input
                  className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
                  placeholder="Pattern idx"
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
                  Tombstone Pattern
                </button>
              </OpPanel>
            </div>

            {/* Tombstoned list */}
            <div className="rounded-lg border border-slate-700 bg-slate-800/30">
              <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between">
                <h2 className="text-xs uppercase tracking-wider text-slate-500 flex items-center gap-2">
                  <Trash2 size={12} />
                  Tombstoned ({deletedPatterns.length})
                </h2>
                <span className="text-[10px] text-slate-600">
                  Save the cart to GC permanently
                </span>
              </div>
              {deletedPatterns.length === 0 ? (
                <div className="p-4 text-center text-xs text-slate-600 italic">
                  No tombstoned passages.
                </div>
              ) : (
                <div className="divide-y divide-slate-800">
                  {deletedPatterns.map((p) => (
                    <div key={p.idx} className="px-4 py-2 flex items-center gap-3 text-sm">
                      <span className="text-xs text-slate-500 font-mono w-12 shrink-0">#{p.idx}</span>
                      <span className="flex-1 truncate text-slate-400">{p.title}</span>
                      <button
                        onClick={() => handleRestore(p.idx)}
                        className="px-2 py-1 rounded text-xs bg-slate-700/50 hover:bg-slate-600 text-slate-300 hover:text-slate-100 flex items-center gap-1"
                      >
                        <RotateCcw size={11} /> Restore
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
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

        {/* Footer save bar — when dirty */}
        {dirty && mounted && (
          <div className="sticky bottom-0 -mx-6 px-6 py-3 border-t border-purple-500/30 bg-slate-900/95 backdrop-blur flex items-center justify-between">
            <div className="text-sm text-amber-300 flex items-center gap-2">
              <AlertCircle size={14} />
              Unsaved changes — tombstones and adds aren't on disk yet.
            </div>
            <button
              onClick={async () => {
                const r = await saveCartridge()
                log('save', r.message, r.success)
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
    if (!mounted) {
      return (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 flex items-start gap-3">
          <AlertCircle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <div className="text-amber-200 font-medium mb-1">No cart mounted</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              Edit Carts is the destructive screen — passages get tombstoned, updated, and deleted here.
              Mount a cart from the <strong className="text-slate-200">Search</strong> screen first
              (the toolbar there has the file picker, upload, and the cart list), then come back to edit.
              Or pick from <strong className="text-slate-200">My Carts</strong> below if you already
              know which one you want.
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-3 flex items-center gap-3">
        <Database size={16} className="text-purple-400 shrink-0" />
        <div className="flex-1">
          <div className="text-sm text-slate-200 font-medium">{status?.mounted_cartridge}</div>
          <div className="text-[11px] text-slate-500">{patternCount.toLocaleString()} patterns · {dirty ? 'unsaved changes' : 'clean'}</div>
        </div>
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
        <button
          onClick={async () => {
            const name = status?.mounted_cartridge || 'cart'
            await unmount()
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const openFolderPicker = useCartBuilderStore((s) => s.openFolderPicker)
  const refreshBrowser = useCartBuilderStore((s) => s.refreshBrowser)
  const mountCart = useAppStore((s) => s.mount)

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
    try {
      const sanitizedName = cartName.trim().replace(/[^A-Za-z0-9_-]/g, '_') || 'new-cart'
      // Write the cart bundle to the server-side folder. The server honors
      // the cart's own permissions sidecar (no forced read-only), so the
      // mounted cart is editable immediately.
      const resp = await cb.buildToFolder({
        cartBlob: result.cartBlob,
        manifestBlob: result.manifestBlob,
        permissionsBlob: result.permissionsBlob,
        folder: destFolder,
        cartName: sanitizedName,
      })
      log('save', `Saved ${resp.mounted_filename} to ${resp.folder}`, true)

      // Auto-mount + switch to Open Cart mode so the user can start editing
      // immediately. Refresh the cart browser so the new cart appears in
      // the catalog at the bottom.
      try {
        await mountCart(resp.cart_path)
        log('mount', `Mounted ${resp.mounted_filename} for editing`, true)
        await refreshBrowser()
        setMode('open')
      } catch (mountErr) {
        const m = mountErr instanceof Error ? mountErr.message : String(mountErr)
        log('mount', `Cart saved but auto-mount failed: ${m}`, false)
      }
    } catch (err) {
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
              title={`Save to ${destFolder} and mount for editing.`}
            >
              <Save size={12} />
              Save + mount for editing
            </button>
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
