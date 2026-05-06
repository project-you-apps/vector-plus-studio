import { useEffect, useState } from 'react'
import {
  Pencil, Plus, Save, Trash2, RotateCcw, Lock, Unlock, FilePlus2,
  FolderOpen, Database, AlertCircle, CheckCircle2, Loader2, Info,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import CartBrowser from './CartBrowser'

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
type OpKind = 'add' | 'update' | 'delete' | 'restore'

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

  const handleUpdateSave = async () => {
    const idx = parseInt(updateIdx, 10)
    const text = updateText.trim()
    if (isNaN(idx) || !text) return
    // TODO Andy: add PUT /api/patterns/{idx} for true in-place update.
    // For now: add-new + tombstone-old (matches appStore.saveEditor).
    const addResp = await addPassage(text)
    if (!addResp.success) {
      log('update', `Add-new failed: ${addResp.message}`, false)
      return
    }
    await deleteResult(idx)
    log('update', `Updated pattern #${idx} (new entry + tombstoned old)`, true)
    setUpdateText('')
    setUpdateIdx('')
  }

  // ── Delete ──
  const handleDelete = async () => {
    const idx = parseInt(deleteIdx, 10)
    if (isNaN(idx)) return
    await deleteResult(idx)
    log('delete', `Tombstoned pattern #${idx}`, true)
    setDeleteIdx('')
  }

  const handleRestore = async (idx: number) => {
    await restoreResult(idx)
    log('restore', `Restored pattern #${idx}`, true)
  }

  // ── New cart ──
  const handleCreateNewCart = async () => {
    // TODO Andy: backend route doesn't exist yet. Options:
    //   (a) New /api/cartridges/new endpoint that creates an empty cart on disk
    //   (b) Reuse /api/forge with an empty file list (would need backend change)
    //   (c) Defer — start by mounting an existing cart, save-as later
    // For mockup: just show what the flow looks like.
    log('add', `(stub) Would create empty cart "${newCartName}" — backend route TBD`, false)
  }

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
          <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-6 space-y-4">
            <h2 className="text-sm font-medium text-slate-200 flex items-center gap-2">
              <FilePlus2 size={16} className="text-purple-400" />
              New Cart
            </h2>
            <p className="text-xs text-slate-500 leading-relaxed">
              Start with an empty cart and Add passages from scratch. For document-based ingestion,
              use Cart Builder instead.
            </p>
            <div className="flex gap-2">
              <input
                className="flex-1 rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60"
                placeholder="cart-name (alphanumeric, dashes, underscores)"
                value={newCartName}
                onChange={(e) => setNewCartName(e.target.value)}
              />
              <button
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  newCartName
                    ? 'bg-purple-500/20 border border-purple-500/40 text-purple-300 hover:bg-purple-500/30'
                    : 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
                }`}
                disabled={!newCartName}
                onClick={handleCreateNewCart}
              >
                Create empty cart
              </button>
            </div>
            <div className="text-[10px] text-amber-400/80 italic flex items-start gap-1">
              <Info size={11} className="mt-0.5 shrink-0" />
              Backend route TBD — see TODO in component. For now, mount an existing cart and use
              "Open Cart" mode.
            </div>
          </div>
        )}

        {/* Cart browser — same component embedded in Cart Builder. From Edit
            Carts, clicking a cart MOUNTS it (passage-level CRUD context),
            unlike Cart Builder which loads it into a workspace. */}
        <CartBrowser
          headerLabel="Carts available to edit"
          onCartClick={(cart) => {
            // Mount via the existing /api/cartridges/mount path (absolute path)
            useAppStore.getState().mount(cart.path)
            log('add', `Mounted ${cart.name} for editing`, true)
          }}
        />

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
                log('add', r.message, r.success)
              }}
              className="px-4 py-1.5 rounded-lg text-sm font-medium bg-purple-500/30 border border-purple-500/50 text-purple-200 hover:bg-purple-500/40"
            >
              <Save size={14} className="inline mr-1.5" />
              Save Cart
            </button>
          </div>
        )}
      </div>
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
          onClick={unmount}
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
            <span className={`uppercase tracking-wider w-16 shrink-0 ${
              e.kind === 'add' ? 'text-green-400'
                : e.kind === 'update' ? 'text-cyan-400'
                : e.kind === 'delete' ? 'text-rose-400'
                : 'text-slate-400'
            }`}>{e.kind}</span>
            <span className="flex-1 truncate text-slate-400">{e.detail}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
