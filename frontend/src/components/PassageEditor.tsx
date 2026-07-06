import { useEffect } from 'react'
import { FileText, Pencil, Plus, Loader2, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'

// Promoted to an App-level modal overlay 2026-07-02 (Andy). Previously
// rendered inside SearchScreenLayout, which meant clicking Edit from the
// Edit Carts drill-down flipped editorOpen but rendered nothing on that tab.
// Modal-style overlay sits over whichever tab is active so Edit works
// consistently across Search + Edit Carts (and any future surface that
// reuses openEditor).
export default function PassageEditor() {
  const {
    editorOpen,
    editorText, editorOriginalIdx, editorOriginalText,
    addingPassage,
    setEditorText, saveEditor, closeEditor,
  } = useAppStore()

  // Escape to cancel — mirrors DesktopHelperPairModal / BriefingModal.
  useEffect(() => {
    if (!editorOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeEditor()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [editorOpen, closeEditor])

  if (!editorOpen) return null

  const isEdit = editorOriginalIdx !== null
  const hasChanges = editorText.trim() !== editorOriginalText.trim()
  const canSave = editorText.trim().length > 0 && (isEdit ? hasChanges : true)

  const handleSave = async () => {
    await saveEditor()
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-label={isEdit ? 'Edit passage' : 'Add passage'}
      onClick={(e) => { if (e.target === e.currentTarget) closeEditor() }}
    >
      <div
        className="relative w-full max-w-3xl max-h-[85vh] rounded-2xl border border-purple-500/40 bg-slate-900 shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Editor header */}
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-300">
            {isEdit ? (
              <>
                <Pencil size={18} className="text-purple-400" />
                <h2 className="text-lg font-medium">Edit Passage</h2>
                <span className="text-xs text-slate-500 ml-2">Pattern #{editorOriginalIdx}</span>
              </>
            ) : (
              <>
                <Plus size={18} className="text-green-400" />
                <h2 className="text-lg font-medium">Add Passage</h2>
              </>
            )}
          </div>
          <button
            onClick={closeEditor}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm text-slate-500 hover:text-slate-300 hover:bg-slate-700/50 transition-colors"
            title="Close (Esc)"
          >
            <X size={14} />
            Cancel
          </button>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden px-5 py-4">
          {/* Info banner for edits */}
          {isEdit && (
            <div className="mb-3 px-4 py-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-xs text-slate-400 flex items-center gap-2">
              <FileText size={14} className="text-slate-500 shrink-0" />
              <span>
                Saving changes creates a new pattern with the updated text and retires the original.
                The old version can be restored from the Tombstones panel.{' '}
                <strong className="text-slate-300">NOTE:</strong> Restoring an edited passage still
                leaves the edited version of the passage appended to the cart too, so you will see
                it also returned in search results if matched.
              </span>
            </div>
          )}

          {/* Textarea -- takes all available space */}
          <textarea
            value={editorText}
            onChange={(e) => setEditorText(e.target.value)}
            placeholder="Type or paste your passage text here..."
            className="flex-1 min-h-[240px] w-full px-4 py-3 bg-slate-800/40 border border-slate-700/50 rounded-xl
                       text-slate-200 text-sm leading-relaxed placeholder-slate-600
                       resize-none focus:outline-none focus:border-purple-500/50
                       focus:ring-1 focus:ring-purple-500/20 transition-all font-mono"
            autoFocus
          />

          {/* Footer with save button */}
          <div className="flex items-center justify-between mt-4">
            <div className="text-xs text-slate-600">
              {editorText.length.toLocaleString()} characters
              {isEdit && !hasChanges && ' (no changes)'}
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={closeEditor}
                className="px-4 py-2 rounded-lg text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={!canSave || addingPassage}
                className="px-6 py-2 rounded-lg gradient-bg text-white text-sm font-medium
                           hover:opacity-90 transition-opacity disabled:opacity-40
                           flex items-center gap-2"
              >
                {addingPassage ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Saving...
                  </>
                ) : isEdit ? (
                  'Save Changes'
                ) : (
                  'Add Passage'
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
