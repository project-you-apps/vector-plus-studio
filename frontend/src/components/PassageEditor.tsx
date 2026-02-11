import { FileText, Pencil, Plus, Loader2, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'

export default function PassageEditor() {
  const {
    editorText, editorOriginalIdx, editorOriginalText,
    addingPassage,
    setEditorText, saveEditor, closeEditor,
  } = useAppStore()

  const isEdit = editorOriginalIdx !== null
  const hasChanges = editorText.trim() !== editorOriginalText.trim()
  const canSave = editorText.trim().length > 0 && (isEdit ? hasChanges : true)

  const handleSave = async () => {
    await saveEditor()
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Editor header */}
      <div className="flex items-center justify-between mb-4">
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
        >
          <X size={14} />
          Cancel
        </button>
      </div>

      {/* Info banner for edits */}
      {isEdit && (
        <div className="mb-3 px-4 py-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-xs text-slate-400 flex items-center gap-2">
          <FileText size={14} className="text-slate-500 shrink-0" />
          <span>
            Saving changes will create a new pattern with the updated text and retire the original.
            The old version can be restored from the Tombstones panel.
          </span>
        </div>
      )}

      {/* Textarea -- takes all available space */}
      <textarea
        value={editorText}
        onChange={(e) => setEditorText(e.target.value)}
        placeholder="Type or paste your passage text here..."
        className="flex-1 w-full px-4 py-3 bg-slate-800/40 border border-slate-700/50 rounded-xl
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
  )
}
