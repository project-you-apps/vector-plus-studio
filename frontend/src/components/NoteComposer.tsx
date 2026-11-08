import { useState } from 'react'
import { StickyNote } from 'lucide-react'
import { addProvenanceNote } from '../api/client'
import { useAppStore } from '../store/appStore'
import type { ProvenanceNote } from '../api/types'

/**
 * "Add why" — one implementation, used by both the result card and the passage modal.
 *
 * ⭐ EXTRACTED RATHER THAN COPIED (2026-11-04). The second surface was the moment to do
 * it: two composers would have drifted on the things that matter least visibly and most
 * actually — the private-by-default rule, whether a redaction warning is shown, whether a
 * failure is surfaced at all. A note written through a surface that silently swallowed
 * its error is the exact invisible loss this feature exists to prevent, and it would have
 * been invisible in review too.
 */

export type NoteKind = ProvenanceNote['kind']

// ⚠ Must stay in step with note_store.VALID_KINDS — the server rejects an unknown kind
// with a 422, so a mismatch here is a save that fails only when someone picks the new one.
// ⭐ Ordered by how often they are reached for, not alphabetically.
const KINDS: NoteKind[] = [
  'why', 'decision', 'recommendation', 'caution', 'context', 'prediction',
]

interface Props {
  cartName: string | null | undefined
  /** The passage's server-computed key. Composer hides itself without one. */
  targetKey: string | null | undefined
  /** Called after a successful save so the caller can re-read provenance. */
  onSaved?: () => void
  /** 'inline' for the result card, 'panel' for the roomier modal. */
  variant?: 'inline' | 'panel'
  /**
   * Notes already on this passage, offered as supersede targets.
   *
   * ⛔ WITHOUT THIS THERE IS NO WAY TO CORRECT A NOTE AT ALL (found 2026-11-07). The
   * route, the client and the renderer all supported `supersedes`; only the composer
   * had no control, so every correction became a second unlinked note and the original
   * never struck through. The append-only design was complete and unreachable.
   */
  existingNotes?: ProvenanceNote[]
}

export default function NoteComposer({
  cartName, targetKey, onSaved, variant = 'inline', existingNotes,
}: Props) {
  const [open, setOpen] = useState(false)
  const [body, setBody] = useState('')
  const [kind, setKind] = useState<NoteKind>('why')
  const [shared, setShared] = useState(false)
  const [supersedes, setSupersedes] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [redacted, setRedacted] = useState(false)

  // ⚠ Advisory, not a permission — the server decides. Used only to explain BEFORE the
  // click, because a disabled-looking box with no reason is worse than a clear refusal.
  const notesWritable = useAppStore((s) => s.status?.notes_writable ?? true)
  const cartLocked = useAppStore((s) => s.status?.read_only ?? false)

  // ⛔ No key, no composer. There would be nothing to attach the note to, and an input
  // that silently discarded what was typed is worse than no input at all.
  if (!cartName || !targetKey) return null

  const save = async () => {
    setSaving(true)
    setError(null)
    setRedacted(false)
    try {
      const res = await addProvenanceNote(cartName, {
        target_key: targetKey,
        body: body.trim(),
        kind,
        scope: shared ? 'shared' : 'private',
        ...(supersedes ? { supersedes } : {}),
      })
      const hadRedactions = (res.redactions?.length ?? 0) > 0
      setRedacted(hadRedactions)
      setBody('')
      setSupersedes('')
      // Stay OPEN when text was altered, so the warning is read rather than dismissed by
      // the panel closing underneath it.
      if (!hadRedactions) setOpen(false)
      onSaved?.()
    } catch (e) {
      // The server's own message distinguishes "sign in", "your access is viewer" and
      // "this server is read-only" — three different problems a generic string flattens.
      setError(e instanceof Error ? e.message : 'could not save that note')
    } finally {
      setSaving(false)
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        disabled={!notesWritable}
        title={notesWritable
          ? 'Attach a note explaining why this passage says what it says'
          : 'This server runs read-only and cannot store notes'}
        className={`flex items-center gap-1 text-slate-500 hover:text-amber-500 disabled:cursor-not-allowed disabled:opacity-50 ${
          variant === 'panel' ? 'text-xs' : 'text-[10px]'
        }`}
      >
        <StickyNote size={variant === 'panel' ? 12 : 10} className="shrink-0" />
        <span>add why</span>
      </button>
    )
  }

  return (
    <div className="space-y-1 rounded border border-slate-700/70 bg-slate-900/60 p-2">
      <textarea
        value={body}
        onChange={(e) => setBody(e.target.value)}
        rows={variant === 'panel' ? 3 : 2}
        autoFocus
        placeholder="Why does this say what it says?"
        className="w-full resize-y rounded bg-slate-950/60 px-2 py-1 text-[11px] text-slate-200 outline-none focus:ring-1 focus:ring-amber-500/40"
      />
      <div className="flex flex-wrap items-center gap-2 text-[10px]">
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value as NoteKind)}
          className="rounded bg-slate-950/60 px-1 py-0.5 text-slate-300"
        >
          {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
        {/* ⚠ Default PRIVATE, matching note_store: a note is born private and becomes
            shared only when its author says so. Surfaced rather than buried so sharing
            is always a deliberate act. */}
        <label className="flex items-center gap-1 text-slate-400">
          <input type="checkbox" checked={shared} onChange={(e) => setShared(e.target.checked)} />
          <span>share with the team</span>
        </label>
        {/* ⭐⭐ CORRECT AN EARLIER NOTE. The whole append-only design turns on this: a
            correction must LINK to what it corrects, or it is just a second opinion
            sitting next to the first with nothing saying which is current.
            ⛔ Only offered when there is something to correct, and superseded notes are
            excluded — superseding an already-superseded note makes a chain nobody can
            read, and the current note is the one a corrector means. */}
        {(existingNotes ?? []).some((n) => !n.superseded) && (
          <select
            value={supersedes}
            onChange={(e) => setSupersedes(e.target.value)}
            title="Mark an earlier note as corrected by this one. Nothing is deleted."
            className="max-w-[12rem] truncate rounded bg-slate-950/60 px-1 py-0.5 text-slate-300"
          >
            <option value="">corrects nothing</option>
            {(existingNotes ?? []).filter((n) => !n.superseded).map((n) => (
              <option key={n.id} value={n.id}>
                corrects: {(n.body || '').slice(0, 40)}…
              </option>
            ))}
          </select>
        )}
        <button
          type="button"
          disabled={saving || body.trim().length === 0}
          onClick={save}
          className="ml-auto rounded bg-amber-600/80 px-2 py-0.5 font-medium text-white disabled:opacity-40"
        >
          {saving ? 'saving…' : 'save'}
        </button>
        <button
          type="button"
          onClick={() => { setOpen(false); setError(null) }}
          className="text-slate-500 hover:text-slate-300"
        >
          cancel
        </button>
      </div>

      {/* ⭐⭐ THE ANSWER TO "should I be able to add a why to a read-only document?"
          (Andy, 2026-11-04), said WHERE THE QUESTION ARISES. Yes — and here is why, next
          to the badge that prompted the doubt. The lock guards the document; the note
          goes beside it. Shown only when the cart is actually locked, so it reads as an
          explanation rather than boilerplate. */}
      {cartLocked && notesWritable && (
        <div className="text-[10px] leading-relaxed text-slate-500">
          This cart is locked for editing. Notes are stored <em>alongside</em> the
          document, never inside it — so the passage stays exactly as it is.
        </div>
      )}
      {/* ⛔ Errors are SHOWN. A write that failed quietly means the author's reasoning was
          never recorded while they believe it was. */}
      {error && <div className="text-[10px] text-rose-400">{error}</div>}
      {redacted && (
        <div className="text-[10px] text-amber-300">
          ⚠ Saved, but something in that note looked like a secret and was replaced with a
          marker. Check the stored text.
        </div>
      )}
    </div>
  )
}
