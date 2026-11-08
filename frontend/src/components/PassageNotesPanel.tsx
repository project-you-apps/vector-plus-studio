import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import * as api from '../api/client'
import { getProvenanceForResults } from '../api/client'
import type { ProvenanceNote } from '../api/types'
import NoteComposer from './NoteComposer'
import NoteList from './NoteList'

/**
 * Notes for one passage in the Edit Carts list, opened ON DEMAND.
 *
 * ⭐ ANDY'S RULING, 2026-11-04: *"notes should be visible and appendable from the Edit
 * Carts tab… And I think notes should be open on entries in Edit Carts only if a button
 * is pushed on demand."* So nothing is fetched until someone asks for this row.
 *
 * ⭐⭐ WHICH IS ALSO WHY THE KEY IS FETCHED HERE RATHER THAN ADDED TO THE LIST PAYLOAD.
 * `PatternListItem` has no `content_key`, and putting one on all 25 rows would hash 25
 * passages per page turn to serve the one row a user might expand. `/api/patterns/{idx}`
 * already returns the key, and it is the same key the search path produces — one
 * implementation, fetched when the question is actually asked.
 *
 * ⛔ No delete and no edit here either. Notes are permanent and appendable by ruling; a
 * correction is a new note carrying `supersedes`, and a superseded note renders struck
 * through and KEPT. The Edit Carts screen is where a user would most expect a delete
 * button, which is exactly why its absence has to be deliberate rather than an oversight.
 */
export default function PassageNotesPanel({
  cartName,
  idx,
}: {
  cartName: string
  idx: number
}) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [key, setKey] = useState<string | null>(null)
  const [notes, setNotes] = useState<ProvenanceNote[]>([])
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    ;(async () => {
      try {
        // Two steps on purpose: the pattern fetch is what KNOWS the key, and the
        // provenance fetch is what knows the notes. Neither can answer for the other.
        const pattern = await api.getPattern(idx)
        if (cancelled) return
        const k = pattern.content_key ?? null
        setKey(k)
        if (!k) {
          // ⚠ Named, not blank. "This cart has no provenance layer" and "this panel
          // failed" look identical as an empty box, and that confusion is the thing the
          // whole feature exists to prevent.
          setError('This passage has no provenance key, so notes cannot be attached to it.')
          return
        }
        const prov = await getProvenanceForResults(cartName, [k])
        if (cancelled) return
        setNotes(prov?.by_key?.[k]?.notes ?? [])
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'could not load notes')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [cartName, idx, nonce])

  return (
    <div className="border-t border-slate-800 bg-slate-900/40 px-4 py-3">
      {loading ? (
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          <Loader2 size={11} className="animate-spin" />
          loading notes…
        </div>
      ) : error ? (
        <div className="text-[11px] text-amber-400/90">{error}</div>
      ) : (
        <>
          {notes.length === 0 ? (
            <p className="mb-2 text-[11px] italic text-slate-600">
              No one has recorded why this passage says what it says.
            </p>
          ) : (
            <NoteList notes={notes} />
          )}
          <div className="mt-2">
            <NoteComposer
              cartName={cartName}
              targetKey={key}
              existingNotes={notes}
              variant="panel"
              onSaved={() => setNonce((n) => n + 1)}
            />
          </div>
        </>
      )}
    </div>
  )
}
