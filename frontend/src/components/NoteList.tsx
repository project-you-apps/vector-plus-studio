import { StickyNote } from 'lucide-react'
import type { ProvenanceNote } from '../api/types'

/**
 * Notes on a passage, rendered as margin annotations.
 *
 * ⭐ Shared by the result card and the passage modal so the SUPERSEDED treatment has one
 * implementation. That one matters more than it looks: by ruling (Andy, 2026-11-04) notes
 * are permanent and appendable, and a surface that quietly hid superseded notes instead of
 * striking them through would be destroying the record visually while the file kept it —
 * the appearance of a clean history, which is precisely the thing an audit trail must not
 * manufacture.
 *
 * Styling lives in index.css as `.provenance-note` (soft fill + heavy left edge). Not
 * dark-only Tailwind utilities: this app has no `dark:` variants, and that is what made
 * the posture chip illegible in light mode earlier the same day.
 */
/**
 * Who to show as the author. ⛔ NEVER the raw seat UUID.
 *
 * A note read "— 2301149e-e6a4-43b4-b9ef-f41c8462c1c0" on 2026-11-07, which is worse than
 * ugly on a provenance record: the entire value of a note is WHO said it, and a UUID
 * answers that with noise. It also breaks the NAMES ONLY, NEVER KEYS rule the occupancy
 * pill already follows — a seat id on screen is a disclosure to everyone who reads it.
 *
 * `by_name` is resolved server-side from a cache of seats seen since boot, so it can be
 * absent for a legitimately old note. ⚠ The fallback is a SHORT prefix, not the whole
 * UUID: enough to tell two unknown authors apart, not enough to be a usable seat id, and
 * visibly not a person's name so nobody mistakes it for one.
 */
function authorLabel(n: ProvenanceNote): string {
  const name = (n.by_name ?? '').trim()
  if (name) return name
  const seat = (n.by ?? '').trim()
  if (!seat) return 'unknown'
  if (seat === 'local-studio') return 'this install'
  return `seat ${seat.slice(0, 6)}…`
}

export default function NoteList({ notes }: { notes: ProvenanceNote[] }) {
  if (!notes || notes.length === 0) return null
  return (
    <div className="space-y-1">
      {notes.map((n) => (
        <div
          key={n.id}
          className="provenance-note flex items-start gap-1.5 rounded-r px-2 py-1 text-[11px]"
        >
          <StickyNote size={11} className="provenance-note-kind mt-0.5 shrink-0" />
          <span>
            <span className="provenance-note-kind mr-1 text-[9px] uppercase tracking-wide">
              {n.kind}
            </span>
            <span className={n.superseded ? 'line-through opacity-60' : undefined}>
              {n.body}
            </span>
            <span className="provenance-note-by ml-1.5 font-mono text-[10px]">
              — {authorLabel(n)}{n.scope === 'private' ? ' (private)' : ''}
              {n.superseded ? ' · superseded' : ''}
            </span>
          </span>
        </div>
      ))}
    </div>
  )
}
