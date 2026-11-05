import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ShieldCheck, Globe, ShieldAlert, HelpCircle } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { computePosture, buildStamp, type Posture } from '../lib/posture'

/**
 * WHICH BUILD am I looking at, and WHERE is it running? Lives in the nav rail, under
 * Settings.
 *
 * ⭐ WHY THE RAIL AND NOT A CORNER (Andy, 2026-11-04): *"I want it visible on the sidebar
 * in an obvious spot because of how reassuring it will be psychologically."* That is the
 * actual product requirement and it outranks tidiness — a privacy indicator a cautious
 * user has to go looking for does not do the job a privacy indicator exists to do. It sits
 * in the same column as the screens, so it reads as part of the app's furniture rather
 * than as a debug overlay.
 *
 * ⚠ Its other job is unglamorous and equally load-bearing: it makes every SCREENSHOT
 * self-describing. On 11-04 a screenshot was the only evidence for a structural conclusion
 * about a feature and carried neither the build nor the origin, so "this build predates the
 * code" and "this path cannot support it" looked identical.
 *
 * ⛔ WHAT IT DOES NOT CLAIM. Not a data-residency guarantee. See lib/posture.ts and
 * docs/SPEC-deployment-posture-and-what-it-proves-2026-11-04.md: it says the app and its
 * API are on this machine, never that no data left it.
 */

// Palette lives in index.css as `.posture-chip-*`, NOT in Tailwind utilities here.
// ⭐ WHY: this app has no `dark:` variants — light mode is an `html.light` class with
// overrides in index.css (see the input/textarea block there for the same pattern). A
// chip styled only with dark-mode Tailwind classes is exactly what produced pale green on
// pale green when Andy switched to light mode. One class, two themes, both deliberate.
const VERDICT_ICON: Record<Posture['verdict'], typeof ShieldCheck> = {
  local: ShieldCheck,
  hosted: Globe,
  mixed: ShieldAlert,
  unknown: HelpCircle,
}

/** Short enough for a 176px rail. The full sentence is in the popover and the tooltip. */
const SHORT: Record<Posture['verdict'], string> = {
  local: 'Local install',
  hosted: 'Hosted',
  mixed: 'Mixed — check this',
  unknown: 'Checking…',
}

export default function PostureBadge() {
  const status = useAppStore((s) => s.status)
  const [open, setOpen] = useState(false)
  const btnRef = useRef<HTMLButtonElement>(null)
  // Where to draw the popover, measured at click time.
  const [at, setAt] = useState<{ top: number; left: number } | null>(null)

  // ⛔ PORTALLED, NOT ABSOLUTELY POSITIONED IN THE RAIL. NavRail's parent in App.tsx is
  // `<div className="flex flex-1 overflow-hidden">`, which CLIPS an absolutely positioned
  // child — the panel would have rendered and been invisible, so clicking the chip would
  // have looked like it did nothing. A portal to <body> also survives anyone later adding
  // a transform or backdrop-filter to an ancestor, which would defeat `position: fixed`
  // too. Measured on open rather than tracked, because the rail does not scroll or move.
  const toggle = () => {
    if (!open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect()
      // Clamp so a short viewport cannot push the panel off the bottom of the screen.
      const top = Math.min(r.top, Math.max(8, window.innerHeight - 320))
      setAt({ top, left: r.right + 8 })
    }
    setOpen((v) => !v)
  }

  // ⛔ `?? null`, never `?? false`. An absent field means the server has not answered;
  // treating that as "not public" would show green for a backend we never reached.
  const posture = computePosture(status?.public_host ?? null)
  const stamp = buildStamp(posture)
  const Icon = VERDICT_ICON[posture.verdict]

  return (
    <div className="relative px-1 pt-1">
      <button
        ref={btnRef}
        type="button"
        onClick={toggle}
        title={posture.detail}
        aria-label={`Deployment posture: ${posture.label}. Build ${stamp}.`}
        aria-expanded={open}
        className={`posture-chip posture-chip-${posture.verdict} w-full flex items-start gap-2 rounded-lg border px-3 py-2 text-left transition-opacity hover:opacity-100`}
      >
        <Icon size={16} className="mt-0.5 flex-shrink-0" />
        <span className="min-w-0">
          <span className="block text-sm font-medium leading-tight">
            {SHORT[posture.verdict]}
          </span>
          <span className="posture-chip-sub block font-mono text-[10px] leading-tight">
            {stamp}
          </span>
        </span>
      </button>

      {/* Opens to the RIGHT of the rail, not below it: the rail is 176px wide and the
          detail rows are label+value pairs that would wrap into unreadability in it. */}
      {open && at && createPortal(
        <>
          {/* Click-away. Covers the viewport under the panel so the chip behaves like a
              popover rather than something you have to click twice to dismiss. */}
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
        <div
          style={{ top: at.top, left: at.left }}
          className="fixed z-50 w-80 rounded-lg border border-slate-700 bg-slate-900/95 p-3 text-xs shadow-xl backdrop-blur"
        >
          <div className="mb-2 text-sm font-semibold text-slate-100">{posture.label}</div>
          <p className="mb-3 leading-relaxed text-slate-400">{posture.detail}</p>
          <dl className="space-y-1 text-[11px]">
            <Row k="Served from" v={posture.servedFrom} src="browser-enforced" />
            <Row
              k="Backend"
              v={
                posture.backendIsPublic === null
                  ? 'not reached'
                  : posture.backendIsPublic
                    ? 'public host'
                    : 'not a public host'
              }
              src="server-asserted"
            />
            {posture.buildMode === 'serve' && posture.devProxyTarget && (
              <Row k="Dev proxy →" v={posture.devProxyTarget} src="build-time" />
            )}
            <Row
              k="Build"
              v={`${stamp}${posture.buildMode === 'serve' ? ' (dev server)' : ''}`}
              src="build-time"
            />
            {posture.buildTime && (
              <Row k="Built" v={new Date(posture.buildTime).toLocaleString()} src="build-time" />
            )}
          </dl>
          {posture.buildDirty && (
            <p className="mt-2 text-[11px] leading-relaxed text-amber-300">
              ⚠ Built from uncommitted source. The commit above does not fully describe
              what is running.
            </p>
          )}
          <p className="mt-2 border-t border-slate-700/60 pt-2 text-[10px] leading-relaxed text-slate-500">
            Describes where the app and its API are running. Not a claim that no data left
            this machine.
          </p>
        </div>
        </>,
        document.body,
      )}
    </div>
  )
}

function Row({ k, v, src }: { k: string; v: string; src: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="flex-shrink-0 text-slate-500">{k}</dt>
      <dd className="truncate text-right font-mono text-slate-300" title={v}>
        {v}
        {/* ⭐ The trust label is not decoration. Browser-enforced, server-asserted and
            build-time are not equally trustworthy, and a value whose source is unstated
            invites the reader to trust the weakest as much as the strongest. */}
        <span className="ml-2 font-sans text-slate-600">{src}</span>
      </dd>
    </div>
  )
}
