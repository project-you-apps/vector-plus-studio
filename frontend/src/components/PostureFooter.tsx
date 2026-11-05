import { useState } from 'react'
import { useAppStore } from '../store/appStore'
import { computePosture, buildStamp, type Posture } from '../lib/posture'

/**
 * A small always-present chip: WHICH BUILD am I looking at, and WHERE is it running?
 *
 * ⭐ WHY IT IS ALWAYS VISIBLE rather than tucked into Settings. Its first job is to make
 * every SCREENSHOT self-describing. On 11-04 a screenshot was the only evidence for a
 * structural conclusion about a feature, and it carried neither the build nor the origin
 * — so "this build predates the feature" and "this path can never support the feature"
 * looked the same. A diagnostic that has to be gone looking for does not get looked at.
 *
 * Its second job is Andy's PII/HIPAA case: one browser for both the local and the hosted
 * stack, where a cautious user can see which one they are on. See lib/posture.ts for what
 * that does and does NOT establish — in particular it is not a data-residency claim.
 */

const CHIP: Record<Posture['verdict'], string> = {
  // Green only for the coherent local case. Nothing else gets a reassuring color.
  local: 'bg-emerald-900/70 text-emerald-200 border-emerald-700/60',
  hosted: 'bg-sky-900/70 text-sky-200 border-sky-700/60',
  // ⛔ Red. A local-looking address bar with a public backend is the case a user acting
  // on a "local means private" belief would get wrong, so it is styled as a warning.
  mixed: 'bg-rose-900/80 text-rose-100 border-rose-600/70',
  unknown: 'bg-slate-800/80 text-slate-300 border-slate-600/60',
}

const DOT: Record<Posture['verdict'], string> = {
  local: 'bg-emerald-400',
  hosted: 'bg-sky-400',
  mixed: 'bg-rose-400',
  unknown: 'bg-slate-500',
}

export default function PostureFooter() {
  const status = useAppStore((s) => s.status)
  const [open, setOpen] = useState(false)

  // ⛔ `?? null`, never `?? false`. An absent field means the server has not answered (or
  // is too old to send it); treating that as "not public" would show green for a backend
  // we have never heard from.
  const posture = computePosture(status?.public_host ?? null)
  const stamp = buildStamp(posture)

  return (
    <div className="fixed bottom-2 left-2 z-40 flex flex-col items-start gap-1 text-xs">
      {open && (
        <div className="max-w-sm rounded border border-slate-700 bg-slate-900/95 p-3 shadow-lg backdrop-blur">
          <div className="mb-2 font-semibold text-slate-200">{posture.label}</div>
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
            {/* Shown only in dev: in a static build there is no proxy to describe. */}
            {posture.buildMode === 'serve' && posture.devProxyTarget && (
              <Row k="Dev proxy → " v={posture.devProxyTarget} src="build-time" />
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
        </div>
      )}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={posture.detail}
        className={`flex items-center gap-2 rounded border px-2 py-0.5 font-mono transition-opacity hover:opacity-100 ${CHIP[posture.verdict]} ${open ? '' : 'opacity-70'}`}
      >
        <span className={`inline-block h-1.5 w-1.5 rounded-full ${DOT[posture.verdict]}`} />
        <span>{posture.label}</span>
        <span className="text-slate-400">·</span>
        <span>{stamp}</span>
      </button>
    </div>
  )
}

function Row({ k, v, src }: { k: string; v: string; src: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="shrink-0 text-slate-500">{k}</dt>
      <dd className="text-right font-mono text-slate-300">
        {v}
        {/* ⭐ The trust label is not decoration. A value whose source is unstated invites
            the reader to trust all three sources equally, and they are not equal. */}
        <span className="ml-2 font-sans not-italic text-slate-600">{src}</span>
      </dd>
    </div>
  )
}
