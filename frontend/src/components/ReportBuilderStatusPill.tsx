// Report Builder status pill — Reports + Agents tabs.
//
// Same 3-state banner as ImageBuilderStatusPill (Cart Builder tab):
//   - detected-paired    → violet, "Connected — local carts route here"
//   - detected-unpaired  → amber, "Detected (pair via Cart Builder pill)"
//   - not-found          → slate, "Not running — Download builders"
// Detection re-fires when the caller passes onRecheck; the outer screen
// calls detectReportBuilder() on mount so a fresh visit re-probes.
//
// Pairing is delegated to the Desktop Helper flow (Cart Builder tab)
// because the token is shared across all three Builders — one pairing
// unlocks Cart Builder + Image Builder + Report Builder. This keeps the
// pair modal in one place; the pill on Reports/Agents tabs just tells
// users where to find it.

import { HardDrive, Loader2, RefreshCw, Download } from 'lucide-react'
import type { ReportBuilderState } from '../store/appStore'

interface Props {
  state: ReportBuilderState
  onRecheck: () => void
}

// Download-builders link — same target as CartBuilderScreen's variant.
// Left as a plain external link so the file downloads via the browser's
// default handler (no download-wait modal here; that's a Cart-Builder
// tab affordance where the wait matters for the pipeline).
const BUILDERS_DOWNLOAD_URL = 'https://project-you.app/downloads/vps-suite.zip'

export default function ReportBuilderStatusPill({ state, onRecheck }: Props) {
  if (state === 'unknown') return null

  const bannerBase =
    'flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl border w-full'

  if (state === 'detecting') {
    return (
      <div className={`${bannerBase} bg-slate-800/60 border-slate-700/50 text-slate-300`}>
        <div className="flex items-center gap-2 text-sm">
          <Loader2 size={14} className="animate-spin text-slate-400" />
          <span>Checking for Report Builder…</span>
        </div>
      </div>
    )
  }

  if (state === 'detected-paired') {
    return (
      <div
        className={`${bannerBase} bg-violet-500/10 border-violet-500/50`}
        title="Reports + Agents on local: carts route to the local exe (127.0.0.1:7880). Your cart data stays on this machine; only anonymized prompts + retrieved context leave for LLM synthesis."
      >
        <div className="flex items-center gap-2 text-sm">
          <HardDrive size={14} className="text-violet-300" />
          <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
          <span className="font-semibold text-violet-200">Report Builder: Connected</span>
          <span className="text-violet-300/70">— local carts execute on this machine</span>
        </div>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    )
  }

  if (state === 'detected-unpaired') {
    return (
      <div
        className={`${bannerBase} bg-amber-500/10 border-amber-500/50`}
        title="Report Builder is running but not paired. Pair the Desktop Cart Builder on the Cart Builder tab — all three Builders share the same token."
      >
        <div className="flex items-center gap-2 text-sm">
          <HardDrive size={14} className="text-amber-400" />
          <span className="w-2 h-2 rounded-full bg-amber-400" />
          <span className="font-semibold text-amber-300">Report Builder: Detected</span>
          <span className="text-amber-300/70">(pair Desktop Helper to unlock)</span>
        </div>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    )
  }

  // 'not-found' — no local Report Builder on 7880. Local: carts fall
  // through to the droplet-side amber panel ("browser-only cart, pick a
  // server cart"). Not fatal — server carts continue to work.
  return (
    <div
      className={`${bannerBase} bg-slate-800/60 border-slate-700/50`}
      title="No Report Builder detected on 127.0.0.1:7880. Server carts work fine; local: carts need this exe to run reports + agents on your machine."
    >
      <div className="flex items-center gap-2 text-sm">
        <HardDrive size={14} className="text-slate-400" />
        <span className="w-2 h-2 rounded-full bg-slate-500" />
        <span className="font-semibold text-slate-300">Report Builder: Not running</span>
        <span className="text-slate-500">— needed for reports + agents on local: carts</span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <a
          href={BUILDERS_DOWNLOAD_URL}
          className="inline-flex items-center gap-1 text-xs text-slate-300
                     hover:text-slate-100 px-2 py-1 rounded border border-slate-700
                     hover:border-slate-500 transition-colors"
          title="Download the Vector+ Builders bundle (zip)"
        >
          <Download size={11} />
          Download builders
        </a>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    </div>
  )
}

function RecheckButton({ onRecheck }: { onRecheck: () => void }) {
  return (
    <button
      type="button"
      onClick={onRecheck}
      className="inline-flex items-center gap-1 text-xs text-slate-400
                 hover:text-slate-200 px-2 py-1 rounded border border-slate-700
                 hover:border-slate-500 transition-colors shrink-0"
      title="Re-probe 127.0.0.1:7880"
    >
      <RefreshCw size={11} />
      Recheck
    </button>
  )
}
