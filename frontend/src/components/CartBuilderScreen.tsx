import { lazy, Suspense, useEffect, useState } from 'react'
import { Hammer, Loader2, Cpu, Image as ImageIcon, RefreshCw, Download } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import CartBrowser from './CartBrowser'
import DownloadWaitModal from './DownloadWaitModal'

// BrowserCartBuilder lazy-loaded — pulls in transformers.js (~600KB gzip),
// pdfjs, mammoth, xlsx, npyjs, jszip, and the cart-builder-v2 pipeline.
// Users on the Search screen never pay this cost; only Cart Builder visitors
// trigger the ~1.5MB additional download.
const BrowserCartBuilder = lazy(() => import('./BrowserCartBuilder'))

// Cart Builder — v1.1 surface is browser-side (LOCAL mode). The pre-WebGPU
// legacy server-side flow (drop zone, server-managed workspace, Pattern 0
// preview panel, sticky build bar) was retired in favor of the single
// unified BrowserCartBuilder component. CLOUD mode (carts persist in your
// own cloud data store + cloud-source picker for input documents) lands
// with v1.2 alongside the OAuth substrate.
//
// CartBrowser at the bottom shows curated/sandbox carts as a read-only
// catalog. Click takes you to the Search screen rather than back into a
// (no-longer-existing) workspace.

export default function CartBuilderScreen() {
  const readOnlyMode = useAppStore((s) => s.status?.read_only_mode ?? false)
  const refreshBrowser = useCartBuilderStore((s) => s.refreshBrowser)
  const detectDesktopHelper = useAppStore((s) => s.detectDesktopHelper)
  const desktopHelperState = useAppStore((s) => s.desktopHelperState)
  const openDesktopHelperPairModal = useAppStore((s) => s.openDesktopHelperPairModal)
  const detectImageBuilder = useAppStore((s) => s.detectImageBuilder)
  const imageBuilderState = useAppStore((s) => s.imageBuilderState)

  // Probe for the Desktop Cart Builder exe when the user lands on this tab.
  // 1-sec timeout inside the store action; failure is silent (badge falls
  // through to GPU/WebGPU/CPU). We fire on mount rather than app boot so
  // Search-only sessions never do a loopback probe.
  // Image Builder gets the same treatment; both probes are Promise.all-ed so
  // the two 1-sec budgets overlap (~1 sec total instead of 2). Failure is
  // silent for both — the pills fall through to their 'not-found' state.
  useEffect(() => {
    void Promise.all([detectDesktopHelper(), detectImageBuilder()])
  }, [detectDesktopHelper, detectImageBuilder])

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full space-y-5">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
            <Hammer size={28} className="text-purple-300" />
            Cart Builder
          </h1>
          <p className="text-sm text-slate-500">
            Drag-and-drop documents to build a Membot brain cartridge — entirely in your browser.
          </p>
        </div>

        {/* Desktop Helper status banner. The header BackendBadge is
            generic; Cart Builder specifically needs to show whether
            Build will route to the local exe or run in the browser.
            Detection re-fires on tab mount (see useEffect above); the
            Recheck button lets the user re-probe after starting the exe
            without leaving + returning to the tab.
            Rendered as a full-width banner because the prior corner-pill
            was easy to miss and users were confused when the header
            BackendBadge kept reading GPU. Hidden only in the 'unknown'
            pre-detection flicker window. */}
        <DesktopHelperStatusPill
          state={desktopHelperState}
          onPair={openDesktopHelperPairModal}
          onRecheck={() => void detectDesktopHelper()}
        />
        {/* Image Builder status pill (Day 2). Distinct from the Desktop
            Helper pill so users can see at a glance which Builders are
            available — Cart Builder can build without Image Builder as long
            as they don't drop any images or scanned PDFs. */}
        <ImageBuilderStatusPill
          state={imageBuilderState}
          onRecheck={() => void detectImageBuilder()}
        />

        {/* Browser-side cart builder. WebGPU pipeline with WASM fallback,
            self-contained: parses → chunks → embeds → packages → downloads. */}
        <Suspense
          fallback={
            <div className="rounded-xl border border-slate-700 bg-slate-800/30 p-5 flex items-center gap-3 text-sm text-slate-400">
              <Loader2 size={16} className="animate-spin text-purple-300" />
              Loading cart builder…
            </div>
          }
        >
          <BrowserCartBuilder />
        </Suspense>

        {/* Cart browser embedded as a read-only catalog. In v1.1 this surfaces
            curated + sandbox carts. Clicks route to the Search screen rather
            than into a (now-retired) server-side workspace. */}
        <CartBrowser
          headerLabel="Available Carts"
          onCartClick={(cart) => {
            // No more server-side workspace to load into. Send the user to
            // Search with a toast suggesting they mount the cart there.
            useCartBuilderStore.getState().pushToast(
              'info',
              `To search ${cart.name || 'this cart'}, mount it from the Search screen.`,
              4000,
            )
            if (readOnlyMode) {
              return
            }
            // Refresh in case the catalog changed (e.g., uploaded carts in
            // the local-server case).
            refreshBrowser()
          }}
        />
      </div>

      {/* FolderPickerModal lives at App level (App.tsx) — it's store-driven
          and used by CartBrowser from both Cart Builder AND Edit Carts. */}
    </main>
  )
}

// Full-width banner telling the user whether Build will run natively on
// their GPU via the paired Desktop Cart Builder exe, or fall back to the
// browser. Color-coded per state and left-anchored with a Recheck button
// on the right so users can re-probe after starting the exe. Hidden in
// 'unknown' state to avoid flashing before detection fires.
function DesktopHelperStatusPill({
  state,
  onPair,
  onRecheck,
}: {
  state: 'unknown' | 'detecting' | 'not-found' | 'detected-unpaired' | 'detected-paired'
  onPair: () => void
  onRecheck: () => void
}) {
  if (state === 'unknown') return null

  // Shared banner shell — colors + interior swapped per state below.
  const bannerBase = 'flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl border w-full'

  if (state === 'detecting') {
    return (
      <div className={`${bannerBase} bg-slate-800/60 border-slate-700/50 text-slate-300`}>
        <div className="flex items-center gap-2 text-sm">
          <Loader2 size={14} className="animate-spin text-slate-400" />
          <span>Checking for Desktop Helper…</span>
        </div>
      </div>
    )
  }

  if (state === 'detected-paired') {
    return (
      <div
        className={`${bannerBase} bg-purple-500/10 border-purple-500/50`}
        title="Builds run natively on your GPU via the local exe."
      >
        <div className="flex items-center gap-2 text-sm">
          <Cpu size={14} className="text-purple-300" />
          <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
          <span className="font-semibold text-purple-200">Desktop Helper: Connected</span>
          <span className="text-purple-300/70">— builds run on your GPU</span>
        </div>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    )
  }

  if (state === 'detected-unpaired') {
    return (
      <div className={`${bannerBase} bg-amber-500/10 border-amber-500/50`}>
        <button
          type="button"
          onClick={onPair}
          className="flex items-center gap-2 text-sm hover:opacity-80 transition-opacity"
          title="Desktop Cart Builder detected on this machine — click to pair"
        >
          <Cpu size={14} className="text-amber-400" />
          <span className="w-2 h-2 rounded-full bg-amber-400" />
          <span className="font-semibold text-amber-300">Desktop Helper: Detected</span>
          <span className="text-amber-300/70">(click to pair)</span>
        </button>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    )
  }

  // state === 'not-found' — running in browser only. Kept prominent so
  // Andy knows why the header BackendBadge shows GPU/WebGPU instead of
  // DESKTOP HELPER (2026-07-02 confusion).
  return (
    <div
      className={`${bannerBase} bg-slate-800/60 border-slate-700/50`}
      title="No Desktop Cart Builder detected on 127.0.0.1:7878 — builds run in your browser via WebGPU/WASM."
    >
      <div className="flex items-center gap-2 text-sm">
        <Cpu size={14} className="text-slate-400" />
        <span className="w-2 h-2 rounded-full bg-slate-500" />
        <span className="font-semibold text-slate-300">Desktop Helper: Not detected</span>
        <span className="text-slate-500">— running in browser (WebGPU/WASM)</span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <DownloadBuildersButton />
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
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-slate-300 hover:text-slate-100 hover:bg-slate-700/40 transition-colors shrink-0"
      title="Re-probe 127.0.0.1:7878 for the Desktop Cart Builder exe"
    >
      <RefreshCw size={12} />
      Recheck
    </button>
  )
}

// Vector+ Suite download link (2026-07-11). Surfaced on the "Not detected"
// pill state for both Cart Builder and Image Builder so new users have an
// obvious CTA to install the local exes. Points at
// `${BASE_URL}downloads/vps-suite.zip` — the actual zip is scp'd to the
// droplet's dist/downloads/ folder, so nginx's SPA static-first fallback
// serves it directly without any route-map change. Contextual placement per
// [[project_serve_builders_from_droplet_2026-07-08]] — download button on
// the pill beats a nav-bar link because the affordance sits at the exact
// confusion point.
function DownloadBuildersButton() {
  // Vite substitutes BASE_URL at build time — dev = "/", prod = "/vps/app/".
  // Constructing the URL this way means the same code works in both
  // environments without an env-aware branch.
  const href = `${import.meta.env.BASE_URL}downloads/vps-suite.zip`
  // Local modal state — a ~1.5 GB download takes 5-20 min, so on click we
  // pop a "meanwhile, prepare your machine" checklist to keep the user
  // engaged and pre-stage the install steps. We do NOT preventDefault on
  // the anchor click: the browser's download behavior fires normally and
  // the modal sits on top.
  const [showModal, setShowModal] = useState(false)
  return (
    <>
      <a
        href={href}
        download="vps-suite.zip"
        onClick={() => setShowModal(true)}
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-purple-200 border border-purple-500/40 bg-purple-500/10 hover:bg-purple-500/20 hover:text-purple-100 transition-colors shrink-0"
        title="Download the Vector+ Suite (~1.5 GB zip). Unzip and follow the README instructions inside."
      >
        <Download size={12} />
        Download builders
      </a>
      <DownloadWaitModal open={showModal} onClose={() => setShowModal(false)} />
    </>
  )
}

// Image Builder status pill (Day 2). Same banner shape as
// DesktopHelperStatusPill but on port 7879. Simpler than the Cart Builder
// pill because Image Builder shares Cart Builder's pairing token — there's
// no separate pair modal to open. When neither Builder is paired yet, the
// pill just reads "Detected" (color: amber) and pairing happens via the
// Desktop Helper flow (same token unlocks both).
function ImageBuilderStatusPill({
  state,
  onRecheck,
}: {
  state: 'unknown' | 'detecting' | 'not-found' | 'detected-unpaired' | 'detected-paired'
  onRecheck: () => void
}) {
  if (state === 'unknown') return null

  const bannerBase = 'flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl border w-full'

  if (state === 'detecting') {
    return (
      <div className={`${bannerBase} bg-slate-800/60 border-slate-700/50 text-slate-300`}>
        <div className="flex items-center gap-2 text-sm">
          <Loader2 size={14} className="animate-spin text-slate-400" />
          <span>Checking for Image Builder…</span>
        </div>
      </div>
    )
  }

  if (state === 'detected-paired') {
    return (
      <div
        className={`${bannerBase} bg-purple-500/10 border-purple-500/50`}
        title="Images + scanned PDFs will be OCR'd by the local Image Builder exe (127.0.0.1:7879)."
      >
        <div className="flex items-center gap-2 text-sm">
          <ImageIcon size={14} className="text-purple-300" />
          <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
          <span className="font-semibold text-purple-200">Image Builder: Connected</span>
          <span className="text-purple-300/70">— OCR routes to your local exe</span>
        </div>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    )
  }

  if (state === 'detected-unpaired') {
    return (
      <div
        className={`${bannerBase} bg-amber-500/10 border-amber-500/50`}
        title="Image Builder is running but not paired. Pair the Desktop Cart Builder above — Image Builder shares the same token."
      >
        <div className="flex items-center gap-2 text-sm">
          <ImageIcon size={14} className="text-amber-400" />
          <span className="w-2 h-2 rounded-full bg-amber-400" />
          <span className="font-semibold text-amber-300">Image Builder: Detected</span>
          <span className="text-amber-300/70">(pair Desktop Helper to unlock)</span>
        </div>
        <RecheckButton onRecheck={onRecheck} />
      </div>
    )
  }

  // 'not-found' — running without an Image Builder exe. Cart Builder still
  // works for text files; only image / scanned-PDF drops will trigger the
  // fallback dialog.
  return (
    <div
      className={`${bannerBase} bg-slate-800/60 border-slate-700/50`}
      title="No Image Builder detected on 127.0.0.1:7879. Text builds work fine; image + scanned-PDF drops will surface a fallback prompt at Build time."
    >
      <div className="flex items-center gap-2 text-sm">
        <ImageIcon size={14} className="text-slate-400" />
        <span className="w-2 h-2 rounded-full bg-slate-500" />
        <span className="font-semibold text-slate-300">Image Builder: Not running</span>
        <span className="text-slate-500">— image / scanned-PDF drops need it</span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <DownloadBuildersButton />
        <RecheckButton onRecheck={onRecheck} />
      </div>
    </div>
  )
}
