import { lazy, Suspense } from 'react'
import { Hammer, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useCartBuilderStore } from '../store/cartBuilderStore'
import CartBrowser from './CartBrowser'

// BrowserCartBuilder lazy-loaded — pulls in transformers.js (~600KB gzip),
// pdfjs, mammoth, xlsx, npyjs, jszip, and the cart-builder-v2 pipeline.
// Users on the Search screen never pay this cost; only Cart Builder visitors
// trigger the ~1.5MB additional download.
const BrowserCartBuilder = lazy(() => import('./BrowserCartBuilder'))

// Cart Builder — v1.1 surface is browser-side (LOCAL mode). The pre-WebGPU
// legacy server-side flow (drop zone, server-managed workspace, Pattern 0
// preview panel, sticky build bar) was retired 2026-05-10 in favor of the
// single unified BrowserCartBuilder component. CLOUD mode (carts persist in
// your own cloud data store + cloud-source picker for input documents)
// lands with v1.2 alongside the OAuth substrate.
//
// See CC_unified-source-picker_2026-05.md for the three-axis architecture
// this UI eventually expresses: source × format × destination.
//
// CartBrowser at the bottom shows curated/sandbox carts as a read-only
// catalog. Click takes you to the Search screen rather than back into a
// (no-longer-existing) workspace.

export default function CartBuilderScreen() {
  const readOnlyMode = useAppStore((s) => s.status?.read_only_mode ?? false)
  const refreshBrowser = useCartBuilderStore((s) => s.refreshBrowser)

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
