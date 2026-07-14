import { useEffect, useRef } from 'react'
import { useAppStore } from './store/appStore'
import { useAuthStore } from './store/authStore'
import Header from './components/Header'
import NavRail from './components/NavRail'
import SearchToolbar from './components/SearchToolbar'
import SearchBar from './components/SearchBar'
import ResultsList from './components/ResultsList'
import Pattern0TocPanel from './components/Pattern0TocPanel'
import PassageEditor from './components/PassageEditor'
import PassageModal from './components/PassageModal'
import MemboxPanel from './components/MemboxPanel'
import OverviewScreen from './components/OverviewScreen'
import SettingsScreen from './components/SettingsScreen'
import CartBuilderScreen from './components/CartBuilderScreen'
import CRUDScreen from './components/CRUDScreen'
import ReportsScreen from './components/ReportsScreen'
import AgentsScreen from './components/AgentsScreen'
import SQLEditorScreen from './components/SQLEditorScreen'
import FolderPickerModal from './components/FolderPickerModal'
import SignInModal from './components/SignInModal'
import DesktopHelperPairModal from './components/DesktopHelperPairModal'
import CookieBanner from './components/CookieBanner'
import Toaster from './components/Toaster'

// Search screen layout — Andy 2026-07-01, revised 2026-07-02.
// Old-flow restore (2026-07-02): TOC and results are MUTUALLY EXCLUSIVE.
// TOC visible on fresh cart mount; a search hides it and results fill the
// space; the Pattern-0 button in SearchToolbar brings the TOC back. Store
// field `showTocPanel` drives which surface renders.
// Editor path 2026-07-02: PassageEditor promoted to App-level modal overlay
// so Edit works from Edit Carts too. Search UI stays visible behind it.
function SearchScreenLayout() {
  const mountedCartridge = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const showTocPanel = useAppStore((s) => s.showTocPanel)
  const cartMounted = !!(mountedCartridge || activeLocalCart)
  const showToc = cartMounted && showTocPanel

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <SearchToolbar />
      <main className="flex-1 flex flex-col p-6 overflow-hidden w-full max-w-7xl mx-auto">
        <SearchBar />
        <div className="mt-6 flex-1 overflow-hidden flex flex-col">
          {showToc ? (
            // TOC fills the space below the search bar until the user runs a
            // search. Internal list scroll + JUMP pagination handles overflow.
            <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
              <Pattern0TocPanel />
            </div>
          ) : (
            <div className="flex-1 overflow-hidden flex flex-col">
              <ResultsList />
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default function App() {
  const { fetchStatus, status, activeScreen, detectWebGpuOnce } = useAppStore()
  const initAuth = useAuthStore((s) => s.init)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    // Bootstrap Supabase session from cookies + subscribe to auth state changes.
    initAuth()
  }, [initAuth])

  useEffect(() => {
    // Probe WebGPU once on app load. Result lives in the store and gates
    // browser-side Associate when the server lacks CUDA. Safe to call repeatedly;
    // the underlying detector dedupes on the first call.
    void detectWebGpuOnce()
  }, [detectWebGpuOnce])

  useEffect(() => {
    fetchStatus()
    // Poll status every 2 seconds (for training progress)
    intervalRef.current = setInterval(fetchStatus, 2000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchStatus])

  // Slow down polling once not training
  useEffect(() => {
    if (status && !status.training_active && intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = setInterval(fetchStatus, 10000)
    }
  }, [status?.training_active, fetchStatus, status])

  // Warn before closing tab/window if there are unsaved changes
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (useAppStore.getState().status?.dirty) {
        e.preventDefault()
      }
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [])

  return (
    <div className="h-screen flex flex-col">
      <Header />

      {/* Global training-progress banner. Surfaces below the header when
          status.training_active so it's visible regardless of which screen
          you're on. Replaces the sidebar's Training section that was killed
          in the 2026-05-04 reorg. */}
      {status?.training_active && (
        <div className="px-6 py-1.5 border-b border-amber-500/30 bg-amber-500/5 flex items-center gap-3 flex-shrink-0">
          <span className="text-[10px] uppercase tracking-wider text-amber-400 font-semibold flex-shrink-0">Training</span>
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden max-w-md">
            <div
              className="h-full gradient-bg transition-all duration-500"
              style={{ width: `${status.training_total > 0 ? (status.training_progress / status.training_total) * 100 : 0}%` }}
            />
          </div>
          <span className="text-xs text-amber-300 font-mono flex-shrink-0">
            {status.training_progress.toLocaleString()} / {status.training_total.toLocaleString()}
          </span>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <NavRail />

        {/* Search screen — original VPS 1.0 experience, with cart picker +
            search mode now in the SearchToolbar at the top. Sidebar killed
            in the 2026-05-04 reorg; Build Cartridge moved to its own screen,
            Add Passage and Tombstoned restore deferred to the CRUD planning
            session. */}
        {activeScreen === 'search' && <SearchScreenLayout />}

        {activeScreen === 'overview' && <OverviewScreen />}

        {activeScreen === 'cartBuilder' && <CartBuilderScreen />}

        {activeScreen === 'crud' && <CRUDScreen />}

        {activeScreen === 'reports' && <ReportsScreen />}

        {activeScreen === 'agents' && <AgentsScreen />}

        {activeScreen === 'sql' && <SQLEditorScreen />}

        {activeScreen === 'settings' && <SettingsScreen />}
      </div>
      <PassageModal />
      <MemboxPanel />
      {/* Folder picker is store-driven and used by CartBrowser regardless of
          which screen the user is on (Cart Builder OR Edit Carts). Mounted at
          app level so it renders no matter which screen owns the trigger. */}
      <FolderPickerModal />
      {/* PassageEditor promoted to App-level modal 2026-07-02 so Edit works
          from both Search + Edit Carts. Self-guards on editorOpen. */}
      <PassageEditor />
      <SignInModal />
      <DesktopHelperPairModal />
      <CookieBanner />
      <Toaster />
    </div>
  )
}
