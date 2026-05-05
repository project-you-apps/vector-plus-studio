import { useEffect, useRef } from 'react'
import { useAppStore } from './store/appStore'
import Header from './components/Header'
import NavRail from './components/NavRail'
import SearchToolbar from './components/SearchToolbar'
import SearchBar from './components/SearchBar'
import ResultsList from './components/ResultsList'
import PassageEditor from './components/PassageEditor'
import PassageModal from './components/PassageModal'
import MemboxPanel from './components/MemboxPanel'
import OverviewScreen from './components/OverviewScreen'
import SettingsScreen from './components/SettingsScreen'
import CartBuilderScreen from './components/CartBuilderScreen'
import CRUDScreen from './components/CRUDScreen'

// Stub placeholders for nav-rail screens introduced in 2026-05-03 reorg.
// Each will be promoted to its own component file as it gets fleshed out.
function ScreenStub({ title, body }: { title: string; body: string }) {
  return (
    <main className="flex-1 flex flex-col items-center justify-center p-6 text-center">
      <h2 className="text-2xl font-bold gradient-text mb-3">{title}</h2>
      <p className="text-sm text-slate-500 max-w-md leading-relaxed">{body}</p>
    </main>
  )
}

export default function App() {
  const { fetchStatus, status, editorOpen, activeScreen } = useAppStore()
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

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
        {activeScreen === 'search' && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {!editorOpen && <SearchToolbar />}
            <main className="flex-1 flex flex-col p-6 overflow-hidden">
              {editorOpen ? (
                <PassageEditor />
              ) : (
                <>
                  <SearchBar />
                  <div className="mt-6 flex-1 overflow-hidden flex flex-col">
                    <ResultsList />
                  </div>
                </>
              )}
            </main>
          </div>
        )}

        {activeScreen === 'overview' && <OverviewScreen />}

        {activeScreen === 'cartBuilder' && <CartBuilderScreen />}

        {activeScreen === 'crud' && <CRUDScreen />}

        {activeScreen === 'sql' && (
          <ScreenStub
            title="SQL Editor"
            body="SQL-like query editor for the mounted cartridge. Toggle alongside the natural-language search bar; ~10 commands (SELECT / INSERT / UPDATE / DELETE / MOUNT / SAVE / WHERE tags= / LIMIT / etc.). Results render in the same area as semantic search."
          />
        )}

        {activeScreen === 'settings' && <SettingsScreen />}
      </div>
      <PassageModal />
      <MemboxPanel />
    </div>
  )
}
