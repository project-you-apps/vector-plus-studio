import { useEffect, useRef } from 'react'
import { useAppStore } from './store/appStore'
import Header from './components/Header'
import NavRail from './components/NavRail'
import Sidebar from './components/Sidebar'
import SearchToolbar from './components/SearchToolbar'
import SearchBar from './components/SearchBar'
import ResultsList from './components/ResultsList'
import PassageEditor from './components/PassageEditor'
import PassageModal from './components/PassageModal'
import MemboxPanel from './components/MemboxPanel'

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
      <div className="flex flex-1 overflow-hidden">
        <NavRail />

        {/* Search screen — original VPS 1.0 experience, kept intact while we
            iterate. Future passes migrate the existing Sidebar's contents into
            a per-screen toolbar at the top of the main area. */}
        {activeScreen === 'search' && (
          <>
            <Sidebar />
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
          </>
        )}

        {activeScreen === 'overview' && (
          <ScreenStub
            title="Overview"
            body="Cart stats, mounted carts, system health. Coming next — Supabase-style Project Overview equivalent (mounted cartridges + passage counts + GPU/CPU mode + droplet health if connected)."
          />
        )}

        {activeScreen === 'cartBuilder' && (
          <ScreenStub
            title="Cart Builder"
            body="Drag-and-drop cart creation from documents. The hackathon prep work for this is mostly complete — next pass ports it into this screen. Will support .txt / .pdf / .docx / .md / .jsonl ingestion with chunker config and manifest preview."
          />
        )}

        {activeScreen === 'sql' && (
          <ScreenStub
            title="SQL Editor"
            body="SQL-like query editor for the mounted cartridge. Toggle alongside the natural-language search bar; ~10 commands (SELECT / INSERT / UPDATE / DELETE / MOUNT / SAVE / WHERE tags= / LIMIT / etc.). Results render in the same area as semantic search."
          />
        )}

        {activeScreen === 'settings' && (
          <ScreenStub
            title="Settings"
            body="Search mode picker (cosine / hamming / blend / pure_brain / associate), FAST/BALANCED/QUALITY profile toggle, theme, advanced flags. Will absorb most of the current Sidebar's lower sections."
          />
        )}
      </div>
      <PassageModal />
      <MemboxPanel />
    </div>
  )
}
