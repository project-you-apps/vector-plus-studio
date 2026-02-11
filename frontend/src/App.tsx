import { useEffect, useRef } from 'react'
import { useAppStore } from './store/appStore'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import SearchBar from './components/SearchBar'
import ResultsList from './components/ResultsList'
import PassageEditor from './components/PassageEditor'

export default function App() {
  const { fetchStatus, status, editorOpen } = useAppStore()
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
        <Sidebar />
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
    </div>
  )
}
