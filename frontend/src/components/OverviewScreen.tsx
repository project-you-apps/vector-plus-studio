import { useEffect } from 'react'
import { Database, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'

// Overview screen — system health + cartridge inventory at a glance.
// Read-only by design; cart mounting still happens from the Search screen
// toolbar (one canonical location). This screen is for "what's here?" not
// "do something to it."

type Accent = 'purple' | 'green' | 'amber' | 'rose' | 'slate'

function StatCard({ label, value, sub, accent = 'slate' }: {
  label: string
  value: React.ReactNode
  sub?: React.ReactNode
  accent?: Accent
}) {
  const accentClass: Record<Accent, string> = {
    purple: 'text-purple-300',
    green: 'text-green-400',
    amber: 'text-amber-400',
    rose: 'text-rose-400',
    slate: 'text-slate-200',
  }
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/40 p-4">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold truncate ${accentClass[accent]}`}>{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}

function HealthRow({ ok, label, detail }: { ok: boolean; label: string; detail?: string }) {
  return (
    <div className="flex items-center gap-2 py-1.5 text-sm">
      {ok
        ? <CheckCircle2 size={14} className="text-green-400 flex-shrink-0" />
        : <AlertCircle size={14} className="text-slate-500 flex-shrink-0" />}
      <span className={`flex-1 ${ok ? 'text-slate-300' : 'text-slate-500'}`}>{label}</span>
      {detail && <span className="text-xs text-slate-500 font-mono">{detail}</span>}
    </div>
  )
}

export default function OverviewScreen() {
  const { status, cartridges, fetchStatus, fetchCartridges } = useAppStore()

  useEffect(() => {
    fetchStatus()
    fetchCartridges()
  }, [fetchStatus, fetchCartridges])

  if (!status) {
    return (
      <main className="flex-1 flex items-center justify-center">
        <div className="flex items-center gap-2 text-slate-500">
          <Loader2 size={16} className="animate-spin" />
          Loading status…
        </div>
      </main>
    )
  }

  const totalSize = cartridges.reduce((sum, c) => sum + c.size_mb, 0)

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-1">Overview</h1>
          <p className="text-sm text-slate-500">System health and cartridge inventory</p>
        </div>

        {/* Stat grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Engine"
            value={status.gpu_available ? 'GPU' : 'CPU'}
            sub={status.gpu_available ? 'Lattice physics enabled' : 'Embedding-only mode'}
            accent={status.gpu_available ? 'green' : 'amber'}
          />
          <StatCard
            label="Mounted"
            value={status.mounted_cartridge ?? <span className="italic text-slate-500 text-xl">None</span>}
            sub={status.mounted_cartridge ? `${status.pattern_count.toLocaleString()} patterns` : 'No cartridge active'}
            accent={status.mounted_cartridge ? 'purple' : 'slate'}
          />
          <StatCard
            label="Lock state"
            value={status.read_only ? 'Read-only' : 'Editable'}
            sub={status.dirty ? 'Unsaved changes' : 'Clean'}
            accent={status.read_only ? 'rose' : 'green'}
          />
          <StatCard
            label="Cartridges"
            value={cartridges.length}
            sub={`${totalSize.toFixed(1)} MB total on disk`}
          />
        </div>

        {/* System Health */}
        <div className="rounded-lg border border-slate-700 bg-slate-800/30 p-4">
          <h2 className="text-xs uppercase tracking-wider text-slate-500 mb-3">System Health</h2>
          <div className="space-y-0.5">
            <HealthRow ok={status.engine_ready} label="Engine ready" />
            <HealthRow
              ok={status.gpu_available}
              label="GPU available"
              detail={status.gpu_available ? 'physics modes enabled' : 'CPU-only mode'}
            />
            <HealthRow
              ok={status.signatures_loaded}
              label="Signatures loaded"
              detail={status.signatures_loaded ? 'pure_brain mode available' : 'unavailable'}
            />
            <HealthRow
              ok={status.physics_trained}
              label="Physics trained"
              detail={status.physics_trained ? 'smart / associate available' : 'mount a cart to train'}
            />
            {status.training_active && (
              <div className="flex items-center gap-2 py-1.5 text-sm">
                <Loader2 size={14} className="animate-spin text-amber-400 flex-shrink-0" />
                <span className="text-slate-300 flex-1">Training in progress</span>
                <span className="text-xs text-amber-400 font-mono">
                  {status.training_progress.toLocaleString()} / {status.training_total.toLocaleString()}
                </span>
              </div>
            )}
            {status.deleted_count > 0 && (
              <HealthRow
                ok={false}
                label="Tombstoned passages"
                detail={`${status.deleted_count} pending GC`}
              />
            )}
          </div>
        </div>

        {/* Cartridges list */}
        <div className="rounded-lg border border-slate-700 bg-slate-800/30 overflow-hidden">
          <div className="px-4 py-2 border-b border-slate-700 flex items-center justify-between">
            <h2 className="text-xs uppercase tracking-wider text-slate-500">
              Available Cartridges ({cartridges.length})
            </h2>
            <span className="text-xs text-slate-500">Mount from the Search screen toolbar</span>
          </div>
          {cartridges.length === 0 ? (
            <div className="p-6 text-center text-sm text-slate-500 italic">
              No cartridges found in <code className="font-mono text-slate-400">cartridges/</code>
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {cartridges.map((c) => {
                const isMounted = status.mounted_cartridge === c.name
                return (
                  <div
                    key={c.filename}
                    className={`px-4 py-2.5 flex items-center gap-3 text-sm transition-colors ${
                      isMounted ? 'bg-purple-500/5' : 'hover:bg-slate-800/40'
                    }`}
                  >
                    <Database size={14} className={isMounted ? 'text-purple-400' : 'text-slate-600'} />
                    <span className={`flex-1 truncate ${isMounted ? 'text-slate-100 font-medium' : 'text-slate-300'}`}>
                      {c.name}
                      {isMounted && (
                        <span className="ml-2 text-[10px] text-purple-400 uppercase tracking-wider">Mounted</span>
                      )}
                    </span>
                    <span className="text-xs text-slate-500 font-mono w-20 text-right">
                      {c.size_mb.toFixed(1)} MB
                    </span>
                    <div className="flex gap-1.5 w-32 justify-end">
                      {c.has_brain && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20">
                          Brain
                        </span>
                      )}
                      {c.has_signatures && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                          Sigs
                        </span>
                      )}
                      {c.has_manifest && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400 border border-slate-600/50">
                          SHA
                        </span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <p className="text-center text-xs text-slate-600 italic pt-2">
          More overview tiles (recent searches, droplet health, multi-cart sessions) coming as features land.
        </p>
      </div>
    </main>
  )
}
