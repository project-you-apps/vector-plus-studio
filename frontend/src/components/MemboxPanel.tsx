import { useEffect, useRef, useState } from 'react'
import { Lock, Unlock, X, Activity, Plus, Trash2, Send } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import * as api from '../api/client'
import type { MemboxCartInfo } from '../api/types'

// Stable color palette for agent badges -- assigned by hash of agent_id
const AGENT_COLORS = [
  'bg-purple-500/20 text-purple-300 border-purple-500/40',
  'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
  'bg-amber-500/20 text-amber-300 border-amber-500/40',
  'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  'bg-rose-500/20 text-rose-300 border-rose-500/40',
  'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
]

function agentColor(agentId: string): string {
  if (!agentId) return AGENT_COLORS[0]
  let hash = 0
  for (let i = 0; i < agentId.length; i++) {
    hash = (hash * 31 + agentId.charCodeAt(i)) | 0
  }
  return AGENT_COLORS[Math.abs(hash) % AGENT_COLORS.length]
}

function formatTimestamp(iso: string): string {
  if (!iso) return '?'
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return iso.substring(0, 19)
  }
}

function CartRow({ cart, selected, onClick, onUnmount }: {
  cart: MemboxCartInfo
  selected: boolean
  onClick: () => void
  onUnmount: () => void
}) {
  const { holder, is_locked, lease_seconds, held_for_seconds } = cart.lock
  const remaining = held_for_seconds != null ? Math.max(0, lease_seconds - held_for_seconds) : null
  const expiringSoon = remaining != null && remaining < lease_seconds * 0.25

  let dotColor = 'bg-emerald-500'
  if (is_locked) dotColor = expiringSoon ? 'bg-amber-500' : 'bg-rose-500'

  return (
    <div
      className={`w-full px-3 py-2 rounded-lg border transition-all cursor-pointer
        ${selected
          ? 'bg-purple-500/10 border-purple-500/40'
          : 'bg-slate-900/40 border-slate-800 hover:bg-slate-900/60 hover:border-slate-700'
        }`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className={`w-2 h-2 rounded-full ${dotColor} flex-shrink-0`} />
          <span className="font-mono text-sm text-slate-200 truncate">{cart.cart_id}</span>
        </div>
        <span className="text-xs text-slate-500 flex-shrink-0">{cart.n_patterns}p</span>
        <button
          onClick={(e) => { e.stopPropagation(); onUnmount() }}
          className="text-slate-600 hover:text-rose-400 transition-colors"
          title="Unmount this cart"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
      {(cart.role || holder) && (
        <div className="mt-1 flex items-center gap-2 text-xs">
          {cart.role && <span className="text-slate-500">{cart.role}</span>}
          {holder && (
            <span className={`px-1.5 py-0.5 rounded border ${agentColor(holder)}`}>
              {holder}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function MountForm({ onMounted }: { onMounted: () => void }) {
  const [open, setOpen] = useState(false)
  const [path, setPath] = useState('')
  const [cartId, setCartId] = useState('')
  const [role, setRole] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  const submit = async () => {
    if (!path.trim()) return
    setBusy(true)
    setMsg('')
    try {
      const r = await api.memboxMount({
        cart_path: path.trim(),
        cart_id: cartId.trim() || null,
        role: role.trim() || null,
      })
      setMsg(r.message)
      if (r.success) {
        setPath(''); setCartId(''); setRole('')
        onMounted()
        setTimeout(() => { setMsg(''); setOpen(false) }, 1500)
      }
    } catch (e: any) {
      setMsg(`error: ${e.message}`)
    } finally {
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded border border-dashed border-slate-700 text-xs text-slate-400 hover:border-purple-500/50 hover:text-purple-300 transition-colors"
      >
        <Plus className="w-3 h-3" /> mount cart via membox
      </button>
    )
  }

  return (
    <div className="space-y-1.5 p-3 rounded border border-slate-800 bg-slate-900/40">
      <input
        type="text"
        value={path}
        onChange={(e) => setPath(e.target.value)}
        placeholder="cart path (e.g. cartridges/heartbeat.cart.npz)"
        className="w-full px-2 py-1 text-xs rounded bg-[var(--chrome-bg)] border border-slate-700 text-slate-200 focus:outline-none focus:border-purple-500/50"
      />
      <div className="flex gap-1.5">
        <input
          type="text"
          value={cartId}
          onChange={(e) => setCartId(e.target.value)}
          placeholder="cart_id (optional)"
          className="flex-1 px-2 py-1 text-xs rounded bg-[var(--chrome-bg)] border border-slate-700 text-slate-200 focus:outline-none focus:border-purple-500/50"
        />
        <input
          type="text"
          value={role}
          onChange={(e) => setRole(e.target.value)}
          placeholder="role"
          className="flex-1 px-2 py-1 text-xs rounded bg-[var(--chrome-bg)] border border-slate-700 text-slate-200 focus:outline-none focus:border-purple-500/50"
        />
      </div>
      <div className="flex gap-1.5">
        <button
          onClick={submit}
          disabled={busy || !path.trim()}
          className="flex-1 px-2 py-1 text-xs rounded bg-purple-500/20 text-purple-300 border border-purple-500/40 hover:bg-purple-500/30 disabled:opacity-50"
        >
          {busy ? 'mounting...' : 'mount'}
        </button>
        <button
          onClick={() => { setOpen(false); setMsg('') }}
          className="px-2 py-1 text-xs rounded text-slate-500 hover:text-slate-300"
        >
          cancel
        </button>
      </div>
      {msg && <div className="text-xs text-slate-400">{msg}</div>}
    </div>
  )
}

function ImprintForm({ cartId }: { cartId: string }) {
  const [text, setText] = useState('')
  const [agentId, setAgentId] = useState('andy')
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!text.trim()) return
    setBusy(true)
    try {
      await api.memboxImprint({
        cart_id: cartId,
        text: text.trim(),
        agent_id: agentId.trim() || 'andy',
      })
      setText('')
    } catch (e) {
      console.error('imprint failed:', e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-1.5 p-2 rounded border border-slate-800 bg-slate-900/40">
      <div className="flex gap-1.5">
        <input
          type="text"
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          placeholder="agent_id"
          className="w-24 px-2 py-1 text-xs rounded bg-[var(--chrome-bg)] border border-slate-700 text-slate-200 focus:outline-none focus:border-purple-500/50"
        />
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="write a pattern..."
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          className="flex-1 px-2 py-1 text-xs rounded bg-[var(--chrome-bg)] border border-slate-700 text-slate-200 focus:outline-none focus:border-purple-500/50"
        />
        <button
          onClick={submit}
          disabled={busy || !text.trim()}
          className="px-2 py-1 text-xs rounded bg-purple-500/20 text-purple-300 border border-purple-500/40 hover:bg-purple-500/30 disabled:opacity-50"
        >
          <Send className="w-3 h-3" />
        </button>
      </div>
    </div>
  )
}

function LockDetail() {
  const { memboxStatus, memboxStatusLoading } = useAppStore()
  if (memboxStatusLoading && !memboxStatus) {
    return <div className="text-xs text-slate-500 italic">Loading...</div>
  }
  if (!memboxStatus) return null

  const { lock } = memboxStatus
  const remaining = lock.held_for_seconds != null
    ? Math.max(0, lock.lease_seconds - lock.held_for_seconds)
    : null
  const remainingPct = remaining != null ? (remaining / lock.lease_seconds) * 100 : 0

  let barColor = 'bg-emerald-500'
  if (remainingPct < 50) barColor = 'bg-amber-500'
  if (remainingPct < 25) barColor = 'bg-rose-500'

  const contentionPct = lock.acquire_count > 0
    ? Math.round((lock.wait_count / lock.acquire_count) * 100)
    : 0

  return (
    <div className="space-y-2 text-sm">
      <div className="flex items-center gap-2">
        {lock.is_locked ? (
          <Lock className="w-3.5 h-3.5 text-rose-400" />
        ) : (
          <Unlock className="w-3.5 h-3.5 text-emerald-400" />
        )}
        <span className="text-slate-300">
          {lock.holder ? (
            <>Held by <span className={`px-1.5 py-0.5 rounded border ${agentColor(lock.holder)}`}>{lock.holder}</span></>
          ) : (
            <span className="text-slate-500 italic">idle</span>
          )}
        </span>
      </div>

      {lock.is_locked && remaining != null && (
        <div>
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>lease</span>
            <span>{remaining.toFixed(1)}s / {lock.lease_seconds}s</span>
          </div>
          <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
            <div
              className={`h-full ${barColor} transition-all duration-200`}
              style={{ width: `${remainingPct}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>contention</span>
        <span className="font-mono">
          {lock.wait_count}/{lock.acquire_count} ({contentionPct}%)
        </span>
      </div>
    </div>
  )
}

function WriteFeed() {
  const { memboxStatus } = useAppStore()
  if (!memboxStatus) return null

  const { recent_writes, writes_by_agent } = memboxStatus
  const agents = Object.keys(writes_by_agent)
  const totalWrites = Object.values(writes_by_agent).reduce((a, b) => a + b, 0)

  return (
    <div className="space-y-3">
      {/* Per-agent summary bar */}
      {agents.length > 0 && (
        <div>
          <div className="text-xs text-slate-500 mb-1.5">writes by agent ({totalWrites} total)</div>
          <div className="flex gap-1 mb-2">
            {agents.map((agent) => {
              const count = writes_by_agent[agent]
              const pct = totalWrites > 0 ? (count / totalWrites) * 100 : 0
              const colorClass = agentColor(agent).split(' ')[0] // bg only
              return (
                <div
                  key={agent}
                  className={`h-2 rounded-sm ${colorClass}`}
                  style={{ width: `${pct}%` }}
                  title={`${agent}: ${count}`}
                />
              )
            })}
          </div>
          <div className="flex flex-wrap gap-1">
            {agents.map((agent) => (
              <span
                key={agent}
                className={`px-1.5 py-0.5 rounded border text-xs ${agentColor(agent)}`}
              >
                {agent} {writes_by_agent[agent]}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Recent writes timeline */}
      <div>
        <div className="text-xs text-slate-500 mb-1.5 flex items-center gap-1.5">
          <Activity className="w-3 h-3" />
          recent writes
        </div>
        {recent_writes.length === 0 ? (
          <div className="text-xs text-slate-600 italic">no writes yet</div>
        ) : (
          <div className="space-y-1.5 max-h-64 overflow-y-auto">
            {recent_writes.slice().reverse().map((w, i) => (
              <div key={`${w.local_addr}-${i}`} className="flex gap-2 text-xs">
                <span className={`px-1.5 py-0.5 rounded border flex-shrink-0 ${agentColor(w.agent_id)}`}>
                  {w.agent_id}
                </span>
                <span className="font-mono text-slate-500 flex-shrink-0">{formatTimestamp(w.written_at)}</span>
                <span className="text-slate-400 truncate" title={w.text_preview}>
                  #{w.local_addr} {w.text_preview}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function MemboxPanel() {
  const {
    memboxPanelOpen,
    memboxCarts,
    selectedMemboxCart,
    toggleMemboxPanel,
    fetchMemboxCarts,
    selectMemboxCart,
    fetchMemboxStatus,
  } = useAppStore()

  const pollRef = useRef<number | null>(null)

  // Poll cart list every 1s while panel is open
  useEffect(() => {
    if (!memboxPanelOpen) {
      if (pollRef.current != null) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
      return
    }
    fetchMemboxCarts()
    pollRef.current = window.setInterval(() => {
      fetchMemboxCarts()
      const cur = useAppStore.getState().selectedMemboxCart
      if (cur) fetchMemboxStatus(cur)
    }, 1000)
    return () => {
      if (pollRef.current != null) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [memboxPanelOpen, fetchMemboxCarts, fetchMemboxStatus])

  if (!memboxPanelOpen) return null

  return (
    <div className="fixed right-0 top-0 bottom-0 w-96 bg-[var(--chrome-bg)] border-l border-slate-800 shadow-2xl z-40 flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" />
          <h2 className="text-sm font-semibold text-slate-200">Membox</h2>
        </div>
        <button
          onClick={toggleMemboxPanel}
          className="text-slate-500 hover:text-slate-300 transition-colors"
          aria-label="Close membox panel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Mount form (temporary -- will be unified with single-cart mount in Phase 2) */}
        <MountForm onMounted={fetchMemboxCarts} />

        {/* Section 1: Cart overview */}
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
            mounted carts ({memboxCarts.length})
          </div>
          {memboxCarts.length === 0 ? (
            <div className="text-xs text-slate-600 italic px-3 py-4 rounded border border-dashed border-slate-800">
              No Membox carts mounted yet. Use the form above.
            </div>
          ) : (
            <div className="space-y-1.5">
              {memboxCarts.map((cart) => (
                <CartRow
                  key={cart.cart_id}
                  cart={cart}
                  selected={selectedMemboxCart === cart.cart_id}
                  onClick={() => selectMemboxCart(
                    selectedMemboxCart === cart.cart_id ? null : cart.cart_id
                  )}
                  onUnmount={async () => {
                    try {
                      await api.memboxUnmount(cart.cart_id)
                      if (selectedMemboxCart === cart.cart_id) selectMemboxCart(null)
                      fetchMemboxCarts()
                    } catch (e) { console.error('unmount failed', e) }
                  }}
                />
              ))}
            </div>
          )}
        </div>

        {/* Section 2 + 3: Lock detail and write feed for selected cart */}
        {selectedMemboxCart && (
          <>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">lock state</div>
              <LockDetail />
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">activity</div>
              <WriteFeed />
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">write a pattern</div>
              <ImprintForm cartId={selectedMemboxCart} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
