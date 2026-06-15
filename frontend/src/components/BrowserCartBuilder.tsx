import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle,
  Cpu,
  Download,
  FileText,
  Globe,
  Hammer,
  Loader2,
  Upload,
  X,
  Zap,
} from 'lucide-react'
import {
  buildCartFromFiles,
  downloadBuiltCart,
  type BuiltCart,
  type PipelineProgress,
} from '../cart-builder-v2'

type WebGPUStatus = 'detecting' | 'available' | 'unavailable'

// Block 4 (UI integration) — self-contained browser-side cart builder.
// Renders inside CartBuilderScreen. Uses Block 1-3 pipeline:
// parse → chunk → embed → write → download. No server roundtrip.
// Headline pitch: "your data never leaves your machine."

// File extensions the picker will accept. Order roughly groups by category for
// readability. Anything not in the explicit parser registry falls through to
// textParser.parse() (raw UTF-8 read), which works fine for text-based formats
// (code, YAML, JSON, RST, etc.) — they're just embedded as their source text.
// HTML gets a dedicated parser (parsers/html.ts) that strips tags first.
const ACCEPT_TYPES = [
  // Documents (existing parsers)
  '.pdf', '.docx', '.xlsx', '.csv', '.txt', '.md', '.markdown', '.rtf',
  // Web / markup
  '.html', '.htm', '.xml',
  // Structured data + config
  '.json', '.jsonl', '.ndjson',
  '.yaml', '.yml', '.toml',
  '.ini', '.cfg', '.conf', '.properties', '.log',
  // Documentation variants
  '.mdx', '.qmd', '.rst', '.tex', '.adoc', '.org',
  // Code — general
  '.py', '.pyi', '.ipynb',
  '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs',
  '.go', '.rs', '.java', '.kt', '.scala',
  '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hh',
  '.rb', '.cs', '.php', '.swift',
  '.lua', '.luau', '.zig',
  // Code — shell + scripts
  '.sh', '.bash', '.zsh', '.fish',
  '.ps1', '.psm1', '.bat', '.cmd',
  // Code — functional + niche
  '.ex', '.exs', '.jl', '.elm', '.clj', '.cljs', '.hs',
  // Code — DSL + query
  '.sql', '.graphql', '.gql',
  // Code — mobile / framework
  '.m', '.mm', '.dart',
  '.vue', '.svelte',
  '.groovy', '.gradle',
  // Code — hardware / shader
  '.v', '.sv',
  '.cu', '.cuh', '.wgsl', '.glsl', '.hlsl',
  // Code — Fortran
  '.f', '.f90', '.f95', '.f03', '.f08',
  // Code — Pascal / Delphi
  '.pas', '.pp', '.dpr', '.dpk', '.lpr', '.inc', '.dfm', '.lfm', '.lpk',
  // Code — infra-as-code
  '.tf', '.hcl',
].join(',')

interface QueuedFile {
  file: File
  id: string
}

export default function BrowserCartBuilder() {
  const [cartName, setCartName] = useState('')
  const [queued, setQueued] = useState<QueuedFile[]>([])
  const [progress, setProgress] = useState<PipelineProgress | null>(null)
  const [building, setBuilding] = useState(false)
  const [result, setResult] = useState<BuiltCart | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [webgpuStatus, setWebgpuStatus] = useState<WebGPUStatus>('detecting')
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Probe WebGPU once on mount so users see upfront whether they're on the
  // fast path or WASM fallback. requestAdapter() is the only reliable check —
  // `'gpu' in navigator` is true on browsers that expose the API but lack
  // working drivers (returns null adapter).
  useEffect(() => {
    let cancelled = false
    const probe = async () => {
      const nav = navigator as Navigator & {
        gpu?: { requestAdapter: () => Promise<unknown> }
      }
      if (!nav.gpu) {
        if (!cancelled) setWebgpuStatus('unavailable')
        return
      }
      try {
        const adapter = await nav.gpu.requestAdapter()
        if (!cancelled) {
          setWebgpuStatus(adapter ? 'available' : 'unavailable')
        }
      } catch {
        if (!cancelled) setWebgpuStatus('unavailable')
      }
    }
    void probe()
    return () => {
      cancelled = true
    }
  }, [])

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return
    setError(null)
    setResult(null)
    setProgress(null)
    setQueued((prev) => [
      ...prev,
      ...Array.from(files).map((f) => ({
        file: f,
        id:
          typeof crypto !== 'undefined' && 'randomUUID' in crypto
            ? crypto.randomUUID()
            : `${Date.now()}_${Math.random().toString(36).slice(2)}`,
      })),
    ])
  }

  const removeFile = (id: string) => {
    setQueued((prev) => prev.filter((q) => q.id !== id))
  }

  const clearAll = () => {
    setQueued([])
    setProgress(null)
    setResult(null)
    setError(null)
  }

  const handleBuild = async () => {
    if (queued.length === 0 || building) return
    setBuilding(true)
    setError(null)
    setResult(null)
    try {
      const sanitizedName =
        cartName.trim().replace(/[^A-Za-z0-9_-]/g, '_') || 'my-cart'
      const buildResult = await buildCartFromFiles(
        queued.map((q) => q.file),
        {
          cartName: sanitizedName,
          onProgress: setProgress,
        }
      )
      setResult(buildResult.cart)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBuilding(false)
    }
  }

  const overallPct = computeOverallPct(progress)

  return (
    <div className="rounded-xl border border-purple-500/40 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        <Globe size={22} className="text-purple-300 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <h2 className="text-base font-semibold text-purple-100 flex items-center gap-2 flex-wrap">
            Build a cart in your browser
            <span className="text-[10px] uppercase tracking-wider text-purple-300 bg-purple-500/20 px-2 py-0.5 rounded font-mono">
              new
            </span>
            <BackendBadge status={webgpuStatus} />
            <span
              className="text-[10px] uppercase tracking-wider text-cyan-300 bg-cyan-500/15 border border-cyan-500/40 px-2 py-0.5 rounded font-mono"
              title="Built carts download to your machine. CLOUD mode (carts persist in your own cloud data store — R2, S3, or your hosted Vector+ Studio instance) lands with v1.2."
            >
              local
            </span>
            <span
              className="text-[10px] uppercase tracking-wider text-slate-500 bg-slate-700/30 border border-slate-700 px-2 py-0.5 rounded font-mono cursor-help"
              title="CLOUD mode lands with v1.2 — finished carts persist in your own cloud data store (R2, S3, or your hosted Vector+ Studio instance). We never see your documents in either mode."
            >
              cloud · v1.2
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">
            Your documents never leave your machine — parsing, embedding, and packaging all run client-side via WebGPU
            (with WASM fallback). The finished cart downloads to your machine — mount it locally, on the public demo,
            or on any Vector+ Studio instance you control.
          </p>
        </div>
      </div>

      {/* Cart name */}
      <div>
        <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
          Cart name
        </label>
        <input
          type="text"
          value={cartName}
          onChange={(e) => setCartName(e.target.value)}
          placeholder="Enter cart name…"
          disabled={building}
          className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
        />
        <div className="text-[10px] text-slate-600 mt-1 italic">
          alphanumeric, dashes, underscores only — sanitized at build time
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!building) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          if (!building) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => !building && fileInputRef.current?.click()}
        className={`rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
          building
            ? 'border-slate-800 bg-slate-900/40 cursor-not-allowed opacity-60'
            : dragOver
              ? 'border-purple-400 bg-purple-500/10 cursor-pointer'
              : 'border-slate-700 bg-slate-800/30 hover:border-purple-500/60 cursor-pointer'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ACCEPT_TYPES}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <Upload
          size={28}
          className={`mx-auto mb-2 ${dragOver ? 'text-purple-300' : 'text-slate-500'}`}
        />
        <div className="text-sm text-slate-300 font-medium">
          Drop documents or click to browse
        </div>
        <div className="text-[10px] text-slate-500 mt-1 font-mono">
          PDF · DOCX · XLSX · HTML · Code · Configs · 70+ types
        </div>
      </div>

      {/* Queued files */}
      {queued.length > 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/50 divide-y divide-slate-800">
          <div className="px-3 py-2 text-[10px] uppercase tracking-wider text-slate-500 flex items-center justify-between">
            <span>
              {queued.length} file{queued.length === 1 ? '' : 's'} queued
            </span>
            {!building && (
              <button
                onClick={clearAll}
                className="text-slate-500 hover:text-rose-400 normal-case"
              >
                Clear
              </button>
            )}
          </div>
          {queued.map((q) => (
            <div
              key={q.id}
              className="px-3 py-2 flex items-center gap-2 text-xs"
            >
              <FileText size={12} className="text-slate-500" />
              <span className="text-slate-300 font-medium truncate flex-1">
                {q.file.name}
              </span>
              <span className="text-slate-500 font-mono shrink-0">
                {(q.file.size / 1024).toFixed(1)} KB
              </span>
              {!building && (
                <button
                  onClick={() => removeFile(q.id)}
                  className="text-slate-600 hover:text-rose-400 transition-colors"
                  title="Remove"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Build button */}
      <button
        onClick={handleBuild}
        disabled={building || queued.length === 0}
        className={`w-full rounded-lg px-4 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
          building || queued.length === 0
            ? 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
            : 'bg-purple-500/30 border border-purple-500/50 text-purple-100 hover:bg-purple-500/40'
        }`}
      >
        {building ? (
          <Loader2 size={14} className="animate-spin" />
        ) : (
          <Hammer size={14} />
        )}
        {building ? 'Building in browser…' : 'Build cart in browser'}
      </button>

      {/* Progress */}
      {building && progress && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Loader2 size={12} className="text-amber-400 animate-spin shrink-0" />
            <span className="text-amber-200 font-mono uppercase tracking-wider text-[10px]">
              {progress.stage}
            </span>
            <span className="text-slate-400 font-mono truncate flex-1">
              {progress.message ?? ''}
            </span>
            {overallPct !== null && (
              <span className="text-amber-300 font-mono shrink-0">
                {overallPct}%
              </span>
            )}
          </div>
          {overallPct !== null && (
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-400 to-purple-500 transition-all duration-300"
                style={{ width: `${overallPct}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 flex items-start gap-2 text-xs">
          <AlertCircle size={14} className="text-rose-400 shrink-0 mt-0.5" />
          <div className="text-rose-200 min-w-0 flex-1">
            <div className="font-medium mb-0.5">Build failed</div>
            <div className="text-rose-300/80 font-mono text-[11px] break-words">
              {error}
            </div>
          </div>
        </div>
      )}

      {/* Result */}
      {result && !building && (
        <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 space-y-2">
          <div className="flex items-center gap-2 text-sm flex-wrap">
            <CheckCircle size={14} className="text-emerald-400 shrink-0" />
            <span className="text-emerald-200 font-medium">
              Built {result.cartFilename}
            </span>
            <span className="text-slate-400 font-mono text-[10px] ml-auto">
              {result.manifest.count} chunks · fingerprint {result.manifest.fingerprint}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => { void downloadBuiltCart(result) }}
              className="px-3 py-1.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 text-xs font-medium hover:bg-emerald-500/30 flex items-center gap-1.5 transition-colors"
              title="Pick a destination folder (Chrome/Edge/Opera 86+) or fall back to your Downloads folder (Firefox / Safari)."
            >
              <Download size={12} />
              Save cart bundle…
            </button>
            <span className="text-[10px] text-slate-500">
              .cart.npz + .cart_manifest.json + .permissions.json
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// WebGPU/WASM status badge. Surfaces the embedder backend before the user
// triggers a build, so they know upfront if they're on the slow path.
function BackendBadge({ status }: { status: WebGPUStatus }) {
  if (status === 'detecting') {
    return (
      <span className="text-[10px] uppercase tracking-wider text-slate-500 bg-slate-700/30 border border-slate-600/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1">
        <Loader2 size={9} className="animate-spin" />
        detecting
      </span>
    )
  }
  if (status === 'available') {
    return (
      <span
        className="text-[10px] uppercase tracking-wider text-emerald-300 bg-emerald-500/15 border border-emerald-500/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1"
        title="Your browser exposes WebGPU. The embedder will run on GPU compute — fast path."
      >
        <Zap size={9} />
        WebGPU
      </span>
    )
  }
  return (
    <span
      className="text-[10px] uppercase tracking-wider text-amber-300 bg-amber-500/15 border border-amber-500/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1"
      title="WebGPU not available — embedder will use WebAssembly. Functional but slower (model load + per-chunk embedding take ~2-3× longer)."
    >
      <Cpu size={9} />
      WASM (slower)
    </span>
  )
}

// Map a PipelineProgress event to an overall 0-100% value for the progress bar.
// Stage-weighted: parsing 0-15, chunking 15-18, model load 18-30, embed 30-90,
// write 90-100. Returns null for the 'error' stage so the bar disappears.
function computeOverallPct(progress: PipelineProgress | null): number | null {
  if (!progress) return null
  switch (progress.stage) {
    case 'idle':
      return 0
    case 'parsing': {
      const total = progress.filesTotal || 1
      const parsed = progress.filesParsed ?? 0
      return Math.round((parsed / total) * 15)
    }
    case 'chunking':
      return 18
    case 'embedding': {
      // Two sub-phases: model download (18-30%) then chunk embedding (30-90%).
      if (progress.embeddingsCompleted && progress.embeddingsTotal) {
        const embedPct = (progress.embeddingsCompleted / progress.embeddingsTotal) * 60
        return Math.round(30 + embedPct)
      }
      const modelPct = progress.modelDownloadProgress ?? 0
      return Math.round(18 + (modelPct / 100) * 12)
    }
    case 'writing':
      return 95
    case 'done':
      return 100
    case 'error':
      return null
  }
}
