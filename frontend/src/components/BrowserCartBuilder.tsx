import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
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
  classifyPdf,
  downloadBuiltCart,
  isImageFile,
  probeWebGpuCapability,
  syncRouteForFile,
  webGpuAdapterAvailable,
  type BuiltCart,
  type EmbedderBackend,
  type FileRoute,
  type PipelineProgress,
} from '../cart-builder-v2'
import { useAppStore } from '../store/appStore'
import {
  buildCart as apiBuildCart,
  clearWorkspace as apiClearWorkspace,
  getBuildStatus as apiGetBuildStatus,
  uploadFiles as apiUploadFiles,
} from '../api/cartbuilder'
import ImageBuilderMissingDialog from './ImageBuilderMissingDialog'

// 'detecting'      — initial mount, fast adapter check in flight
// 'probing'        — adapter exists, running tiny real-workload embed to
//                    verify the device can actually sustain inference
//                    (catches the laptop-integrated-Radeon case where
//                    requestAdapter() succeeds but ONNX hangs the device)
// 'webgpu'         — probe succeeded; fast path
// 'wasm'           — no adapter OR probe failed at startup; CPU path
// 'wasm-fallback'  — probe said WebGPU was fine, but device hung mid-build;
//                    failover swap happened, the build continued on WASM
type WebGPUStatus = 'detecting' | 'probing' | 'webgpu' | 'wasm' | 'wasm-fallback'

// Delegated-build progress phases + a stage-weighted overall percentage.
// clearing: 0-5, uploading: 5-30 (proportional to files done), building: 30-100.
// Kept separate from PipelineProgress so the browser path's percentage
// calculations don't need to grow a second case.
interface DesktopBuildProgress {
  phase: 'clearing' | 'uploading' | 'building' | 'done' | 'error'
  message: string
  pct: number
  chunksDone?: number
  chunksTotal?: number
}

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
  // Day 2 — images (route to Image Builder /ocr). Kept out of the binary
  // rejection set below so the drop zone accepts them + the badge system
  // paints them as amber IMAGE.
  '.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.webp', '.bmp',
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

// Extensions we know to be binary / non-textual. We reject these at queue
// time with a user-visible note rather than letting them fall through to the
// text parser, which would read random bytes, chunk them into garbage tokens,
// and crash ONNX with an int32 overflow inside SafeInt. Magic-byte sniffing
// would be more robust but extension-based is enough for the common cases.
const BINARY_EXTENSIONS = new Set([
  // Raster + vector images
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff', '.ico',
  '.heic', '.heif', '.avif',
  // Photoshop / design / vector working files
  '.psd', '.psb', '.ai', '.sketch', '.fig', '.xcf',
  // Audio
  '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus',
  // Video
  '.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v',
  // Archive
  '.zip', '.tar', '.gz', '.tgz', '.bz2', '.7z', '.rar', '.xz',
  // Executable / native
  '.exe', '.dll', '.so', '.dylib', '.bin', '.dmg', '.deb', '.rpm', '.msi',
  // Font
  '.ttf', '.otf', '.woff', '.woff2', '.eot',
  // Database
  '.db', '.sqlite', '.sqlite3', '.mdb',
  // Numeric / scientific binary
  '.npy', '.npz', '.h5', '.hdf5', '.parquet', '.feather', '.arrow',
  '.mat', '.nc', '.fits',
  // Pickled / serialized (Python, Java, R, etc.)
  '.pkl', '.pickle', '.joblib', '.rds', '.rda', '.ser',
  // Compiled / native object code
  '.o', '.a', '.obj', '.lib', '.exp', '.pdb', '.ilk',
  '.class', '.jar', '.pyc', '.pyd', '.pyo',
  // Disk / OS images
  '.iso', '.img', '.vhd', '.vmdk', '.qcow2',
])

function isBinaryFile(file: File): boolean {
  // Day 2 — images are handled by Image Builder rather than the browser
  // parser + embedder pair, so they're no longer "binary rejects." Punch a
  // hole in the extension check for anything isImageFile() recognizes so
  // the rest of BINARY_EXTENSIONS (audio, video, archives, exec) stays
  // enforced as before.
  if (isImageFile(file)) return false
  const name = file.name.toLowerCase()
  // Find the last . to get the extension
  const dot = name.lastIndexOf('.')
  if (dot === -1) return false
  const ext = name.slice(dot)
  return BINARY_EXTENSIONS.has(ext)
}

interface QueuedFile {
  file: File
  id: string
  // Day 2 — route decision drives the badge in the file queue AND the
  // Build-click check for Image Builder availability. Initial value is
  // syncRouteForFile()'s output; for PDFs the async classifyPdf() upgrades
  // 'pending' → 'text' | 'scanned' when the pdfjs call resolves.
  route: FileRoute
}

export default function BrowserCartBuilder() {
  const [cartName, setCartName] = useState('')
  const [queued, setQueued] = useState<QueuedFile[]>([])
  const [skippedNotice, setSkippedNotice] = useState<string | null>(null)
  const [progress, setProgress] = useState<PipelineProgress | null>(null)
  const [building, setBuilding] = useState(false)
  const [result, setResult] = useState<BuiltCart | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [webgpuStatus, setWebgpuStatus] = useState<WebGPUStatus>('detecting')
  // Optional Pattern-0 metadata captured via the "Cart metadata" disclosure.
  // Collapsed by default so the main Cart Name + Drop Zone flow stays uncluttered;
  // empty inputs let the server apply generic defaults for description/agent_briefing.
  const [metadataOpen, setMetadataOpen] = useState(false)
  const [metaDescription, setMetaDescription] = useState('')
  const [metaAgentBriefing, setMetaAgentBriefing] = useState('')
  const [metaTags, setMetaTags] = useState('')
  const [metaOwner, setMetaOwner] = useState('')
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  // Fix #11 (corrected 2026-06-30 PM): while the native file picker dialog is
  // open, the drop zone must reject drag-and-drop. Otherwise the user can
  // drag files FROM the picker's own file list INTO the drop zone, which is
  // confusing dual-input UX. Browsers don't fire a clean "picker closed"
  // event, so we track it via: set true when we call .click(), set false
  // on onChange (files selected) OR on window focus regain (dialog dismissed
  // without selection). The window-focus branch has a small delay so onChange
  // fires first when files WERE selected — otherwise we'd race.
  const [pickerOpen, setPickerOpen] = useState(false)
  useEffect(() => {
    if (!pickerOpen) return
    const onFocus = () => {
      // Small delay lets onChange fire first when the user actually picked
      // files. If onChange runs, it also clears pickerOpen — this handler
      // then just no-ops. If the user cancelled, onChange never fires and
      // this handler is the only thing that clears pickerOpen.
      window.setTimeout(() => setPickerOpen(false), 150)
    }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [pickerOpen])

  // Fix #10 — Force unmount before build. A user should have to unmount any
  // currently mounted cart (backend or browser-side LocalCart) before firing a
  // new build. Prevents the "why did my old cart disappear?" surprise when the
  // built cart is saved and then mounted, and keeps mental state one-cart-at-a-
  // time. Backend mount = status.mounted_cartridge; browser mount = activeLocalCart.
  const backendMounted = useAppStore((s) => s.status?.mounted_cartridge ?? null)
  const activeLocalCart = useAppStore((s) => s.activeLocalCart)
  const hasMountedCart = !!backendMounted || !!activeLocalCart
  // When the Desktop Cart Builder exe is paired, Build delegates to the
  // loopback exe (native ONNX + GPU cooldown built-in). Unpaired browser
  // sessions run the client-side WebGPU/WASM pipeline. Header + button label
  // + Pattern-0 creator all flip on this same flag so the paired state is
  // consistent across the surface. Primitive selector — Zustand's Object.is
  // equality re-renders on 'detected-paired' <-> other-state transitions.
  const desktopHelperState = useAppStore((s) => s.desktopHelperState)
  const openDesktopHelperPairModal = useAppStore((s) => s.openDesktopHelperPairModal)
  const isDesktopPaired = desktopHelperState === 'detected-paired'
  const buildLabelIdle = isDesktopPaired ? 'Build cart on local drive' : 'Build cart in browser'
  const buildCreator = isDesktopPaired ? 'Cart Builder (local)' : 'Cart Builder (browser)'

  // Delegated-path state. Kept separate from the browser pipeline's
  // (progress, result) so the two paths don't leak into each other's UI
  // reducers. desktopProgress mirrors the exe's /build/status shape into a
  // display-ready form; desktopResult holds the on-disk cart_path returned
  // when status flips to 'done'.
  const [desktopProgress, setDesktopProgress] = useState<DesktopBuildProgress | null>(null)
  const [desktopResult, setDesktopResult] = useState<{
    cartPath: string
    cartSizeMb: number | null
    chunksTotal: number | null
  } | null>(null)
  const [repairPromptOpen, setRepairPromptOpen] = useState(false)

  // Day 2 — Image Builder detection state and fallback dialog. We consult
  // imageBuilderState at Build-click time to decide whether to route through
  // Image Builder (paired) or surface the missing-Image-Builder dialog
  // (not-found + queue has image / scanned-PDF files). The dialog stores the
  // affected filenames so the user can see WHAT they'd be skipping before
  // confirming.
  const imageBuilderState = useAppStore((s) => s.imageBuilderState)
  const [imageBuilderMissingOpen, setImageBuilderMissingOpen] = useState(false)
  const [imageBuilderMissingFiles, setImageBuilderMissingFiles] = useState<string[]>([])

  // Two-stage WebGPU capability probe on mount so the backend badge tells the
  // truth instead of advertising green for "adapter exists but inference will
  // hang." Stage 1 (fast, ~ms): does a WebGPU adapter exist at all? Stage 2
  // (slow, ~10-30s): can the adapter actually sustain the Nomic embed model?
  // The second stage matters because constrained devices (integrated GPUs
  // sharing system RAM, older drivers, Chromium Windows quirks) return a
  // valid adapter from requestAdapter() but hang D3D12 mid-inference with
  // DXGI_ERROR_DEVICE_HUNG. probeWebGpuCapability() runs a tiny real embed;
  // if it fails or times out, the cached embedder is silently swapped to WASM
  // BEFORE the user's first Build click — no failed builds, no page reload.
  // Pre-2026-06-29 the probe only did requestAdapter(); see
  // [[CC_webgpu-device-lost-fallback-2026-06-29]] for the laptop-build repro
  // that motivated the deeper probe.
  useEffect(() => {
    let cancelled = false
    const probe = async () => {
      if (!webGpuAdapterAvailable()) {
        if (!cancelled) setWebgpuStatus('wasm')
        // No adapter — pre-warm WASM so the first Build click doesn't pay
        // model-download cost on top of unavoidable embed time.
        await probeWebGpuCapability().catch(() => {})
        return
      }
      if (!cancelled) setWebgpuStatus('probing')
      // onDecision fires the moment the heuristic makes its call (~ms),
      // BEFORE the slow model pre-load starts. Without this the badge sits
      // on "probing" for 10-30s during model download, confusing users into
      // thinking the probe is stuck (2026-06-30 PM Andy laptop test repro).
      const backend = await probeWebGpuCapability({
        onDecision: (decided) => {
          if (!cancelled) setWebgpuStatus(decided === 'webgpu' ? 'webgpu' : 'wasm')
        },
      })
      if (cancelled) return
      setWebgpuStatus(backend === 'webgpu' ? 'webgpu' : 'wasm')
    }
    void probe()
    return () => {
      cancelled = true
    }
  }, [])

  // Called from buildCartFromFiles via the onBackendChange hook when the
  // embedder backend swaps mid-build (e.g. WebGPU device hung → automatic
  // WASM failover). Updates the badge so the user sees the graceful
  // degradation in real time instead of wondering why the build kept going
  // after the console exploded.
  const handleBackendChange = (backend: EmbedderBackend) => {
    if (backend === 'wasm') setWebgpuStatus('wasm-fallback')
    else if (backend === 'webgpu') setWebgpuStatus('webgpu')
  }

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return
    setError(null)
    setResult(null)
    setProgress(null)
    // Partition incoming files into text-based (accepted) vs binary (rejected).
    // Binary files would crash the embedder with garbage-byte tokens, so we
    // reject them up front with a clear notice rather than failing mid-build.
    const all = Array.from(files)
    const accepted: File[] = []
    const rejected: string[] = []
    for (const f of all) {
      if (isBinaryFile(f)) {
        rejected.push(f.name)
      } else {
        accepted.push(f)
      }
    }
    if (rejected.length > 0) {
      setSkippedNotice(
        `Skipped ${rejected.length} binary file${rejected.length === 1 ? '' : 's'} (images/audio/video/archives not yet supported): ` +
        rejected.slice(0, 5).join(', ') +
        (rejected.length > 5 ? `, +${rejected.length - 5} more` : '')
      )
    } else {
      setSkippedNotice(null)
    }
    if (accepted.length === 0) return
    // Build QueuedFile records with an initial route (sync MIME check).
    // PDFs land as 'pending' — we fire the async classifyPdf below and
    // upgrade the badge when it resolves.
    const newQueueEntries = accepted.map((f) => ({
      file: f,
      id:
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `${Date.now()}_${Math.random().toString(36).slice(2)}`,
      route: syncRouteForFile(f),
    }))
    setQueued((prev) => [...prev, ...newQueueEntries])
    // Kick off PDF classification for any newly-added PDFs. We update the
    // queue entry in place once each classification resolves — matching by
    // id since the queue could have been reordered / trimmed by then.
    // Failures fall back to 'text' (the safer default: at worst, the PDF
    // parser returns a few chunks of extracted text).
    for (const entry of newQueueEntries) {
      if (entry.route !== 'pending') continue
      void (async () => {
        let resolvedRoute: FileRoute
        try {
          const kind = await classifyPdf(entry.file)
          resolvedRoute = kind === 'text' ? 'text' : 'scanned'
        } catch {
          resolvedRoute = 'text'
        }
        setQueued((prev) =>
          prev.map((q) => (q.id === entry.id ? { ...q, route: resolvedRoute } : q)),
        )
      })()
    }
  }

  const removeFile = (id: string) => {
    setQueued((prev) => prev.filter((q) => q.id !== id))
  }

  const clearAll = () => {
    setQueued([])
    setProgress(null)
    setResult(null)
    setError(null)
    setDesktopProgress(null)
    setDesktopResult(null)
    setRepairPromptOpen(false)
  }

  // Fix #7 — Post-save clean-slate refresh. After the user successfully saves
  // the built cart to disk, wipe all component state so the tab returns to a
  // pristine "ready to build a new cart" surface. We do NOT reset the WebGPU
  // status — it's device-level, not build-specific, and re-probing on every
  // save would burn 10-30s on the next build.
  const resetToCleanSlate = () => {
    setCartName('')
    setQueued([])
    setSkippedNotice(null)
    setProgress(null)
    setResult(null)
    setError(null)
    setMetadataOpen(false)
    setMetaDescription('')
    setMetaAgentBriefing('')
    setMetaTags('')
    setMetaOwner('')
    setDesktopProgress(null)
    setDesktopResult(null)
    setRepairPromptOpen(false)
  }

  // Wraps downloadBuiltCart so the clean-slate reset only fires on a genuine
  // successful save. If showSaveFilePicker throws (user cancelled, permission
  // denied), we leave the built cart on-screen so they can try again.
  const handleSaveBundle = async () => {
    if (!result) return
    try {
      await downloadBuiltCart(result)
      resetToCleanSlate()
    } catch (e) {
      // User cancelled the save picker or the browser blocked it. Leave the
      // result visible; surface any real error to the user.
      const msg = e instanceof Error ? e.message : String(e)
      // AbortError is the standard cancel signal from showSaveFilePicker; not
      // an error worth showing.
      if (!/abort/i.test(msg)) setError(`Save failed: ${msg}`)
    }
  }

  // Day 2 — Image Builder gating. Returns the list of queued filenames that
  // need Image Builder (image OR scanned-PDF route). Called at Build-click
  // time to decide whether to surface the fallback dialog. Pending PDF
  // classifications count as "possibly needs it" — safer to prompt the user
  // than to guess wrong and stall mid-build.
  const filesNeedingImageBuilder = (): string[] => {
    return queued
      .filter((q) => q.route === 'image' || q.route === 'scanned' || q.route === 'pending')
      .map((q) => q.file.name)
  }

  // Wraps the actual build kickoff. Called directly on Build click when
  // Image Builder is available or no queued file needs it; called from the
  // Skip button in ImageBuilderMissingDialog with the queue pre-filtered.
  const runBuild = async (filesToBuild: QueuedFile[]) => {
    if (filesToBuild.length === 0 || building || hasMountedCart) return
    setBuilding(true)
    setError(null)
    setResult(null)
    setDesktopResult(null)
    setDesktopProgress(null)
    setRepairPromptOpen(false)
    const sanitizedName =
      cartName.trim().replace(/[^A-Za-z0-9_-]/g, '_') || 'my-cart'
    // Parse tags from the comma-separated input; drop empties/whitespace-only
    // entries so the pattern0_data tags list is a clean array of strings.
    const parsedTags = metaTags
      .split(',')
      .map((t) => t.trim())
      .filter((t) => t.length > 0)
    const meta = {
      description: metaDescription.trim(),
      agent_briefing: metaAgentBriefing.trim(),
      owner: metaOwner.trim(),
      tags: parsedTags,
      creator: buildCreator,
    }
    try {
      if (isDesktopPaired) {
        // Delegated path uploads via the queue's file list — we pass the
        // pre-filtered set explicitly so a Skip decision from the fallback
        // dialog also propagates to the delegated backend upload loop.
        await runDesktopBuild(sanitizedName, meta, filesToBuild)
      } else {
        const buildResult = await buildCartFromFiles(
          filesToBuild.map((q) => q.file),
          {
            cartName: sanitizedName,
            onProgress: setProgress,
            onBackendChange: handleBackendChange,
            buildOptions: {
              pattern0Meta: meta,
            },
          }
        )
        setResult(buildResult.cart)
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      // The api/cartbuilder client's shared 401 handler already flipped the
      // store to 'detected-unpaired' before the fetch threw. When we detect
      // that shape of failure, surface the re-pair CTA in-line instead of
      // leaving the user staring at a raw "Cart Builder API error 401" that
      // implies a build failure rather than an auth expiry.
      if (isDesktopPaired && /\b401\b/.test(msg)) {
        setRepairPromptOpen(true)
      }
    } finally {
      setBuilding(false)
    }
  }

  // Day 2 — Build click entry point. Checks the Image Builder availability
  // gate: if any queued file needs Image Builder and Image Builder isn't
  // paired-and-running, surface the fallback dialog. The dialog's Skip
  // button strips the affected files and calls runBuild directly.
  // Delegated path (isDesktopPaired) is exempt from the gate — the exe
  // reads the shared token from ~/.vector-plus/token and does its own
  // Image Builder call over loopback (see api/cartbuilder/builder.py Day 2
  // delegation). The browser can't observe whether the exe's own probe
  // succeeded, so we let the exe decide + report any failure back via the
  // standard build-status error path.
  const handleBuild = () => {
    if (queued.length === 0 || building || hasMountedCart) return
    if (isDesktopPaired) {
      void runBuild(queued)
      return
    }
    const needing = filesNeedingImageBuilder()
    const gateNeeded = needing.length > 0 && imageBuilderState !== 'detected-paired'
    if (gateNeeded) {
      setImageBuilderMissingFiles(needing)
      setImageBuilderMissingOpen(true)
      return
    }
    void runBuild(queued)
  }

  // Fallback dialog handlers. Skip strips the amber files, closes the
  // dialog, and starts the build with what's left. Cancel closes the
  // dialog and leaves the queue untouched — the user starts Image Builder
  // and retries.
  const handleSkipImageBuilderFiles = () => {
    const survivors = queued.filter(
      (q) => q.route !== 'image' && q.route !== 'scanned' && q.route !== 'pending',
    )
    setImageBuilderMissingOpen(false)
    setImageBuilderMissingFiles([])
    if (survivors.length === 0) {
      // Nothing left to build after skipping — don't kick off an empty build,
      // just close the dialog and let the user retry.
      setError('All queued files needed Image Builder; nothing left to build.')
      return
    }
    void runBuild(survivors)
  }
  const handleCancelImageBuilderMissing = () => {
    setImageBuilderMissingOpen(false)
    setImageBuilderMissingFiles([])
  }

  // Delegated build path. The exe mounts the same /api/cartbuilder/*
  // FastAPI router the VPS ships (see api/cartbuilder/__init__.py); the API
  // client (api/cartbuilder.ts) resolveTarget() routes to loopback + Bearer
  // when 'detected-paired', so the same request functions hit the exe. The
  // exe handles the parse/embed/write pipeline natively (ONNX + built-in
  // GPU cooldown) and reports progress via /build/status polling.
  const runDesktopBuild = async (
    sanitizedName: string,
    meta: {
      description: string
      agent_briefing: string
      owner: string
      tags: string[]
      creator: string
    },
    filesToBuild: QueuedFile[],
  ) => {
    // Phase 1: reset server-side workspace so leftover files from a prior
    // session in the same exe process don't leak into this build.
    setDesktopProgress({ phase: 'clearing', message: 'Preparing workspace…', pct: 2 })
    await apiClearWorkspace()

    // Phase 2: multipart-upload the queued files. One file per request so
    // we can surface a per-file progress line — the exe parses + chunks on
    // upload so uploading N files sequentially also lets the user see any
    // per-file parse error before the whole build kicks off. Upload phase
    // spans 5-30% of the overall bar.
    const total = filesToBuild.length
    for (let i = 0; i < total; i++) {
      const q = filesToBuild[i]
      const uploadPct = 5 + Math.round(((i + 1) / total) * 25)
      setDesktopProgress({
        phase: 'uploading',
        message: `Uploading ${q.file.name} (${i + 1}/${total})`,
        pct: uploadPct,
      })
      await apiUploadFiles([q.file])
    }

    // Phase 3: kick off the async build. The exe returns immediately after
    // spawning the background thread; the actual embed/write runs off-thread
    // and is reflected by /build/status transitions.
    setDesktopProgress({
      phase: 'building',
      message: 'Building cart on local drive…',
      pct: 30,
    })
    await apiBuildCart(sanitizedName, {
      description: meta.description,
      agent_briefing: meta.agent_briefing,
      owner: meta.owner,
      tags: meta.tags,
      creator: meta.creator,
    })

    // Phase 4: poll /build/status at 500ms until done or error. The exe's
    // build_state.progress reports 0..1 across the whole embed+write span;
    // we map that onto the 30-100% slice of our display bar. Loop exit is
    // via return (done) or throw (error) — never falls through.
    for (;;) {
      await sleep(500)
      const s = await apiGetBuildStatus()
      if (s.status === 'done') {
        setDesktopProgress({
          phase: 'done',
          message: 'Build complete',
          pct: 100,
          chunksDone: s.chunks_done,
          chunksTotal: s.chunks_total,
        })
        setDesktopResult({
          cartPath: s.cart_path ?? '(unknown path)',
          cartSizeMb: typeof (s as unknown as { cart_size_mb?: number }).cart_size_mb === 'number'
            ? (s as unknown as { cart_size_mb?: number }).cart_size_mb ?? null
            : null,
          chunksTotal: s.chunks_total ?? null,
        })
        return
      }
      if (s.status === 'error') {
        const errMsg = s.error || 'Build failed on the desktop helper'
        setDesktopProgress({ phase: 'error', message: errMsg, pct: 0 })
        throw new Error(errMsg)
      }
      // Still 'building' or 'idle' — feed progress + chunk counters back
      // into the display. Some transient 'idle' ticks are expected right
      // after the /build POST returns, before the worker thread flips
      // status to 'building'.
      const buildPct = 30 + Math.round(Math.min(Math.max(s.progress ?? 0, 0), 1) * 70)
      const done = s.chunks_done
      const totalChunks = s.chunks_total
      const message = totalChunks
        ? `Embedding ${done ?? 0}/${totalChunks} chunks…`
        : 'Building cart on local drive…'
      setDesktopProgress({
        phase: 'building',
        message,
        pct: buildPct,
        chunksDone: done,
        chunksTotal: totalChunks,
      })
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
            {isDesktopPaired ? 'Build a cart on your local drive' : 'Build a cart in your browser'}
            <span className="text-[10px] uppercase tracking-wider text-purple-300 bg-purple-500/20 px-2 py-0.5 rounded font-mono">
              new
            </span>
            {!isDesktopPaired && <BackendBadge status={webgpuStatus} />}
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
            {/* Forcing function: visible alpha-state banner for the v1 provenance
                sidecar (source_paths.npy). Annoying enough we want to retire it;
                visible enough externally that pilot customers will ask "when does
                v2 land?", creating sales pressure to actually land v2 schema.
                See CC_cart-provenance-schema_2026-06-15 for v2 spec + pilot
                blocker reasoning. Retire this badge once v2 ships. */}
            <span
              className="text-[10px] uppercase tracking-wider text-amber-300 bg-amber-500/15 border border-amber-500/40 px-2 py-0.5 rounded font-mono cursor-help"
              title="Provenance v1 sidecar — source_paths.npy carries filename per pattern alongside h-row. ALPHA. v2 (h-row source_idx + strings table) required before pilot launches with legal/clinical/CPA customers. See CC_cart-provenance-schema_2026-06-15."
            >
              provenance: v1 (alpha)
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">
            {isDesktopPaired
              ? 'Your documents never leave your machine — the paired Desktop Cart Builder handles parsing, embedding, and packaging natively (ONNX GPU with built-in cooldown). The finished cart is written straight to your local drive.'
              : 'Your documents never leave your machine — parsing, embedding, and packaging all run client-side via WebGPU (with WASM fallback). The finished cart downloads to your machine — mount it locally, on the public demo, or on any Vector+ Studio instance you control.'}
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

      {/* Cart metadata (optional) — collapsed by default. Values flow into the
          built cart's Pattern-0 so the TOC panel can surface them at mount time.
          Empty description / agent briefing get generic server-side defaults;
          tags + owner stay empty when the user leaves them blank. */}
      <div className="rounded-lg border border-slate-800 bg-slate-900/40">
        <button
          type="button"
          onClick={() => setMetadataOpen((v) => !v)}
          disabled={building}
          aria-expanded={metadataOpen}
          className="w-full px-3 py-2 flex items-center gap-2 text-xs text-slate-300 hover:text-slate-100 disabled:opacity-50"
        >
          {metadataOpen ? (
            <ChevronDown size={12} className="text-slate-500" />
          ) : (
            <ChevronRight size={12} className="text-slate-500" />
          )}
          <span className="uppercase tracking-wider text-[10px] text-slate-500">
            Cart metadata
          </span>
          <span className="text-[10px] text-slate-600 italic normal-case">
            (optional)
          </span>
        </button>
        {metadataOpen && (
          <div className="px-3 pb-3 pt-1 space-y-3 border-t border-slate-800">
            <div>
              <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
                Description
              </label>
              <textarea
                value={metaDescription}
                onChange={(e) => setMetaDescription(e.target.value)}
                placeholder="Describe what this cart contains. Leave blank for a generic description."
                disabled={building}
                rows={3}
                className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500/60 disabled:opacity-50 resize-y"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
                Agent briefing
              </label>
              <textarea
                value={metaAgentBriefing}
                onChange={(e) => setMetaAgentBriefing(e.target.value)}
                placeholder="How should agents use this cart? Leave blank for a generic briefing."
                disabled={building}
                rows={5}
                className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500/60 disabled:opacity-50 resize-y"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
                Tags
              </label>
              <input
                type="text"
                value={metaTags}
                onChange={(e) => setMetaTags(e.target.value)}
                placeholder="comma, separated, tags"
                disabled={building}
                className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 block">
                Owner
              </label>
              <input
                type="text"
                value={metaOwner}
                onChange={(e) => setMetaOwner(e.target.value)}
                placeholder="Leave blank to use your account name."
                disabled={building}
                className="w-full rounded-lg bg-slate-950/60 border border-slate-800 px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500/60 disabled:opacity-50"
              />
            </div>
          </div>
        )}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          // Fix #11 (corrected 2026-06-30 PM): reject drag when picker is open.
          if (!building && !pickerOpen) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          // Fix #11 (corrected 2026-06-30 PM): reject drop when picker is open —
          // otherwise a user could drag files FROM the picker INTO the drop zone.
          if (!building && !pickerOpen) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => {
          // Fix #11 — During an active drag-over, suppress the picker click so
          // dropping doesn't ALSO open the OS file browser (double-trigger UX).
          // Browsers fire click after drop in some sequences; gating on dragOver
          // keeps the drop path clean. Also guard against re-opening the picker
          // if it's already open (double-click during animation, etc.).
          if (building || dragOver || pickerOpen) return
          setPickerOpen(true)
          fileInputRef.current?.click()
        }}
        className={`rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
          building
            ? 'border-slate-800 bg-slate-900/40 cursor-not-allowed opacity-60'
            : pickerOpen
              ? 'border-slate-700 bg-slate-900/40 cursor-not-allowed opacity-60'
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
          onChange={(e) => {
            // Fix #11 (corrected 2026-06-30 PM): mark picker closed when the
            // user finishes selecting. This fires before the window-focus
            // handler's setTimeout, so pickerOpen goes false cleanly whether
            // the user selected files or cancelled.
            setPickerOpen(false)
            handleFiles(e.target.files)
          }}
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

      {/* Skipped-binary notice */}
      {skippedNotice && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200 flex items-start gap-2">
          <AlertCircle className="shrink-0 mt-0.5" size={14} />
          <div className="flex-1">{skippedNotice}</div>
          <button
            type="button"
            onClick={() => setSkippedNotice(null)}
            className="shrink-0 text-amber-300/60 hover:text-amber-200"
            aria-label="Dismiss"
          >
            <X size={12} />
          </button>
        </div>
      )}

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
          {queued.map((q) => {
            // Red left border when the file NEEDS Image Builder but the
            // exe isn't detected — makes the risk visible before Build even
            // fires. Amber rows (paired-but-still-image) aren't decorated
            // beyond their badge to keep the queue readable.
            const needsBuilder = q.route === 'image' || q.route === 'scanned' || q.route === 'pending'
            const missingBuilder = needsBuilder && imageBuilderState !== 'detected-paired'
            return (
              <div
                key={q.id}
                className={`px-3 py-2 flex items-center gap-2 text-xs ${
                  missingBuilder ? 'border-l-2 border-l-rose-500/60 pl-[10px]' : ''
                }`}
              >
                <FileText size={12} className="text-slate-500" />
                <span className="text-slate-300 font-medium truncate flex-1">
                  {q.file.name}
                </span>
                {/* Route badge — TEXT PDF / SCANNED PDF / IMAGE / DOC / MD /
                    PDF (…). Slate for local/fast-path types; amber for the
                    Image Builder route (spec Q4). Uses filename extension
                    where MIME isn't specific enough for the label. */}
                <FileRouteBadge file={q.file} route={q.route} />
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
            )
          })}
        </div>
      )}

      {/* Build button */}
      <button
        onClick={handleBuild}
        disabled={building || queued.length === 0 || !cartName.trim() || hasMountedCart}
        className={`w-full rounded-lg px-4 py-2.5 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
          building || queued.length === 0 || !cartName.trim() || hasMountedCart
            ? 'bg-slate-800/50 border border-slate-700 text-slate-600 cursor-not-allowed'
            : 'bg-purple-500/30 border border-purple-500/50 text-purple-100 hover:bg-purple-500/40'
        }`}
      >
        {building ? (
          <Loader2 size={14} className="animate-spin" />
        ) : (
          <Hammer size={14} />
        )}
        {building
          ? (isDesktopPaired ? 'Building on local drive…' : 'Building in browser…')
          : buildLabelIdle}
      </button>
      {!building && hasMountedCart && (
        <div className="text-xs text-amber-400 font-medium px-1 -mt-1">
          Unmount the current cart ({backendMounted ?? activeLocalCart}) before building a new one.
        </div>
      )}
      {!building && !hasMountedCart && !cartName.trim() && (
        <div className="text-[10px] text-amber-400/80 italic px-1 -mt-1">
          Enter a cart name above before building.
        </div>
      )}

      {/* Progress — browser pipeline */}
      {building && progress && !isDesktopPaired && (
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

      {/* Progress — delegated desktop path. Kept in sync with the exe's
          /build/status polling loop; phase drives the badge, pct drives the
          bar, chunks_done/total shows when the exe reports them. */}
      {building && desktopProgress && isDesktopPaired && (
        <div className="rounded-lg border border-purple-500/30 bg-purple-500/5 p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <Loader2 size={12} className="text-purple-300 animate-spin shrink-0" />
            <span className="text-purple-200 font-mono uppercase tracking-wider text-[10px]">
              {desktopProgress.phase}
            </span>
            <span className="text-slate-400 font-mono truncate flex-1">
              {desktopProgress.message}
            </span>
            <span className="text-purple-300 font-mono shrink-0">
              {desktopProgress.pct}%
            </span>
          </div>
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-400 to-indigo-500 transition-all duration-300"
              style={{ width: `${desktopProgress.pct}%` }}
            />
          </div>
          {desktopProgress.chunksTotal ? (
            <div className="text-[10px] text-slate-500 font-mono">
              {desktopProgress.chunksDone ?? 0}/{desktopProgress.chunksTotal} chunks
            </div>
          ) : null}
        </div>
      )}

      {/* Error */}
      {/* Fix #9 (2026-06-30 PM): error banner has a dismiss X so build errors
          don't stick around until the next build attempt. Same pattern as the
          per-file X button in the queued-files list above — small, low-key,
          shrink-0 so long error text doesn't push it off-frame. Clears
          progress + error together so the panel returns to its neutral state. */}
      {error && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 flex items-start gap-2 text-xs">
          <AlertCircle size={14} className="text-rose-400 shrink-0 mt-0.5" />
          <div className="text-rose-200 min-w-0 flex-1">
            <div className="font-medium mb-0.5">Build failed</div>
            <div className="text-rose-300/80 font-mono text-[11px] break-words">
              {error}
            </div>
          </div>
          <button
            onClick={() => {
              setError(null)
              setProgress(null)
            }}
            className="text-rose-400/60 hover:text-rose-200 transition-colors shrink-0 mt-0.5"
            title="Dismiss"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Re-pair prompt — surfaced when the exe returns 401 mid-build (usually
          because the exe restarted between pair and this call, or the persisted
          token was stale). The api/cartbuilder client's shared handler already
          flipped the store to 'detected-unpaired'; this banner just makes the
          recovery obvious instead of leaving the user staring at a raw 401. */}
      {repairPromptOpen && !building && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 flex items-start gap-2 text-xs">
          <AlertCircle size={14} className="text-amber-400 shrink-0 mt-0.5" />
          <div className="text-amber-200 min-w-0 flex-1">
            <div className="font-medium mb-0.5">Desktop helper needs re-pairing</div>
            <div className="text-amber-300/80 mb-2">
              The exe rejected the saved pairing token — it may have restarted since the last pair. Re-pair to continue building on your local drive, or the build will fall back to the browser pipeline.
            </div>
            <button
              onClick={() => {
                setRepairPromptOpen(false)
                openDesktopHelperPairModal()
              }}
              className="px-2.5 py-1 rounded bg-amber-500/20 border border-amber-500/40 text-amber-200 text-xs font-medium hover:bg-amber-500/30"
            >
              Re-pair Desktop Helper
            </button>
          </div>
          <button
            onClick={() => setRepairPromptOpen(false)}
            className="text-amber-400/60 hover:text-amber-200 transition-colors shrink-0 mt-0.5"
            title="Dismiss"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Result — browser pipeline. Local blob → user picks save destination. */}
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
              onClick={() => { void handleSaveBundle() }}
              className="px-3 py-1.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 text-xs font-medium hover:bg-emerald-500/30 flex items-center gap-1.5 transition-colors"
              title="Pick a destination folder (Chrome/Edge/Opera 86+) or fall back to your Downloads folder (Firefox / Safari). After saving, the tab resets to a clean slate."
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

      {/* Result — delegated desktop path. The exe writes the cart to disk
          directly (default: ~/Downloads, or whatever save_dir the user last
          picked), so we don't trigger a browser download — that would produce
          a duplicate. Just surface the on-disk path so Andy can find it. */}
      {desktopResult && !building && (
        <div className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 space-y-2">
          <div className="flex items-center gap-2 text-sm flex-wrap">
            <CheckCircle size={14} className="text-emerald-400 shrink-0" />
            <span className="text-emerald-200 font-medium">Cart saved to local drive</span>
            {desktopResult.chunksTotal !== null && (
              <span className="text-slate-400 font-mono text-[10px] ml-auto">
                {desktopResult.chunksTotal} chunks
                {desktopResult.cartSizeMb !== null ? ` · ${desktopResult.cartSizeMb.toFixed(2)} MB` : ''}
              </span>
            )}
          </div>
          <div className="text-[11px] text-slate-300 font-mono break-all bg-slate-950/40 border border-slate-800 rounded px-2 py-1">
            {desktopResult.cartPath}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={resetToCleanSlate}
              className="px-3 py-1.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 text-xs font-medium hover:bg-emerald-500/30"
            >
              Build another cart
            </button>
            <span className="text-[10px] text-slate-500">
              Mount from the Search screen to try it out.
            </span>
          </div>
        </div>
      )}

      {/* Day 2 — Image Builder missing fallback dialog. Surfaces at Build
          click time when the queue contains files that need Image Builder
          AND Image Builder isn't detected on 127.0.0.1:7879. Three
          choices: Skip / Cancel / Help — see ImageBuilderMissingDialog. */}
      <ImageBuilderMissingDialog
        open={imageBuilderMissingOpen}
        affectedFiles={imageBuilderMissingFiles}
        onSkip={handleSkipImageBuilderFiles}
        onCancel={handleCancelImageBuilderMissing}
      />
    </div>
  )
}

// Timer-based sleep for the /build/status polling loop. Kept as a
// module-local helper rather than pulling in a scheduler dependency —
// this is the only place we need it.
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// WebGPU/WASM status badge. Surfaces the embedder backend before the user
// triggers a build, so they know upfront if they're on the slow path. Five
// states: detecting (fast adapter check), probing (real-workload capability
// probe in flight), webgpu (probe passed; fast path), wasm (probe failed or
// no adapter; CPU path), wasm-fallback (started WebGPU but device hung
// mid-build; failover swap happened). Probing is the long state — model
// download + tiny embed takes 10-30s the first time.
function BackendBadge({ status }: { status: WebGPUStatus }) {
  if (status === 'detecting') {
    return (
      <span className="text-[10px] uppercase tracking-wider text-slate-500 bg-slate-700/30 border border-slate-600/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1">
        <Loader2 size={9} className="animate-spin" />
        detecting
      </span>
    )
  }
  if (status === 'probing') {
    return (
      <span
        className="text-[10px] uppercase tracking-wider text-sky-300 bg-sky-500/15 border border-sky-500/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1"
        title="WebGPU adapter found — probing whether it can sustain the embed model. One-time check (~10-30s) so the build doesn't surprise you with a device-lost error halfway through."
      >
        <Loader2 size={9} className="animate-spin" />
        probing webgpu
      </span>
    )
  }
  if (status === 'webgpu') {
    return (
      <span
        className="text-[10px] uppercase tracking-wider text-emerald-300 bg-emerald-500/15 border border-emerald-500/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1"
        title="WebGPU verified end-to-end on this device. The embedder will run on GPU compute — fast path."
      >
        <Zap size={9} />
        WebGPU
      </span>
    )
  }
  if (status === 'wasm-fallback') {
    return (
      <span
        className="text-[10px] uppercase tracking-wider text-amber-300 bg-amber-500/15 border border-amber-500/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1"
        title="WebGPU was working but the device hung mid-build (DXGI_ERROR_DEVICE_HUNG or equivalent). The embedder swapped to WebAssembly automatically — your build will complete, just slower. No page reload needed."
      >
        <Cpu size={9} />
        WASM (fallback)
      </span>
    )
  }
  return (
    <span
      className="text-[10px] uppercase tracking-wider text-amber-300 bg-amber-500/15 border border-amber-500/40 px-2 py-0.5 rounded font-mono inline-flex items-center gap-1"
      title="WebGPU not available on this device (no adapter, or probe detected it can't sustain the embed model). Embedder will use WebAssembly. Functional but slower (model load + per-chunk embedding take ~2-3× longer)."
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

// Per-file badge (Day 2). Shows a compact TEXT PDF / SCANNED PDF / IMAGE /
// DOC / MD / TXT / CODE label so users can see at a glance which route each
// file will take. Slate for existing-parser fast paths, amber for anything
// going to Image Builder (spec Q4).
function FileRouteBadge({ file, route }: { file: File; route: FileRoute }) {
  const name = file.name.toLowerCase()
  const ext = name.slice(name.lastIndexOf('.'))

  // Amber — routes that will hit Image Builder.
  if (route === 'image') {
    return <RouteBadge tone="amber" label="IMAGE" title="Routes to Image Builder /ocr" />
  }
  if (route === 'scanned') {
    return <RouteBadge tone="amber" label="SCANNED PDF" title="PDF text-check found <500 chars; routes to Image Builder" />
  }
  if (route === 'pending') {
    // PDF classification in flight — show a neutral PDF label with a
    // loader so users see the state is still resolving. Once classifyPdf
    // resolves, the badge flips to TEXT PDF or SCANNED PDF.
    return (
      <span
        className="text-[9px] uppercase tracking-wider text-slate-500 bg-slate-700/30 border border-slate-600/40 px-1.5 py-0.5 rounded font-mono inline-flex items-center gap-1 shrink-0"
        title="Classifying PDF as text vs scan (via pdfjs)…"
      >
        <Loader2 size={8} className="animate-spin" />
        PDF
      </span>
    )
  }

  // Slate — text fast path. Derive a specific label from extension where
  // available; fall back to a generic TEXT for everything else.
  if (ext === '.pdf') {
    return <RouteBadge tone="slate" label="TEXT PDF" title="Text-bearing PDF; existing chunker fast path" />
  }
  if (ext === '.docx' || ext === '.doc') return <RouteBadge tone="slate" label="DOCX" />
  if (ext === '.xlsx' || ext === '.xls') return <RouteBadge tone="slate" label="XLSX" />
  if (ext === '.md' || ext === '.markdown' || ext === '.mdx' || ext === '.qmd') {
    return <RouteBadge tone="slate" label="MD" />
  }
  if (ext === '.txt') return <RouteBadge tone="slate" label="TXT" />
  if (ext === '.html' || ext === '.htm') return <RouteBadge tone="slate" label="HTML" />
  if (ext === '.rtf') return <RouteBadge tone="slate" label="RTF" />
  if (ext === '.json' || ext === '.jsonl' || ext === '.ndjson') return <RouteBadge tone="slate" label="JSON" />
  if (ext === '.yaml' || ext === '.yml') return <RouteBadge tone="slate" label="YAML" />
  if (ext === '.pptx' || ext === '.ppt') return <RouteBadge tone="slate" label="PPTX" />
  // Generic text — code files (.py, .ts, .go, etc.) and misc.
  return <RouteBadge tone="slate" label="TEXT" />
}

function RouteBadge({
  tone,
  label,
  title,
}: {
  tone: 'slate' | 'amber'
  label: string
  title?: string
}) {
  const classes =
    tone === 'amber'
      ? 'text-amber-300 bg-amber-500/15 border-amber-500/40'
      : 'text-slate-400 bg-slate-700/30 border-slate-600/40'
  return (
    <span
      className={`text-[9px] uppercase tracking-wider ${classes} border px-1.5 py-0.5 rounded font-mono shrink-0`}
      title={title}
    >
      {label}
    </span>
  )
}
