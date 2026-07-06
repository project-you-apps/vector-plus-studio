import { create } from 'zustand'
import type {
  CartridgeInfo, SearchResult, StatusResponse, DeletedPattern, SearchMode,
  MemboxCartInfo, MemboxStatus,
} from '../api/types'
import * as api from '../api/client'
import { probeImageBuilder } from '../api/imageBuilder'
import { detectWebGPU, runWebGpuAssociate, loadBrainForCart, isBrainLoadedFor } from '../lib/webgpuAssociate'
import { parseCartFile, cosineSearchLocal } from '../lib/localCart'

// Pattern-0 metadata baked into a browser-built cart. Mirrors the shape of
// api.getCartPattern0's Pattern0Response so Pattern0TocPanel can consume it
// uniformly regardless of mount origin. Fields are all optional/nullable
// since browser-side JSON parse can't guarantee schema conformance.
export interface LocalCartPattern0Meta {
  cart_name?: string | null
  creator?: string | null
  created_at?: string | null
  owner?: string | null
  description?: string | null
  agent_briefing?: string | null
  tags?: string[]
  file_count?: number
  total_chunks?: number
  files?: Array<{
    name: string
    owner?: string | null
    description?: string | null
    tags?: string[]
    chunks?: number
  }>
  // Day 2 — Image Builder integration counts. Zero (or absent) for text-only
  // carts. Pattern0TocPanel shows a "N graphics + M tables" line under the
  // header when either exceeds zero.
  graphic_count?: number
  table_count?: number
}

// Per-pattern metadata sidecar (parsed from per_pattern_meta.npy). One record
// per pattern in `passages`; parallel-indexed. Content-type-specific extras
// (image_b64 for graphics, html for tables, bbox for both) are optional so
// legacy carts with only text patterns don't have to populate them.
// Andy 2026-07-05 PM: added to surface graphic PNG thumbnails throughout the
// UI (Result cards, PassageModal, TOC drill-down).
export interface LocalCartPatternMeta {
  v?: number
  content_type?: 'document' | 'graphic' | 'table' | string
  source?: string
  page?: number | null
  chunk?: number
  chunks?: number
  tags?: string[]
  created_at?: number
  tombstone?: boolean
  // Graphic extras
  caption?: string
  image_b64?: string
  bbox?: number[]
  // Table extras
  html?: string
}

// A cart that lives on the user's local disk and was parsed client-side.
// Nothing ever leaves the browser; search runs locally over these embeddings.
// Read-only in F1-A; brain/signatures + RW come in later phases.
export interface LocalCart {
  name: string                  // filename minus .cart.npz
  filename: string              // original filename for display
  embeddings: Float32Array      // flat N*768 row-major
  embeddingsShape: number[]     // [N, 768]
  passages: string[]            // length N
  // Provenance v1 sidecar — source filename per pattern. Populated for
  // browser-built carts 2026-06-15+ that include source_paths.npy in their
  // .cart.npz. null for legacy server-built carts; cosineSearchLocal will
  // skip populating SearchResult.source_path in that case, and ResultCard
  // will hide the source line. See CC_cart-provenance-schema_2026-06-15
  // for v2 schema upgrade plan (h-row source_idx + strings table).
  sourcePaths: string[] | null  // length N or null
  sizeBytes: number
  mountedAt: number             // performance.now() timestamp
  // Figures embedded in the .cart.npz under 'figures/<hash>.<ext>'. Keyed by
  // the basename (e.g. 'abc123.png') so ResultCard can look them up via the
  // hash extracted from a figure passage's '[figure | hash: <h>]' header.
  // Empty Map when no figures are embedded.
  figures: Map<string, Uint8Array>
  // Pattern-0 metadata parsed from the cart's pattern0.npy (single-element
  // unicode NPY holding a JSON payload; see cart-builder-v2/writer/npz.ts).
  // Populated for browser-built carts 2026-07-02+ that carry the sidecar;
  // null for older carts, in which case Pattern0TocPanel falls back to the
  // hippocampus-derived stats view.
  pattern0Meta: LocalCartPattern0Meta | null
  // Day 2 sidecar — parallel to `passages`, one record per pattern with
  // content_type + type-specific extras (image_b64 for graphics, html for
  // tables). Populated for carts built by the 2026-07-05+ pipeline; null for
  // legacy carts (no thumbnails but everything else works).
  perPatternMeta: LocalCartPatternMeta[] | null
  // In-memory tombstones for Edit Carts support on LocalCart (Andy 2026-06-16
  // PM: the demo path requires editing browser-mounted carts on the public
  // VPS where the droplet can't reach the user's local filesystem). Tombstoned
  // idx are filtered out of cosineSearchLocal results so they stop appearing
  // in search; can be restored via localCartRestore. Persisted by writing the
  // mutated cart back via showSaveFilePicker (Save action) -- this in-memory
  // state is the editable representation; the on-disk file is the snapshot.
  tombstones: Set<number>
  dirty: boolean                // true if tombstones added/restored since mount
}

// One step in the walk trail. Step 0 is always the original text query that
// kicked off the trail. Subsequent steps are walk-from-here anchors.
export interface WalkStep {
  kind: 'query' | 'walk'
  label: string         // text query OR walk-anchor title
  idx?: number          // walk-anchor result idx (undefined for kind='query')
  query: string         // store.query value at this step (for SearchBar restore)
  results: SearchResult[]
  searchModeLabel: string
  searchElapsed: number
}

// Top-level screen state (nav rail picks which screen renders in the main area).
// 'search' is the default; the original VPS 1.0 search/CRUD experience. Other
// screens are stubbed in this iteration and fleshed out incrementally.
export type ActiveScreen = 'search' | 'overview' | 'cartBuilder' | 'crud' | 'sql' | 'settings'

// Desktop Cart Builder helper — detection + pairing state. The exe (Day 1)
// runs a local FastAPI on http://127.0.0.1:7878 wrapping the same
// /api/cartbuilder/* router the VPS backend exposes. When paired, Cart
// Builder API calls swap base to loopback + attach a Bearer token so builds
// run natively on the user's GPU. See docs/vps-internal/Vector+ Desktop
// Builder Day 1 Spec.md for the endpoint contract.
export type DesktopHelperState =
  | 'unknown'
  | 'detecting'
  | 'not-found'
  | 'detected-unpaired'
  | 'detected-paired'

export interface DesktopHelperCapabilities {
  exe: string
  version?: string
  capabilities: string[]
  port: number
  router_prefix: string
}

// localStorage key for the persisted pairing token. The value is regenerated
// by the exe on first launch and printed to its console; user pastes it into
// the pairing modal.
// Note: SAME key is read by Image Builder (Day 2) — the pairing token is
// shared between both Builders per image-builder/auth.py, so one pairing
// unlocks both.
export const DESKTOP_HELPER_TOKEN_KEY = 'vp-desktop-token'
// Fixed for MVP; capabilities.port could override in the future if the exe
// falls back to a different port (7879+) because 7878 was busy.
const DESKTOP_HELPER_ORIGIN = 'http://127.0.0.1:7878'

// Image Builder — Day 2 sibling exe on 127.0.0.1:7879. Same state shape as
// Desktop Helper: unknown → detecting → not-found | detected-unpaired |
// detected-paired. The token is shared (see DESKTOP_HELPER_TOKEN_KEY), so
// 'detected-unpaired' only appears when the user has never paired either
// Builder; otherwise Image Builder inherits the already-paired state on
// detect.
export type ImageBuilderState =
  | 'unknown'
  | 'detecting'
  | 'not-found'
  | 'detected-unpaired'
  | 'detected-paired'

export interface ImageBuilderCapabilities {
  exe: string
  version?: string
  capabilities: string[]
  port: number
  supported_formats: string[]
}

interface AppState {
  // Active screen (nav rail)
  activeScreen: ActiveScreen
  setActiveScreen: (screen: ActiveScreen) => void

  // Status
  status: StatusResponse | null
  statusLoading: boolean

  // Cartridges
  cartridges: CartridgeInfo[]
  mounting: boolean

  // Search
  searchMode: SearchMode
  blendAlpha: number
  topK: number
  query: string
  results: SearchResult[]
  searchModeLabel: string
  searchElapsed: number
  searching: boolean

  // Deleted
  deletedPatterns: DeletedPattern[]

  // Delete confirmation -- only one card can be armed at a time
  confirmDeleteIdx: number | null
  setConfirmDeleteIdx: (idx: number | null) => void

  // Add passage
  addingPassage: boolean

  // WebGPU Associate (browser-side physics — Phase 2e). When the server
  // doesn't have CUDA but the user has WebGPU + the mounted cart has a brain
  // on disk, Associate runs in the browser on the user's GPU.
  webgpuStatus: 'detecting' | 'available' | 'unavailable'
  webgpuBrainLoading: boolean
  webgpuBrainLoadedFor: string | null
  webgpuBrainProgress: { loaded: number; total: number; stage: string } | null
  detectWebGpuOnce: () => Promise<void>

  // Walk-from-here state (W2). When the user clicks Walk on a result card,
  // we anchor a new Associate search on that passage's embedding. The trail
  // is an array of state snapshots: index 0 is the original text query, each
  // subsequent entry is a walk step. The user is always at the LAST entry;
  // the X button jumps back to entry 0; the dropdown lets them jump to any
  // entry without re-running the search (cached results).
  walkTrail: WalkStep[]
  walkFrom: (idx: number, title: string) => Promise<void>
  clearWalk: () => void
  restoreTrailStep: (index: number) => void

  // F1 — Open Cartridge / local carts. Carts the user picked from disk live
  // here; nothing is uploaded. activeLocalCart names the one that's currently
  // selected for search (when null, searches go to the server's mounted cart).
  localCarts: Map<string, LocalCart>
  activeLocalCart: string | null
  localCartLoading: boolean
  mountLocalCart: (file: File) => Promise<{ success: boolean; message: string }>
  unmountLocalCart: () => void
  selectLocalCart: (name: string) => void
  // LocalCart edit operations (Andy 2026-06-16 PM: Edit Carts must work on
  // browser-mounted carts on the public droplet). Each operation mutates the
  // active LocalCart's in-memory state and replaces it in the localCarts Map
  // immutably so React re-renders. Save persists by re-downloading via
  // showSaveFilePicker (user picks where to save the mutated cart).
  localCartTombstone: (idx: number) => void
  localCartRestore: (idx: number) => void
  localCartTombstoneBySource: (sourcePath: string) => number[]
  localCartListSources: () => Array<{ sourcePath: string; count: number; activeCount: number }>
  // Windowed list of passages belonging to a single source file — powers the
  // Edit Carts File→Passages drill-down (Andy 2026-06-28 spec). Walks the
  // cart's sourcePaths, slices a page's worth of matching idx, returns
  // enough per-row info for the drill-down UI to render preview + tombstone
  // status + reuse editorOriginalIdx for per-passage edits.
  localCartListPassagesBySource: (sourcePath: string, offset: number, limit: number) => {
    total: number
    activeCount: number
    tombstonedCount: number
    passages: Array<{ idx: number; title: string; preview: string; tombstoned: boolean }>
  }
  localCartAddPassage: (text: string, source: string) => Promise<{ success: boolean; message: string; idx?: number }>
  localCartSave: () => Promise<{ success: boolean; message: string }>

  // Per-pattern metadata for the SERVER-mounted cart (sandbox upload path).
  // Andy 2026-07-06 AM: fetched via /api/cart/per-pattern-meta after mount.
  // Parallel-indexed to server's passages. Null when nothing is mounted
  // server-side, or when the mounted cart lacks per_pattern_meta.npy
  // (legacy cart). Rendered by ResultCard, PassageModal, Pattern0TocPanel
  // drill-down via a resolvePatternMeta helper that checks EITHER the
  // active LocalCart's perPatternMeta OR this server mirror.
  sandboxPerPatternMeta: LocalCartPatternMeta[] | null
  fetchSandboxPerPatternMeta: () => Promise<void>

  // Strict keyword filter
  strictMode: boolean
  setStrictMode: (strict: boolean) => void

  // Exact phrase match filter
  exactMatch: boolean
  setExactMatch: (exact: boolean) => void

  // Pattern-0 TOC visibility on Search tab (Andy 2026-07-02). Old flow:
  // TOC shows on first cart mount, replaced by results after first search,
  // user can navigate back via the Pattern-0 button in SearchToolbar.
  // Reset true on cart mount/unmount/select; false when search runs.
  showTocPanel: boolean
  setShowTocPanel: (show: boolean) => void

  // Editor -- used for both Add Passage and Edit Passage
  editorOpen: boolean
  editorText: string
  editorOriginalIdx: number | null  // null = new passage, number = editing existing
  editorOriginalText: string        // original text for change detection
  openEditor: (text?: string, idx?: number) => void
  closeEditor: () => void
  setEditorText: (text: string) => void
  saveEditor: () => Promise<{ success: boolean; message: string }>

  // Passage modal (full reader with PREV/NEXT navigation + split-cart load-source)
  modalOpen: boolean
  modalPassage: {
    idx: number
    title: string
    full_text: string
    prev_idx: number | null
    next_idx: number | null
    source_db?: string | null  // present when the mounted cart is split-cart
    paper_id?: string | null   // populated AFTER load-source-from-DB CTA fires
    // Provenance v1 sidecar — source filename of THIS pattern. Populated for
    // local-mounted carts whose .npz includes source_paths.npy. Used to
    // constrain PREV/NEXT navigation to stay within the parent file (see
    // navigateModal local-cart branch below). Null/undefined for legacy carts
    // or server-mounted carts (no client-side per-pattern source info),
    // which fall back to unconstrained neighbor walking.
    source_path?: string | null
  } | null
  modalLoading: boolean
  openModal: (result: SearchResult) => void
  closeModal: () => void
  navigateModal: (idx: number) => Promise<void>
  loadSourceForCurrentPassage: () => Promise<void>  // split-cart RAG+ load-source CTA

  // Membox visualizer
  memboxPanelOpen: boolean
  memboxCarts: MemboxCartInfo[]
  selectedMemboxCart: string | null
  memboxStatus: MemboxStatus | null
  memboxStatusLoading: boolean
  toggleMemboxPanel: () => void
  fetchMemboxCarts: () => Promise<void>
  selectMemboxCart: (cartId: string | null) => void
  fetchMemboxStatus: (cartId: string) => Promise<void>

  // Desktop Cart Builder helper
  desktopHelperState: DesktopHelperState
  desktopHelperCapabilities: DesktopHelperCapabilities | null
  desktopHelperToken: string | null
  desktopHelperError: string | null            // last pair error surface
  desktopHelperPairModalOpen: boolean
  detectDesktopHelper: () => Promise<void>
  pairDesktopHelper: (token: string) => Promise<{ success: boolean; message: string }>
  unpairDesktopHelper: () => void
  openDesktopHelperPairModal: () => void
  closeDesktopHelperPairModal: () => void

  // Image Builder — Day 2. Same detection shape as Desktop Helper; token is
  // shared (see DESKTOP_HELPER_TOKEN_KEY note). detectImageBuilder is called
  // from CartBuilderScreen mount alongside the Desktop Helper probe.
  imageBuilderState: ImageBuilderState
  imageBuilderCapabilities: ImageBuilderCapabilities | null
  detectImageBuilder: () => Promise<void>

  // Actions
  fetchStatus: () => Promise<void>
  fetchCartridges: () => Promise<void>
  mount: (filename: string) => Promise<void>
  unmount: () => Promise<void>
  saveCartridge: () => Promise<{ success: boolean; message: string }>
  toggleLock: () => Promise<void>
  setSearchMode: (mode: SearchMode) => void
  setBlendAlpha: (alpha: number) => void
  setTopK: (k: number) => void
  doSearch: (query: string) => Promise<void>
  addPassage: (text: string) => Promise<{ success: boolean; message: string }>
  deleteResult: (idx: number) => Promise<void>
  restoreResult: (idx: number) => Promise<void>
  fetchDeleted: () => Promise<void>
  navigateToPattern: (idx: number) => Promise<void>
}

export const useAppStore = create<AppState>((set, get) => ({
  activeScreen: 'search' as ActiveScreen,
  setActiveScreen: (screen) => set({ activeScreen: screen }),

  status: null,
  statusLoading: false,
  cartridges: [],
  mounting: false,

  searchMode: 'hamming',
  blendAlpha: 0.7,
  topK: 10,
  query: '',
  results: [],
  searchModeLabel: '',
  searchElapsed: 0,
  searching: false,

  webgpuStatus: 'detecting',
  webgpuBrainLoading: false,
  webgpuBrainLoadedFor: null,
  webgpuBrainProgress: null,
  detectWebGpuOnce: async () => {
    const available = await detectWebGPU()
    set({ webgpuStatus: available ? 'available' : 'unavailable' })
  },

  // ---- Desktop Cart Builder helper ----
  // Hydrate token from localStorage synchronously so a page reload with a
  // still-valid token surfaces as paired the moment detection succeeds
  // (no extra pair round-trip needed for the happy path — the /api call's
  // Bearer header is authoritative; if the token's stale we 401 and drop back).
  desktopHelperState: 'unknown',
  desktopHelperCapabilities: null,
  desktopHelperToken: (() => {
    try {
      return localStorage.getItem(DESKTOP_HELPER_TOKEN_KEY)
    } catch {
      return null
    }
  })(),
  desktopHelperError: null,
  desktopHelperPairModalOpen: false,

  detectDesktopHelper: async () => {
    const { desktopHelperState } = get()
    // Avoid overlapping probes if the tab remounts fast.
    if (desktopHelperState === 'detecting') return
    set({ desktopHelperState: 'detecting', desktopHelperError: null })
    // 1 sec timeout — fast-fail when the exe isn't running so the tab render
    // isn't gated on a stalled connection.
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1000)
    try {
      const healthResp = await fetch(`${DESKTOP_HELPER_ORIGIN}/health`, {
        method: 'GET',
        signal: controller.signal,
      })
      if (!healthResp.ok) {
        set({ desktopHelperState: 'not-found', desktopHelperCapabilities: null })
        return
      }
      // Follow-up capabilities call gets the router_prefix + port so the API
      // client can build the correct base URL. Same 1-sec budget — reuse the
      // remaining time from the shared controller so the whole probe caps
      // at ~1 sec worst case.
      const capsResp = await fetch(`${DESKTOP_HELPER_ORIGIN}/capabilities`, {
        method: 'GET',
        signal: controller.signal,
      })
      if (!capsResp.ok) {
        set({ desktopHelperState: 'not-found', desktopHelperCapabilities: null })
        return
      }
      const caps = (await capsResp.json()) as DesktopHelperCapabilities
      const token = get().desktopHelperToken
      // If we have a persisted token, optimistically promote to paired. The
      // first authenticated call from cartbuilder.ts will 401 (via the shared
      // handler) if the exe restarted with a new token — which flips us back
      // to 'detected-unpaired' + surfaces the re-pair prompt.
      set({
        desktopHelperCapabilities: caps,
        desktopHelperState: token ? 'detected-paired' : 'detected-unpaired',
      })
    } catch {
      // Abort, network error, CORS — all indistinguishable at this layer
      // and all mean "no exe on 7878." Rare false-negative if the user
      // has a machine so loaded the loopback probe times out; a rerun
      // fixes it.
      set({ desktopHelperState: 'not-found', desktopHelperCapabilities: null })
    } finally {
      clearTimeout(timer)
    }
  },

  pairDesktopHelper: async (token: string) => {
    const trimmed = token.trim()
    if (!trimmed) {
      const msg = 'Pairing code is empty'
      set({ desktopHelperError: msg })
      return { success: false, message: msg }
    }
    try {
      const resp = await fetch(`${DESKTOP_HELPER_ORIGIN}/pair`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: trimmed }),
      })
      if (resp.status === 401) {
        const msg = 'Pairing code rejected — check the code from the Desktop Builder console'
        set({ desktopHelperError: msg })
        return { success: false, message: msg }
      }
      if (!resp.ok) {
        const msg = `Pair failed: ${resp.status}`
        set({ desktopHelperError: msg })
        return { success: false, message: msg }
      }
      try { localStorage.setItem(DESKTOP_HELPER_TOKEN_KEY, trimmed) } catch { /* private mode etc. */ }
      set({
        desktopHelperToken: trimmed,
        desktopHelperState: 'detected-paired',
        desktopHelperError: null,
        desktopHelperPairModalOpen: false,
      })
      return { success: true, message: 'Paired with Desktop Cart Builder' }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Pair request failed'
      set({ desktopHelperError: msg })
      return { success: false, message: msg }
    }
  },

  unpairDesktopHelper: () => {
    try { localStorage.removeItem(DESKTOP_HELPER_TOKEN_KEY) } catch { /* ignore */ }
    // Preserve the capabilities so the badge stays on 'detected-unpaired'
    // rather than falling back to GPU/WebGPU/CPU — the exe is still there,
    // the user just isn't authorized. detectDesktopHelper() sets 'not-found'
    // when the exe actually disappears.
    const { desktopHelperCapabilities } = get()
    set({
      desktopHelperToken: null,
      desktopHelperState: desktopHelperCapabilities ? 'detected-unpaired' : 'not-found',
    })
  },

  openDesktopHelperPairModal: () => set({ desktopHelperPairModalOpen: true, desktopHelperError: null }),
  closeDesktopHelperPairModal: () => set({ desktopHelperPairModalOpen: false }),

  // ---- Image Builder (Day 2) ----
  // Loopback probe on 127.0.0.1:7879. Fires alongside the Desktop Helper probe
  // on Cart Builder tab mount (Promise.all in CartBuilderScreen effect); same
  // 1-sec abort budget so the tab render isn't gated on a stalled connection.
  // Detection state falls through to 'not-found' on any failure — abort, CORS,
  // network — since all three mean "no exe on 7879" for our purposes.
  imageBuilderState: 'unknown',
  imageBuilderCapabilities: null,

  detectImageBuilder: async () => {
    const { imageBuilderState, desktopHelperToken } = get()
    if (imageBuilderState === 'detecting') return
    set({ imageBuilderState: 'detecting' })
    // Same 1-sec fast-fail budget as Desktop Helper. Uses shared imageBuilder
    // client so the origin / port constant lives in one place.
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1000)
    try {
      const result = await probeImageBuilder(controller.signal)
      if (!result.ok) {
        set({ imageBuilderState: 'not-found', imageBuilderCapabilities: null })
        return
      }
      // Token is shared with Desktop Helper — inherit its paired state. When a
      // user has already paired one Builder, the other lights up 'paired'
      // automatically. If neither has a token, we land 'detected-unpaired'
      // and the user pairs via the Desktop Helper flow (image-builder/auth.py
      // reads the same ~/.vector-plus/token file).
      set({
        imageBuilderCapabilities: result.capabilities,
        imageBuilderState: desktopHelperToken ? 'detected-paired' : 'detected-unpaired',
      })
    } catch {
      set({ imageBuilderState: 'not-found', imageBuilderCapabilities: null })
    } finally {
      clearTimeout(timer)
    }
  },

  localCarts: new Map(),
  activeLocalCart: null,
  localCartLoading: false,

  sandboxPerPatternMeta: null,
  fetchSandboxPerPatternMeta: async () => {
    try {
      const resp = await api.getCartPerPatternMeta()
      if (!resp.mounted || !resp.records || resp.records.length === 0) {
        set({ sandboxPerPatternMeta: null })
        return
      }
      set({ sandboxPerPatternMeta: resp.records as LocalCartPatternMeta[] })
    } catch {
      // Endpoint may be missing on legacy droplets; fall through silently.
      set({ sandboxPerPatternMeta: null })
    }
  },

  mountLocalCart: async (file: File) => {
    set({ localCartLoading: true })
    try {
      const cart = await parseCartFile(file)
      // Rich Pattern-0 path (Andy 2026-07-02): when the NPZ lacks a
      // source_paths.npy entry — legacy server-built carts, or any cart built
      // before browser-side provenance landed — synthesize sourcePaths from
      // each passage's first line. Mirrors the backend's load_cart source
      // extraction (splits first line, strips " (part N/M)" suffix) so the
      // Source Files panel + Pattern0TocPanel derived-stats path get a
      // populated array instead of the "provenance sidecar missing" empty
      // state. Only synthesize when genuinely absent — never clobber real data.
      const sourcePathsWereSynthesized = !cart.sourcePaths
      if (sourcePathsWereSynthesized) {
        cart.sourcePaths = cart.passages.map((passage, i) => {
          const firstLine = passage.includes('\n') ? passage.split('\n')[0] : ''
          const trimmed = firstLine.includes(' (part ')
            ? firstLine.split(' (part ')[0].trim()
            : firstLine.trim()
          return trimmed || `passage_${i}`
        })
      }
      const next = new Map(get().localCarts)
      next.set(cart.name, cart)
      set({
        localCarts: next,
        activeLocalCart: cart.name,
        results: [],
        query: '',
        searchModeLabel: '',
        searchElapsed: 0,
        walkTrail: [],
        showTocPanel: true,
      })
      return { success: true, message: `Mounted ${cart.name} (${cart.passages.length.toLocaleString()} passages)` }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      console.error('[mountLocalCart] FAIL:', e)
      return { success: false, message: `Local cart mount failed: ${msg}` }
    } finally {
      set({ localCartLoading: false })
    }
  },
  unmountLocalCart: () => {
    set({
      activeLocalCart: null,
      results: [],
      query: '',
      searchModeLabel: '',
      searchElapsed: 0,
      walkTrail: [],
      showTocPanel: true,
    })
  },

  // ---- LocalCart edit operations ----
  // All mutations follow the same pattern: get the active cart, build a new
  // LocalCart with updated state, replace it in a NEW Map, set state.
  // Immutable update so React sees the change. Dirty flag set so the OpenCartBanner
  // shows "unsaved changes" until the user saves.
  localCartTombstone: (idx: number) => {
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return
    const cart = localCarts.get(activeLocalCart)
    if (!cart) return
    if (idx < 0 || idx >= cart.passages.length) return
    const nextTombs = new Set(cart.tombstones)
    nextTombs.add(idx)
    const updatedCart: LocalCart = { ...cart, tombstones: nextTombs, dirty: true }
    const nextCarts = new Map(localCarts)
    nextCarts.set(activeLocalCart, updatedCart)
    set({ localCarts: nextCarts })
  },

  localCartRestore: (idx: number) => {
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return
    const cart = localCarts.get(activeLocalCart)
    if (!cart) return
    if (!cart.tombstones.has(idx)) return
    const nextTombs = new Set(cart.tombstones)
    nextTombs.delete(idx)
    // dirty stays true if there were tombstones before this restore that are
    // still set; if all gone, technically we could clear dirty but it doesn't
    // matter for UX -- user can still Save to commit current state.
    const updatedCart: LocalCart = { ...cart, tombstones: nextTombs, dirty: true }
    const nextCarts = new Map(localCarts)
    nextCarts.set(activeLocalCart, updatedCart)
    set({ localCarts: nextCarts })
  },

  // Tombstone every passage whose sourcePath matches the given source.
  // Returns the count of newly-tombstoned passages (for log/toast messaging).
  // Used by the demo flow's "delete the specific single file" step.
  localCartTombstoneBySource: (sourcePath: string): number[] => {
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return []
    const cart = localCarts.get(activeLocalCart)
    if (!cart || !cart.sourcePaths) return []
    const nextTombs = new Set(cart.tombstones)
    const added: number[] = []
    for (let i = 0; i < cart.sourcePaths.length; i++) {
      if (cart.sourcePaths[i] === sourcePath && !nextTombs.has(i)) {
        nextTombs.add(i)
        added.push(i)
      }
    }
    if (added.length === 0) return added
    const updatedCart: LocalCart = { ...cart, tombstones: nextTombs, dirty: true }
    const nextCarts = new Map(localCarts)
    nextCarts.set(activeLocalCart, updatedCart)
    set({ localCarts: nextCarts })
    return added
  },

  // List unique source files in the cart with active + tombstoned counts.
  // Used by the Delete panel UI to render a clickable list of source files.
  localCartListSources: () => {
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return []
    const cart = localCarts.get(activeLocalCart)
    if (!cart || !cart.sourcePaths) return []
    const counts = new Map<string, { count: number; activeCount: number }>()
    for (let i = 0; i < cart.sourcePaths.length; i++) {
      const sp = cart.sourcePaths[i]
      if (!sp) continue
      const c = counts.get(sp) ?? { count: 0, activeCount: 0 }
      c.count++
      if (!cart.tombstones.has(i)) c.activeCount++
      counts.set(sp, c)
    }
    return Array.from(counts.entries()).map(([sourcePath, c]) => ({
      sourcePath,
      count: c.count,
      activeCount: c.activeCount,
    }))
  },

  // Windowed drill-down of a single source file's passages. Walks the cart's
  // sourcePaths, collects every matching idx (preserving cart order), then
  // slices the requested page. Returns preview + title + tombstone state per
  // row so the drill-down UI can render without any additional lookups.
  localCartListPassagesBySource: (sourcePath: string, offset: number, limit: number) => {
    const empty = { total: 0, activeCount: 0, tombstonedCount: 0, passages: [] }
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return empty
    const cart = localCarts.get(activeLocalCart)
    if (!cart || !cart.sourcePaths) return empty
    const matching: number[] = []
    let activeCount = 0
    let tombstonedCount = 0
    for (let i = 0; i < cart.sourcePaths.length; i++) {
      if (cart.sourcePaths[i] !== sourcePath) continue
      matching.push(i)
      if (cart.tombstones.has(i)) tombstonedCount++
      else activeCount++
    }
    const start = Math.max(0, offset)
    const end = Math.min(matching.length, start + Math.max(0, limit))
    const pageIdx = matching.slice(start, end)
    const passages = pageIdx.map((idx) => {
      const text = cart.passages[idx] ?? ''
      const firstNewline = text.indexOf('\n')
      const rawTitle = firstNewline > 0 ? text.slice(0, firstNewline) : text.slice(0, 120)
      const rawBody = firstNewline > 0 ? text.slice(firstNewline + 1) : text
      const title = rawTitle.trim() || `#${idx}`
      const preview = rawBody.replace(/\s+/g, ' ').trim().slice(0, 160)
      return { idx, title, preview, tombstoned: cart.tombstones.has(idx) }
    })
    return {
      total: matching.length,
      activeCount,
      tombstonedCount,
      passages,
    }
  },

  // Add a new passage to the active LocalCart. Embeds the text via the
  // backend's /api/embed endpoint (works for public droplet too -- the
  // embedding model is server-side), then appends to passages/embeddings/
  // sourcePaths arrays. Marks cart dirty. Demo step 8: "add file with
  // other specific knowledge" flows through here -- user provides text +
  // source label, the new passage becomes immediately searchable in
  // cosineSearchLocal.
  localCartAddPassage: async (text: string, source: string) => {
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return { success: false, message: 'No active local cart' }
    const cart = localCarts.get(activeLocalCart)
    if (!cart) return { success: false, message: 'Active local cart not found' }
    const trimmedText = text.trim()
    if (!trimmedText) return { success: false, message: 'Cannot add empty passage' }
    try {
      const apiBase = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'
      const embResp = await fetch(`${apiBase}/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmedText }),
      })
      if (!embResp.ok) {
        return { success: false, message: `Embed request failed: ${embResp.status}` }
      }
      const embData = await embResp.json()
      const newEmb = new Float32Array(embData.embedding)
      const [, dim] = cart.embeddingsShape
      if (newEmb.length !== dim) {
        return { success: false, message: `Embed dim mismatch: got ${newEmb.length}, expected ${dim}` }
      }
      // Append to flat embeddings array (immutable -- alloc new larger one).
      const oldCount = cart.passages.length
      const newCount = oldCount + 1
      const newEmbeddings = new Float32Array(newCount * dim)
      newEmbeddings.set(cart.embeddings)
      newEmbeddings.set(newEmb, oldCount * dim)
      const newPassages = [...cart.passages, trimmedText]
      const newSourcePaths = cart.sourcePaths
        ? [...cart.sourcePaths, source || '<inline>']
        : null
      const updatedCart: LocalCart = {
        ...cart,
        embeddings: newEmbeddings,
        embeddingsShape: [newCount, dim],
        passages: newPassages,
        sourcePaths: newSourcePaths,
        dirty: true,
      }
      const nextCarts = new Map(localCarts)
      nextCarts.set(activeLocalCart, updatedCart)
      set({ localCarts: nextCarts })
      return {
        success: true,
        message: `Added passage at idx #${oldCount} (source: ${source || '<inline>'})`,
        idx: oldCount,
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error'
      return { success: false, message: `Add failed: ${msg}` }
    }
  },

  // Save the active LocalCart back to disk by triggering a re-download via
  // showSaveFilePicker (Chrome/Edge/Opera) or anchor click (Firefox/Safari).
  // The saved bytes have tombstones APPLIED -- tombstoned passages are removed
  // from the passages array, embeddings, and sourcePaths. User can then re-mount
  // the saved file to confirm the changes persist.
  // Implementation is in lib/localCart.ts (saveCartToDisk).
  localCartSave: async () => {
    const { activeLocalCart, localCarts } = get()
    if (!activeLocalCart) return { success: false, message: 'No active local cart' }
    const cart = localCarts.get(activeLocalCart)
    if (!cart) return { success: false, message: 'Active local cart not found' }
    if (!cart.dirty && cart.tombstones.size === 0) {
      return { success: false, message: 'Nothing to save (no changes)' }
    }
    try {
      const { saveLocalCartToDisk } = await import('../lib/localCart')
      const result = await saveLocalCartToDisk(cart)
      if (result.success) {
        // Mark clean after successful save -- the on-disk snapshot now matches.
        const updatedCart: LocalCart = { ...cart, dirty: false }
        const nextCarts = new Map(localCarts)
        nextCarts.set(activeLocalCart, updatedCart)
        set({ localCarts: nextCarts })
      }
      return result
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown save error'
      return { success: false, message: `Save failed: ${msg}` }
    }
  },
  selectLocalCart: (name: string) => {
    const { localCarts } = get()
    if (!localCarts.has(name)) return
    set({
      activeLocalCart: name,
      results: [],
      query: '',
      searchModeLabel: '',
      searchElapsed: 0,
      walkTrail: [],
      showTocPanel: true,
    })
  },

  walkTrail: [],
  walkFrom: async (idx: number, title: string) => {
    const { status, webgpuStatus, cartridges, topK, query, results, walkTrail, searchModeLabel, searchElapsed } = get()
    const mountedName = status?.mounted_cartridge ?? null
    if (!mountedName) {
      console.warn('walkFrom called with no cart mounted')
      return
    }
    const mountedCart = cartridges.find((c) => c.name === mountedName)
    const cartHasBrain = !!mountedCart?.has_brain
    const useWebGpu =
      webgpuStatus === 'available' && !status?.gpu_available && cartHasBrain

    // If this is the first walk in a trail, capture the current state as the
    // ROOT (text query) step. After that, every walkFrom call appends a walk step.
    const startingTrail: WalkStep[] = walkTrail.length === 0
      ? [{
          kind: 'query',
          label: query || '(empty query)',
          query,
          results,
          searchModeLabel,
          searchElapsed,
        }]
      : walkTrail

    set({
      searching: true,
      searchModeLabel: `Walking from #${idx}…`,
    })

    try {
      if (useWebGpu) {
        if (!isBrainLoadedFor(mountedName)) {
          set({ webgpuBrainLoading: true, webgpuBrainProgress: null })
          await loadBrainForCart(mountedName, (p) => {
            set({ webgpuBrainProgress: { loaded: p.loaded, total: p.total, stage: p.stage } })
          })
          set({ webgpuBrainLoading: false, webgpuBrainLoadedFor: mountedName, webgpuBrainProgress: null })
        }
        const embResp = await fetch(
          `${import.meta.env.VITE_API_BASE || '/api'}/cartridges/${encodeURIComponent(mountedName)}/embedding/${idx}`,
        )
        if (!embResp.ok) throw new Error(`Embedding fetch failed: ${embResp.status}`)
        const embData = await embResp.json()
        const t0 = performance.now()
        const walkResults = await runWebGpuAssociate({
          queryEmbedding: embData.embedding,
          cartName: mountedName,
          topK,
          poolSize: 50,
          onStatus: (msg) => set({ searchModeLabel: msg }),
        })
        const newLabel = `Walk · WebGPU from "${title.slice(0, 40)}${title.length > 40 ? '…' : ''}"`
        const newElapsed = Math.round(performance.now() - t0)
        set({
          results: walkResults,
          searchModeLabel: newLabel,
          searchElapsed: newElapsed,
          walkTrail: [...startingTrail, {
            kind: 'walk', label: title, idx,
            query: startingTrail[0].query,
            results: walkResults,
            searchModeLabel: newLabel,
            searchElapsed: newElapsed,
          }],
        })
      } else {
        // Server-side walk via /api/walk-from
        const t0 = performance.now()
        const resp = await fetch(`${import.meta.env.VITE_API_BASE || '/api'}/walk-from`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cart_name: mountedName, idx, top_k: topK }),
        })
        if (!resp.ok) throw new Error(`Walk-from failed: ${resp.status}`)
        const data = await resp.json()
        const newLabel = `Walk · Server from "${title.slice(0, 40)}${title.length > 40 ? '…' : ''}"`
        const newElapsed = Math.round(performance.now() - t0)
        set({
          results: data.results,
          searchModeLabel: newLabel,
          searchElapsed: newElapsed,
          walkTrail: [...startingTrail, {
            kind: 'walk', label: title, idx,
            query: startingTrail[0].query,
            results: data.results,
            searchModeLabel: newLabel,
            searchElapsed: newElapsed,
          }],
        })
      }
    } catch (e) {
      console.error('walkFrom failed:', e)
      // On error: leave any existing trail in place but mark mode label so the
      // user knows the step didn't take. Don't truncate the trail — they may
      // want to retry or jump back via the dropdown.
      set({ searchModeLabel: 'Walk failed' })
    } finally {
      set({ searching: false, webgpuBrainLoading: false, webgpuBrainProgress: null })
    }
  },
  clearWalk: () => {
    // Jump back to step 0 (the original text query) using its cached results —
    // no re-running the search. Then empty the trail entirely.
    const { walkTrail } = get()
    if (walkTrail.length === 0) return
    const root = walkTrail[0]
    set({
      walkTrail: [],
      query: root.query,
      results: root.results,
      searchModeLabel: root.searchModeLabel,
      searchElapsed: root.searchElapsed,
    })
  },
  restoreTrailStep: (index: number) => {
    // Jump to any step in the trail using its cached results. The trail is
    // truncated to [0..index] so subsequent walks branch from this point.
    // If the user lands on step 0 we collapse the trail (back to text-search mode).
    const { walkTrail } = get()
    if (index < 0 || index >= walkTrail.length) return
    const step = walkTrail[index]
    set({
      walkTrail: index === 0 ? [] : walkTrail.slice(0, index + 1),
      query: step.query,
      results: step.results,
      searchModeLabel: step.searchModeLabel,
      searchElapsed: step.searchElapsed,
    })
  },

  deletedPatterns: [],

  addingPassage: false,

  strictMode: false,
  setStrictMode: (strict) => set({ strictMode: strict }),

  exactMatch: false,
  setExactMatch: (exact) => set({ exactMatch: exact }),

  showTocPanel: true,
  setShowTocPanel: (show) => set({ showTocPanel: show }),

  editorOpen: false,
  editorText: '',
  editorOriginalIdx: null,
  editorOriginalText: '',
  openEditor: (text = '', idx) => set({
    editorOpen: true,
    editorText: text,
    editorOriginalIdx: idx ?? null,
    editorOriginalText: text,
  }),
  closeEditor: () => set({
    editorOpen: false,
    editorText: '',
    editorOriginalIdx: null,
    editorOriginalText: '',
  }),
  setEditorText: (text) => set({ editorText: text }),
  saveEditor: async () => {
    const {
      editorText, editorOriginalIdx,
      addPassage, deleteResult, closeEditor,
      activeLocalCart, localCarts,
      localCartAddPassage, localCartTombstone,
    } = get()
    const text = editorText.trim()
    if (!text) return { success: false, message: 'Text is empty' }

    // LocalCart branch — backend api.addPassage / api.deletePattern only
    // mutate server state, so LocalCart edits made from Edit Carts would
    // silently no-op (Andy 2026-07-02). Route through localCart* actions
    // so the in-memory cart updates + React re-renders + dirty flag flips.
    // Reuse the original passage's sourcePath so the new pattern stays
    // attached to the same file for TOC + drill-down groupings.
    if (activeLocalCart) {
      const cart = localCarts.get(activeLocalCart)
      set({ addingPassage: true })
      try {
        const source =
          editorOriginalIdx !== null && cart?.sourcePaths
            ? cart.sourcePaths[editorOriginalIdx] ?? '<inline>'
            : '<inline>'
        const resp = await localCartAddPassage(text, source)
        if (!resp.success) return resp
        if (editorOriginalIdx !== null) localCartTombstone(editorOriginalIdx)
        closeEditor()
        return resp
      } finally {
        set({ addingPassage: false })
      }
    }

    // Save new passage
    const resp = await addPassage(text)
    if (!resp.success) return resp

    // If editing, tombstone the old pattern
    if (editorOriginalIdx !== null) {
      await deleteResult(editorOriginalIdx)
    }

    closeEditor()
    return resp
  },

  confirmDeleteIdx: null,
  setConfirmDeleteIdx: (idx) => set({ confirmDeleteIdx: idx }),

  // Passage modal
  modalOpen: false,
  modalPassage: null,
  modalLoading: false,
  openModal: (result) => {
    // Constrain the initial PREV/NEXT to stay within the parent file for
    // local carts that carry per-pattern source_paths. Only clip when we
    // actually know the neighbor's source: unknown => preserve today's
    // unconstrained behavior (legacy carts, server-mounted carts).
    const { activeLocalCart, localCarts } = get()
    const cart = activeLocalCart ? localCarts.get(activeLocalCart) : null
    const sourcePaths = cart?.sourcePaths ?? null
    const currentSource = result.source_path ?? sourcePaths?.[result.idx] ?? null
    let prev_idx = result.prev_idx
    let next_idx = result.next_idx
    if (sourcePaths && currentSource != null) {
      if (prev_idx != null && sourcePaths[prev_idx] !== currentSource) prev_idx = null
      if (next_idx != null && sourcePaths[next_idx] !== currentSource) next_idx = null
    }
    set({
      modalOpen: true,
      modalPassage: {
        idx: result.idx,
        title: result.title,
        full_text: result.full_text,
        prev_idx,
        next_idx,
        // Carry split-cart hints from the search result so the modal can render
        // the "Load full passage from <db>" CTA. paper_id is only set after the
        // user clicks load-source (via loadSourceForCurrentPassage below).
        source_db: result.source_db ?? null,
        paper_id: result.paper_id ?? null,
        source_path: currentSource,
      },
    })
  },
  closeModal: () => set({ modalOpen: false, modalPassage: null }),
  navigateModal: async (idx: number) => {
    set({ modalLoading: true })
    try {
      // Local cart path — look up the passage from the in-memory cart so
      // PREV/NEXT navigation works without any backend round-trip.
      const { activeLocalCart, localCarts } = get()
      if (activeLocalCart) {
        const cart = localCarts.get(activeLocalCart)
        if (cart && idx >= 0 && idx < cart.passages.length) {
          const passage = cart.passages[idx] ?? ''
          const firstNewline = passage.indexOf('\n')
          const title = (firstNewline > 0 ? passage.slice(0, firstNewline) : passage.slice(0, 80)).trim()
          const total = cart.passages.length
          // Constrain PREV/NEXT to same-source (parent-file boundary) when the
          // cart carries source_paths.npy. Legacy carts (sourcePaths == null)
          // fall through to unconstrained neighbor walking.
          const sourcePaths = cart.sourcePaths ?? null
          const currentSource = sourcePaths?.[idx] ?? null
          const rawPrev = idx > 0 ? idx - 1 : null
          const rawNext = idx < total - 1 ? idx + 1 : null
          const prev_idx = sourcePaths && currentSource != null && rawPrev != null && sourcePaths[rawPrev] !== currentSource
            ? null
            : rawPrev
          const next_idx = sourcePaths && currentSource != null && rawNext != null && sourcePaths[rawNext] !== currentSource
            ? null
            : rawNext
          set({
            modalPassage: {
              idx,
              title,
              full_text: passage,
              prev_idx,
              next_idx,
              source_db: null,
              paper_id: null,
              source_path: currentSource,
            },
          })
          return
        }
      }
      const pattern = await api.getPattern(idx)
      // Server-mounted carts don't currently return per-pattern source_path;
      // carry the modal's existing source_path forward (unchanged) so any
      // client-side constraint remains stable. When the backend starts
      // returning source_path in PatternResponse, prefer that value.
      const prevSource = get().modalPassage?.source_path ?? null
      set({
        modalPassage: {
          idx: pattern.idx,
          title: pattern.title,
          full_text: pattern.full_text,
          prev_idx: pattern.prev_idx,
          next_idx: pattern.next_idx,
          source_db: pattern.source_db ?? null,
          paper_id: pattern.paper_id ?? null,
          source_path: prevSource,
        },
      })
    } catch (e) {
      console.error('Modal navigate failed:', e)
    } finally {
      set({ modalLoading: false })
    }
  },
  loadSourceForCurrentPassage: async () => {
    // RAG+ split-cart "Load full passage from <db>" CTA. The search response
    // gave us the in-RAM 200-char snippet; this pulls the full passage from
    // the SQLite sidecar via /api/patterns/{idx} (which the backend wires
    // up when the cart is split-cart). After load: full_text is the long
    // version, paper_id is populated.
    const current = useAppStore.getState().modalPassage
    if (!current) return
    set({ modalLoading: true })
    try {
      const pattern = await api.getPattern(current.idx)
      set({
        modalPassage: {
          idx: pattern.idx,
          title: pattern.title,
          full_text: pattern.full_text,
          prev_idx: pattern.prev_idx,
          next_idx: pattern.next_idx,
          source_db: pattern.source_db ?? current.source_db ?? null,
          paper_id: pattern.paper_id ?? null,
          source_path: current.source_path ?? null,
        },
      })
    } catch (e) {
      console.error('Load source failed:', e)
    } finally {
      set({ modalLoading: false })
    }
  },

  // Membox visualizer state
  memboxPanelOpen: false,
  memboxCarts: [],
  selectedMemboxCart: null,
  memboxStatus: null,
  memboxStatusLoading: false,

  toggleMemboxPanel: () => set((state) => ({ memboxPanelOpen: !state.memboxPanelOpen })),

  fetchMemboxCarts: async () => {
    try {
      const carts = await api.fetchMemboxCarts()
      set({ memboxCarts: carts })
    } catch (e) {
      console.error('Membox carts fetch failed:', e)
      set({ memboxCarts: [] })
    }
  },

  selectMemboxCart: (cartId) => {
    set({ selectedMemboxCart: cartId, memboxStatus: null })
    if (cartId) get().fetchMemboxStatus(cartId)
  },

  fetchMemboxStatus: async (cartId: string) => {
    set({ memboxStatusLoading: true })
    try {
      const status = await api.fetchMemboxStatus(cartId)
      set({ memboxStatus: status })
    } catch (e) {
      console.error('Membox status fetch failed:', e)
      set({ memboxStatus: null })
    } finally {
      set({ memboxStatusLoading: false })
    }
  },

  fetchStatus: async () => {
    set({ statusLoading: true })
    try {
      const status = await api.getStatus()
      set({ status })
    } catch (e) {
      console.error('Status fetch failed:', e)
    } finally {
      set({ statusLoading: false })
    }
  },

  fetchCartridges: async () => {
    try {
      const cartridges = await api.getCartridges()
      set({ cartridges })
    } catch (e) {
      console.error('Cartridge list failed:', e)
    }
  },

  mount: async (filename: string) => {
    // Lazy-load toaster from cartBuilderStore so we don't take a hard
    // dependency on it from this store's module init order.
    const pushToast = (kind: 'success' | 'error' | 'info', text: string, ttl?: number) => {
      import('./cartBuilderStore').then((m) => m.useCartBuilderStore.getState().pushToast(kind, text, ttl))
    }
    set({ mounting: true })
    try {
      const resp = await api.mountCartridge(filename)
      if (!resp.success) {
        // Backend returned 200 with success=false (e.g. file not found, manifest
        // mismatch, malformed cart). Previously this fell through silently.
        pushToast('error', `Mount failed: ${resp.message || 'unknown error'}`, 8000)
        return
      }
      await get().fetchStatus()
      await get().fetchCartridges()
      // Fetch per-pattern metadata sidecar so sandbox/server-mounted carts
      // can render graphics + tables the same way LocalCart mounts do
      // (Andy 2026-07-06 AM sandbox-mount parity). Non-blocking on failure —
      // legacy carts have no sidecar and the endpoint returns records=[].
      get().fetchSandboxPerPatternMeta()
      set({
        results: [], query: '', deletedPatterns: [],
        strictMode: false, exactMatch: false,
        walkTrail: [], searchModeLabel: '', searchElapsed: 0,
        showTocPanel: true,
      })
      pushToast('success', `Mounted ${resp.name} (${resp.pattern_count} patterns)`, 4000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Mount request failed'
      console.error('Mount failed:', e)
      pushToast('error', `Mount failed: ${msg}`, 8000)
    } finally {
      set({ mounting: false })
    }
  },

  unmount: async () => {
    try {
      await api.unmountCartridge()
      await get().fetchStatus()
      set({
        results: [], query: '', deletedPatterns: [],
        walkTrail: [], searchModeLabel: '', searchElapsed: 0,
        showTocPanel: true,
        sandboxPerPatternMeta: null,
      })
    } catch (e) {
      console.error('Unmount failed:', e)
    }
  },

  saveCartridge: async () => {
    try {
      const resp = await api.saveCartridge()
      if (resp.success) {
        await get().fetchStatus()
        await get().fetchCartridges()
      }
      return resp
    } catch (e) {
      console.error('Save failed:', e)
      return { success: false, message: e instanceof Error ? e.message : 'Save failed' }
    }
  },

  toggleLock: async () => {
    try {
      const isLocked = get().status?.read_only ?? true
      if (isLocked) {
        await api.unlockCartridge()
      } else {
        await api.lockCartridge()
      }
      await get().fetchStatus()
    } catch (e) {
      console.error('Lock toggle failed:', e)
    }
  },

  setSearchMode: (mode) => {
    set({ searchMode: mode, results: [], searchModeLabel: '', searchElapsed: 0 })
  },
  setBlendAlpha: (alpha) => {
    set({ blendAlpha: alpha })
    // Debounce re-search while slider is being dragged
    clearTimeout((window as any).__blendTimer)
    ;(window as any).__blendTimer = setTimeout(() => {
      const { query } = get()
      if (query) get().doSearch(query)
    }, 300)
  },
  setTopK: (k) => set({ topK: k }),

  doSearch: async (query: string) => {
    const { searchMode, blendAlpha, topK, status, webgpuStatus, webgpuBrainLoadedFor, cartridges, activeLocalCart, localCarts } = get()
    // A fresh text query exits walk-mode — the user is starting over. Also
    // hides the Pattern-0 TOC so results fill the space; the user brings the
    // TOC back via the Pattern-0 button in SearchToolbar.
    set({ searching: true, query, walkTrail: [], showTocPanel: false })

    // F1 — local cart path. The user's file never leaves the browser; we
    // embed the query via /api/embed (server-side Nomic), then cosine-rank
    // against the in-memory embeddings. Fast/Hamming/Smart/Associate aren't
    // available on local carts yet (no brain or sigs locally); we fall back
    // to plain cosine for any mode requested.
    if (activeLocalCart) {
      const cart = localCarts.get(activeLocalCart)
      if (!cart) {
        console.warn('[doSearch] activeLocalCart set but Map lookup failed', activeLocalCart, Array.from(localCarts.keys()))
        set({ searching: false, searchModeLabel: 'Local cart not found' })
        return
      }
      try {
        const t0 = performance.now()
        const apiBase = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'
        const embResp = await fetch(`${apiBase}/embed`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        })
        if (!embResp.ok) throw new Error(`Embed failed: ${embResp.status}`)
        const embData = await embResp.json()
        const queryEmb = new Float32Array(embData.embedding)
        const results = cosineSearchLocal(queryEmb, cart, topK)
        set({
          results,
          searchModeLabel: 'Local cart · cosine (browser)',
          searchElapsed: Math.round(performance.now() - t0),
        })
      } catch (e) {
        console.error('[doSearch] local FAIL:', e)
        set({ results: [], searchModeLabel: 'Local search failed' })
      } finally {
        set({ searching: false })
      }
      return
    }

    // Decide whether browser-side WebGPU Associate should handle this call.
    // Conditions: associate mode + WebGPU available + server lacks CUDA +
    // a cart is mounted + that cart has a brain file we can fetch.
    const mountedName = status?.mounted_cartridge ?? null
    const mountedCart = mountedName ? cartridges.find((c) => c.name === mountedName) : null
    const cartHasBrain = !!mountedCart?.has_brain
    const useWebGpu =
      searchMode === 'associate' &&
      webgpuStatus === 'available' &&
      !status?.gpu_available &&
      !!mountedName &&
      cartHasBrain

    try {
      if (useWebGpu) {
        try {
          const t0 = performance.now()
          if (webgpuBrainLoadedFor !== mountedName && !isBrainLoadedFor(mountedName!)) {
            set({ webgpuBrainLoading: true, webgpuBrainProgress: null })
            await loadBrainForCart(mountedName!, (p) => {
              set({ webgpuBrainProgress: { loaded: p.loaded, total: p.total, stage: p.stage } })
            })
            set({ webgpuBrainLoading: false, webgpuBrainLoadedFor: mountedName, webgpuBrainProgress: null })
          }
          const results = await runWebGpuAssociate({
            query,
            cartName: mountedName!,
            topK,
            poolSize: 50,
            onStatus: (msg) => set({ searchModeLabel: msg }),
          })
          set({
            results,
            searchModeLabel: 'Associate · WebGPU (30f settle)',
            searchElapsed: Math.round(performance.now() - t0),
          })
          return
        } catch (e) {
          console.error('WebGPU Associate failed, falling back to server:', e)
          set({ webgpuBrainLoading: false, webgpuBrainProgress: null })
        }
      }

      const resp = await api.search(query, searchMode, blendAlpha, topK)
      set({
        results: resp.results,
        searchModeLabel: resp.mode,
        searchElapsed: resp.elapsed_ms,
      })
    } catch (e) {
      console.error('Search failed:', e)
    } finally {
      set({ searching: false })
    }
  },

  addPassage: async (text: string) => {
    set({ addingPassage: true })
    try {
      const resp = await api.addPassage(text)
      if (resp.success) {
        await get().fetchStatus()
      }
      return resp
    } catch (e) {
      console.error('Add passage failed:', e)
      return { success: false, message: e instanceof Error ? e.message : 'Failed' }
    } finally {
      set({ addingPassage: false })
    }
  },

  deleteResult: async (idx: number) => {
    try {
      await api.deletePattern(idx)
      set((s) => ({
        results: s.results.filter((r) => r.idx !== idx),
        confirmDeleteIdx: null,
      }))
      await get().fetchStatus()
      await get().fetchDeleted()
    } catch (e) {
      console.error('Delete failed:', e)
    }
  },

  restoreResult: async (idx: number) => {
    try {
      await api.restorePattern(idx)
      await get().fetchStatus()
      await get().fetchDeleted()
    } catch (e) {
      console.error('Restore failed:', e)
    }
  },

  fetchDeleted: async () => {
    try {
      const deleted = await api.getDeletedPatterns()
      set({ deletedPatterns: deleted })
    } catch (e) {
      console.error('Fetch deleted failed:', e)
    }
  },

  navigateToPattern: async (idx: number) => {
    set({ searching: true })
    try {
      const pattern = await api.getPattern(idx)
      // Replace results with the single navigated-to pattern
      const result: SearchResult = {
        rank: 1,
        idx: pattern.idx,
        score: 0,
        cosine_score: null,
        physics_score: null,
        hamming_score: null,
        keyword_boost: null,
        title: pattern.title,
        preview: pattern.preview,
        full_text: pattern.full_text,
        from_lattice: false,
        prev_idx: pattern.prev_idx,
        next_idx: pattern.next_idx,
      }
      set({
        results: [result],
        searchModeLabel: `Navigate → #${idx}`,
        searchElapsed: 0,
        query: '',
      })
    } catch (e) {
      console.error('Navigate failed:', e)
    } finally {
      set({ searching: false })
    }
  },
}))
