import { create } from 'zustand'
import * as cb from '../api/cartbuilder'
import type {
  CartBuilderFile, CartBuilderBuildState, CartBuilderListedCart,
  CartBuilderSubdir, CartBuilderDoc, CartBuilderPattern0,
} from '../api/cartbuilder'
import type { BuiltCart } from '../cart-builder-v2'

// Last-build success surfaces (2026-07-09). Both browser-pipeline and
// delegated-desktop builds produce a "cart saved" card at the bottom of
// the Cart Builder screen. Historically these lived in BrowserCartBuilder
// local state, which meant switching tabs dropped the card — user lost
// track of where their cart landed. Migrated to store so the card
// survives navigation. Cleared only by: (a) user clicks the X button
// (dismissLastBuild), (b) user starts a new build, (c) user completes
// the save-to-disk flow which calls resetToCleanSlate.
export interface LastDesktopBuild {
  cartPath: string
  cartSizeMb: number | null
  chunksTotal: number | null
}

// Cart Builder screen state, separate from the main appStore so the workflow
// (drop → workspace → build) doesn't pollute the search-side store. The two
// stores both hit the same backend via fetch — they don't talk to each other
// directly. The shared CartBrowser component is rendered via this store.

// Toasts — lightweight cross-screen notification system. Cart Builder owns
// it for now (build events + upload errors); other screens can borrow.
export interface Toast {
  id: string
  kind: 'success' | 'error' | 'info'
  text: string
  ttlMs?: number  // null = sticky
}

interface CartBuilderState {
  // Toasts
  toasts: Toast[]
  pushToast: (kind: Toast['kind'], text: string, ttlMs?: number) => void
  dismissToast: (id: string) => void

  // Workspace (uploaded / ingested files awaiting build)
  files: CartBuilderFile[]
  uploading: boolean
  uploadError: string | null

  // Build state (polled from /build/status during a build)
  build: CartBuilderBuildState
  buildPolling: boolean

  // Last build result — survives tab navigation so the "cart saved to X"
  // card stays visible if the user hops to Search and back. See
  // LastDesktopBuild comment above the interface.
  lastBrowserBuild: BuiltCart | null
  lastDesktopBuild: LastDesktopBuild | null
  setLastBrowserBuild: (v: BuiltCart | null) => void
  setLastDesktopBuild: (v: LastDesktopBuild | null) => void
  dismissLastBuild: () => void

  // Cart-name input
  cartName: string
  setCartName: (name: string) => void

  // Pattern 0 preview (manifest TOC)
  pattern0: CartBuilderPattern0 | null

  // Cart browser state (shared with the CartBrowser component used in
  // both Cart Builder and Edit Carts screens)
  browserCarts: CartBuilderListedCart[]
  browserSubdirs: CartBuilderSubdir[]
  browserDocs: CartBuilderDoc[]
  browserFolders: string[]
  browserCurrentPath: string

  // Folder picker (server-side path browser modal)
  pickerPath: string
  pickerDirs: string[]
  pickerParent: string | null
  pickerOpen: boolean
  // Optional completion handler set by the caller of openFolderPicker.
  // When set, "Use this folder" calls this instead of the default
  // addBrowserFolder. Null = legacy behavior (add to saved cart folders).
  // Andy 2026-05-10 — needed so the New Cart destination flow can reuse
  // the same picker without permanently adding the destination to saved
  // folders.
  pickerOnConfirm: ((path: string) => void) | null

  // ── Actions ──
  uploadFiles: (files: File[]) => Promise<void>
  refreshFiles: () => Promise<void>
  setMetadata: (file_id: string, patch: { owner?: string; description?: string; tags?: string[] }) => Promise<void>
  ingestPath: (path: string) => Promise<void>

  refreshPattern0: () => Promise<void>
  startBuild: () => Promise<void>
  refreshBuildStatus: () => Promise<void>
  startBuildPolling: () => void
  stopBuildPolling: () => void

  refreshBrowser: (path?: string) => Promise<void>
  addBrowserFolder: (folder: string) => Promise<void>
  removeBrowserFolder: (folder: string) => Promise<void>
  loadCart: (cart_path: string) => Promise<void>
  clearWorkspace: () => Promise<void>

  openFolderPicker: (options?: { path?: string; onConfirm?: (path: string) => void }) => Promise<void>
  closeFolderPicker: () => void
  navigateFolderPicker: (path: string) => Promise<void>
}

const initialBuild: CartBuilderBuildState = {
  status: 'idle',
  progress: 0,
  chunks_done: 0,
  chunks_total: 0,
}

let buildPollHandle: ReturnType<typeof setInterval> | null = null

export const useCartBuilderStore = create<CartBuilderState>((set, get) => ({
  toasts: [],
  pushToast: (kind, text, ttlMs = 4500) => {
    const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
    set((s) => ({ toasts: [...s.toasts, { id, kind, text, ttlMs }] }))
    if (ttlMs && ttlMs > 0) {
      setTimeout(() => get().dismissToast(id), ttlMs)
    }
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  files: [],
  uploading: false,
  uploadError: null,

  build: initialBuild,
  buildPolling: false,

  lastBrowserBuild: null,
  lastDesktopBuild: null,
  setLastBrowserBuild: (v) => set({ lastBrowserBuild: v }),
  setLastDesktopBuild: (v) => set({ lastDesktopBuild: v }),
  dismissLastBuild: () => set({ lastBrowserBuild: null, lastDesktopBuild: null }),

  cartName: 'my-cart',
  setCartName: (name) => set({ cartName: name }),

  pattern0: null,

  browserCarts: [],
  browserSubdirs: [],
  browserDocs: [],
  browserFolders: [],
  browserCurrentPath: '',

  pickerPath: '',
  pickerDirs: [],
  pickerParent: null,
  pickerOpen: false,
  pickerOnConfirm: null,

  uploadFiles: async (files) => {
    if (files.length === 0) return
    set({ uploading: true, uploadError: null })
    try {
      const resp = await cb.uploadFiles(files)
      set((s) => ({ files: [...s.files, ...resp.files], uploading: false }))
      get().refreshPattern0()
      get().pushToast('success', `Uploaded ${resp.files.length} file${resp.files.length === 1 ? '' : 's'}`)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Upload failed'
      set({ uploading: false, uploadError: msg })
      get().pushToast('error', `Upload failed: ${msg}`, 8000)
    }
  },

  refreshFiles: async () => {
    try {
      const resp = await cb.listFiles()
      set({ files: resp.files })
    } catch (e) {
      console.error('listFiles failed', e)
    }
  },

  setMetadata: async (file_id, patch) => {
    try {
      await cb.setMetadata({ file_id, ...patch })
      // Optimistic local update
      set((s) => ({
        files: s.files.map((f) => f.id === file_id ? { ...f, ...patch } : f),
      }))
    } catch (e) {
      console.error('setMetadata failed', e)
    }
  },

  ingestPath: async (path) => {
    set({ uploading: true, uploadError: null })
    try {
      const resp = await cb.ingestPath(path)
      set((s) => ({ files: [...s.files, resp.file], uploading: false }))
      get().refreshPattern0()
    } catch (e) {
      set({ uploading: false, uploadError: e instanceof Error ? e.message : 'Ingest failed' })
    }
  },

  refreshPattern0: async () => {
    try {
      const p0 = await cb.getPattern0(get().cartName || 'my-cart')
      set({ pattern0: p0 })
    } catch (e) {
      // Backend may be 503 (cart-builder modules unavailable on droplet)
      console.error('pattern0 failed', e)
    }
  },

  startBuild: async () => {
    const name = get().cartName.trim() || 'my-cart'
    try {
      const resp = await cb.buildCart(name)
      get().pushToast('info', `Build started: ${resp.cart_name} (${resp.chunks} chunks)`, 3000)
      get().startBuildPolling()
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Build failed'
      set({ uploadError: msg })
      get().pushToast('error', `Build failed to start: ${msg}`, 8000)
    }
  },

  refreshBuildStatus: async () => {
    try {
      const s = await cb.getBuildStatus()
      const prev = get().build.status
      set({ build: s })
      if (s.status === 'done' && prev !== 'done') {
        get().stopBuildPolling()
        get().refreshBrowser()  // new cart now visible
        get().pushToast('success', `Cart built: ${s.cart_path?.split(/[/\\]/).pop() ?? 'done'}`, 6000)
      } else if (s.status === 'error' && prev !== 'error') {
        get().stopBuildPolling()
        get().pushToast('error', `Build failed: ${s.error || 'unknown error'}`, 10000)
      }
    } catch (e) {
      console.error('build status failed', e)
    }
  },

  startBuildPolling: () => {
    if (buildPollHandle) return
    set({ buildPolling: true })
    buildPollHandle = setInterval(() => get().refreshBuildStatus(), 1000)
    get().refreshBuildStatus()
  },

  stopBuildPolling: () => {
    if (buildPollHandle) {
      clearInterval(buildPollHandle)
      buildPollHandle = null
    }
    set({ buildPolling: false })
  },

  refreshBrowser: async (path?: string) => {
    try {
      const resp = await cb.listCarts(path ?? get().browserCurrentPath)
      set({
        browserCarts: resp.carts,
        browserSubdirs: resp.subdirs ?? [],
        browserDocs: resp.docs ?? [],
        browserFolders: resp.folders ?? [],
        browserCurrentPath: resp.current_path ?? '',
      })
    } catch (e) {
      console.error('listCarts failed', e)
    }
  },

  addBrowserFolder: async (folder) => {
    try {
      const resp = await cb.addCartFolder(folder)
      set({ browserFolders: resp.folders })
      get().refreshBrowser()
    } catch (e) {
      console.error('addCartFolder failed', e)
    }
  },

  removeBrowserFolder: async (folder) => {
    try {
      const resp = await cb.removeCartFolder(folder)
      set({ browserFolders: resp.folders })
      get().refreshBrowser()
    } catch (e) {
      console.error('removeCartFolder failed', e)
    }
  },

  loadCart: async (cart_path) => {
    try {
      const resp = await cb.loadCart(cart_path)
      set({
        files: resp.files,
        cartName: resp.cart_name,
      })
      get().refreshPattern0()
      get().pushToast('success', `Loaded ${resp.cart_name} (${resp.total_passages} passages, ${resp.total_sources} sources)`, 4000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'loadCart failed'
      console.error('loadCart failed', e)
      get().pushToast('error', `Open cart failed: ${msg}`, 8000)
    }
  },

  clearWorkspace: async () => {
    try {
      await cb.clearWorkspace()
      set({ files: [], pattern0: null, build: initialBuild })
    } catch (e) {
      console.error('clearWorkspace failed', e)
    }
  },

  openFolderPicker: async (options) => {
    const path = options?.path
    const onConfirm = options?.onConfirm ?? null
    try {
      const resp = await cb.browseFolders(path ?? '')
      set({
        pickerPath: resp.path,
        pickerDirs: resp.dirs,
        pickerParent: resp.parent ?? null,
        pickerOpen: true,
        pickerOnConfirm: onConfirm,
      })
    } catch (e) {
      console.error('browseFolders failed', e)
    }
  },

  closeFolderPicker: () => set({ pickerOpen: false, pickerOnConfirm: null }),

  navigateFolderPicker: async (path) => {
    try {
      const resp = await cb.browseFolders(path)
      set({
        pickerPath: resp.path,
        pickerDirs: resp.dirs,
        pickerParent: resp.parent ?? null,
      })
    } catch (e) {
      console.error('browseFolders failed', e)
    }
  },
}))
