import { create } from 'zustand'
import * as cb from '../api/cartbuilder'
import type {
  CartBuilderFile, CartBuilderBuildState, CartBuilderListedCart,
  CartBuilderSubdir, CartBuilderDoc, CartBuilderPattern0,
} from '../api/cartbuilder'

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

  openFolderPicker: (path?: string) => Promise<void>
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

  openFolderPicker: async (path?: string) => {
    try {
      const resp = await cb.browseFolders(path ?? '')
      set({
        pickerPath: resp.path,
        pickerDirs: resp.dirs,
        pickerParent: resp.parent ?? null,
        pickerOpen: true,
      })
    } catch (e) {
      console.error('browseFolders failed', e)
    }
  },

  closeFolderPicker: () => set({ pickerOpen: false }),

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
