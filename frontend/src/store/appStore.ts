import { create } from 'zustand'
import type {
  CartridgeInfo, SearchResult, StatusResponse, DeletedPattern, SearchMode,
  MemboxCartInfo, MemboxStatus,
} from '../api/types'
import * as api from '../api/client'

// Top-level screen state (nav rail picks which screen renders in the main area).
// 'search' is the default; the original VPS 1.0 search/CRUD experience. Other
// screens are stubbed in this iteration and fleshed out incrementally.
export type ActiveScreen = 'search' | 'overview' | 'cartBuilder' | 'sql' | 'settings'

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

  // Strict keyword filter
  strictMode: boolean
  setStrictMode: (strict: boolean) => void

  // Exact phrase match filter
  exactMatch: boolean
  setExactMatch: (exact: boolean) => void

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

  deletedPatterns: [],

  addingPassage: false,

  strictMode: false,
  setStrictMode: (strict) => set({ strictMode: strict }),

  exactMatch: false,
  setExactMatch: (exact) => set({ exactMatch: exact }),

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
    const { editorText, editorOriginalIdx, addPassage, deleteResult, closeEditor } = get()
    const text = editorText.trim()
    if (!text) return { success: false, message: 'Text is empty' }

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
  openModal: (result) => set({
    modalOpen: true,
    modalPassage: {
      idx: result.idx,
      title: result.title,
      full_text: result.full_text,
      prev_idx: result.prev_idx,
      next_idx: result.next_idx,
      // Carry split-cart hints from the search result so the modal can render
      // the "Load full passage from <db>" CTA. paper_id is only set after the
      // user clicks load-source (via loadSourceForCurrentPassage below).
      source_db: result.source_db ?? null,
      paper_id: result.paper_id ?? null,
    },
  }),
  closeModal: () => set({ modalOpen: false, modalPassage: null }),
  navigateModal: async (idx: number) => {
    set({ modalLoading: true })
    try {
      const pattern = await api.getPattern(idx)
      set({
        modalPassage: {
          idx: pattern.idx,
          title: pattern.title,
          full_text: pattern.full_text,
          prev_idx: pattern.prev_idx,
          next_idx: pattern.next_idx,
          source_db: pattern.source_db ?? null,
          paper_id: pattern.paper_id ?? null,
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
    set({ mounting: true })
    try {
      await api.mountCartridge(filename)
      await get().fetchStatus()
      await get().fetchCartridges()
      set({ results: [], query: '', deletedPatterns: [], strictMode: false, exactMatch: false })
    } catch (e) {
      console.error('Mount failed:', e)
    } finally {
      set({ mounting: false })
    }
  },

  unmount: async () => {
    try {
      await api.unmountCartridge()
      await get().fetchStatus()
      set({ results: [], query: '', deletedPatterns: [] })
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
    const { searchMode, blendAlpha, topK } = get()
    set({ searching: true, query })
    try {
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
