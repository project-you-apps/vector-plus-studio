import { create } from 'zustand'
import type { CartridgeInfo, SearchResult, StatusResponse, DeletedPattern, SearchMode } from '../api/types'
import * as api from '../api/client'

interface AppState {
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

  // Actions
  fetchStatus: () => Promise<void>
  fetchCartridges: () => Promise<void>
  mount: (filename: string) => Promise<void>
  unmount: () => Promise<void>
  saveCartridge: () => Promise<{ success: boolean; message: string }>
  setSearchMode: (mode: SearchMode) => void
  setBlendAlpha: (alpha: number) => void
  setTopK: (k: number) => void
  doSearch: (query: string) => Promise<void>
  addPassage: (text: string) => Promise<{ success: boolean; message: string }>
  deleteResult: (idx: number) => Promise<void>
  restoreResult: (idx: number) => Promise<void>
  fetchDeleted: () => Promise<void>
}

export const useAppStore = create<AppState>((set, get) => ({
  status: null,
  statusLoading: false,
  cartridges: [],
  mounting: false,

  searchMode: 'smart',
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

  setSearchMode: (mode) => {
    set({ searchMode: mode })
    // Auto-re-search with the new mode if there's already a query
    const { query } = get()
    if (query) {
      // Small delay so the mode state updates before doSearch reads it
      setTimeout(() => get().doSearch(query), 0)
    }
  },
  setBlendAlpha: (alpha) => set({ blendAlpha: alpha }),
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
}))
