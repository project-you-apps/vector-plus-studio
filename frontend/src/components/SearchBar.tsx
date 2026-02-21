import { useState } from 'react'
import { Search, Loader2, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'

export default function SearchBar() {
  const { doSearch, searching, status, query, topK, setTopK } = useAppStore()
  const [input, setInput] = useState('')

  const disabled = !status?.mounted_cartridge || searching
  const hasResults = !!query

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim().length > 2 && !disabled) {
      doSearch(input.trim())
    }
  }

  const handleClear = () => {
    setInput('')
    useAppStore.setState({ query: '', results: [], searchModeLabel: '', searchElapsed: 0 })
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <div className="flex-1 relative">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') e.preventDefault() }}
          placeholder={status?.mounted_cartridge ? 'What are you looking for?' : 'Mount a cartridge first...'}
          disabled={!status?.mounted_cartridge}
          className="w-full pl-11 pr-10 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl
                     text-slate-200 placeholder-slate-600 focus:outline-none focus:border-purple-500/50
                     focus:ring-1 focus:ring-purple-500/20 transition-all disabled:opacity-50"
        />
        {/* Clear button -- visible when there are results */}
        {hasResults && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 px-2 py-0.5 rounded text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-700/50 transition-colors"
            title="Clear search results"
          >
            <X size={12} />
            Clear
          </button>
        )}
      </div>

      {/* Top-K selector */}
      <select
        value={topK}
        onChange={(e) => setTopK(Number(e.target.value))}
        className="px-2 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl text-slate-300 text-sm
                   focus:outline-none focus:border-purple-500/50 cursor-pointer"
        title="Number of results to return"
      >
        <option value={5}>Top 5</option>
        <option value={10}>Top 10</option>
        <option value={20}>Top 20</option>
        <option value={50}>Top 50</option>
      </select>

      <button
        type="submit"
        disabled={disabled || input.trim().length < 3}
        className="px-6 py-3 rounded-xl gradient-bg text-white font-medium
                   hover:opacity-90 transition-opacity disabled:opacity-40
                   flex items-center gap-2"
      >
        {searching ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            Searching
          </>
        ) : (
          <>
            <Search size={16} />
            Search
          </>
        )}
      </button>
    </form>
  )
}
