import { Search, LayoutDashboard, Hammer, Terminal, Settings } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { ActiveScreen } from '../store/appStore'

interface NavItem {
  key: ActiveScreen
  label: string
  icon: React.ComponentType<{ size?: number; className?: string }>
  tooltip: string
}

// Screens are listed in nav order. Search is the default landing screen.
// Membox is intentionally NOT in the rail — it's a slide-out *tool* (invoked
// from the Header Activity icon), conceptually different from a screen.
const NAV_ITEMS: NavItem[] = [
  { key: 'search',      label: 'Search',       icon: Search,          tooltip: 'Search and CRUD on the mounted cartridge (default)' },
  { key: 'overview',    label: 'Overview',     icon: LayoutDashboard, tooltip: 'Cart stats, mounted carts, system health' },
  { key: 'cartBuilder', label: 'Cart Builder', icon: Hammer,          tooltip: 'Drag-and-drop cart creation from documents' },
  { key: 'sql',         label: 'SQL',          icon: Terminal,        tooltip: 'SQL-like query editor (planned)' },
  { key: 'settings',    label: 'Settings',     icon: Settings,        tooltip: 'Search modes, theme, advanced options' },
]

export default function NavRail() {
  const { activeScreen, setActiveScreen } = useAppStore()

  return (
    <nav
      className="w-14 flex flex-col items-center py-3 gap-1 border-r border-slate-800 bg-[var(--chrome-bg)] flex-shrink-0"
      aria-label="Primary navigation"
    >
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon
        const active = activeScreen === item.key
        return (
          <button
            key={item.key}
            onClick={() => setActiveScreen(item.key)}
            aria-label={item.label}
            aria-current={active ? 'page' : undefined}
            className={`group relative w-10 h-10 rounded-lg flex items-center justify-center transition-colors ${
              active
                ? 'bg-purple-500/20 text-purple-300'
                : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <Icon size={18} />
            {/* Discord-style flyout tooltip on hover. Faster + nicer than the
                native title attribute, and gives us room for the longer
                description on a second line. */}
            <span
              role="tooltip"
              className="pointer-events-none absolute left-full top-1/2 -translate-y-1/2 ml-2 px-2.5 py-1.5 rounded-md bg-slate-900 border border-slate-700 text-slate-100 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-lg"
            >
              <span className="block text-xs font-semibold">{item.label}</span>
              <span className="block text-[10px] text-slate-400 mt-0.5 max-w-[14rem] whitespace-normal">{item.tooltip}</span>
            </span>
          </button>
        )
      })}
    </nav>
  )
}
