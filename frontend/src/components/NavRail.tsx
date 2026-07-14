import { useState } from 'react'
import { Search, LayoutDashboard, Hammer, Pencil, FileBarChart, Bot, Terminal, Settings, HelpCircle, Info } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import type { ActiveScreen } from '../store/appStore'
import AboutModal from './AboutModal'
import HelpModal from './HelpModal'

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
  { key: 'crud',        label: 'Edit Carts',   icon: Pencil,          tooltip: 'Add / update / delete passages on the mounted cart' },
  { key: 'reports',     label: 'Reports',      icon: FileBarChart,    tooltip: 'Structured views over the mounted cart (Summary, Timeline, Trend, …)' },
  { key: 'agents',      label: 'Agents',       icon: Bot,             tooltip: 'Scoped agent recipes (Auto-Briefing, Q&A, Professor, Curator) that run against the cart' },
  { key: 'sql',         label: 'SQL',          icon: Terminal,        tooltip: 'SQL-like query editor (planned)' },
  { key: 'settings',    label: 'Settings',     icon: Settings,        tooltip: 'Search modes, theme, advanced options' },
]

export default function NavRail() {
  const { activeScreen, setActiveScreen } = useAppStore()
  const [aboutOpen, setAboutOpen] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)

  return (
    <nav
      className="w-48 flex flex-col py-3 px-2 gap-1 border-r border-slate-800 bg-[var(--chrome-bg)] flex-shrink-0"
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
            title={item.tooltip}
            className={`group flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left ${
              active
                ? 'bg-purple-500/20 text-purple-300 font-medium'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <Icon size={16} className="flex-shrink-0" />
            <span>{item.label}</span>
          </button>
        )
      })}

      <div className="mt-auto pt-2 border-t border-slate-800/60 flex flex-col gap-1">
        <button
          onClick={() => setHelpOpen(true)}
          aria-label="Help"
          title="Getting Started + Docs"
          className="group flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
        >
          <HelpCircle size={16} className="flex-shrink-0" />
          <span>Help</span>
        </button>
        <button
          onClick={() => setAboutOpen(true)}
          aria-label="About"
          title="About Vector+ Studio"
          className="group flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
        >
          <Info size={16} className="flex-shrink-0" />
          <span>About</span>
        </button>
      </div>

      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
      <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </nav>
  )
}
