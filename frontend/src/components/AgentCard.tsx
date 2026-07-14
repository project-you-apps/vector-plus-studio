import {
  Newspaper, MessageCircleQuestion, GraduationCap, ClipboardList,
  Bot, Sparkles, Send,
} from 'lucide-react'
import type { AgentDefinition } from '../agents/agent-definitions'

// Central icon map for agent cards. New agents need one entry here; unknown
// names fall back to Bot so the grid keeps rendering rather than crashing
// on a typo in agent-definitions.ts.
const ICON_MAP: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Newspaper,
  MessageCircleQuestion,
  GraduationCap,
  ClipboardList,
  Bot,
}

// One tile in the Agents grid. Same visual language as ReportCard so both
// tabs read as the same product family — slate-800/40 fill, slate-700
// border, purple hover accent (agents are LLM-first; reports use emerald).
// LLM-dependent agents get the same sparkles badge Reports uses.
export default function AgentCard({
  agent,
  onRun,
}: {
  agent: AgentDefinition
  onRun: () => void
}) {
  const Icon = ICON_MAP[agent.icon] ?? Bot
  return (
    <button
      onClick={onRun}
      className="group text-left rounded-lg border border-slate-700 bg-slate-800/40 p-4
                 transition-all hover:border-purple-500/60 hover:bg-slate-800/60
                 hover:shadow-[0_0_0_1px_rgba(168,85,247,0.25)] cursor-pointer
                 flex flex-col gap-3 min-h-[168px]"
      title={`Send to ${agent.displayName}`}
    >
      <div className="flex items-start gap-3">
        <div className="rounded-lg bg-purple-500/10 border border-purple-500/20 p-2 shrink-0
                        group-hover:bg-purple-500/20 transition-colors">
          <Icon size={20} className="text-purple-300" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-slate-100 truncate">
              {agent.displayName}
            </h3>
            {agent.llmDependency && (
              <span
                className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded
                           bg-purple-500/15 border border-purple-500/40 text-purple-200
                           font-mono flex items-center gap-1 shrink-0"
                title="This agent uses an LLM to synthesize its output."
              >
                <Sparkles size={9} />
                LLM
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">
            {agent.description}
          </p>
        </div>
      </div>

      <div className="mt-auto flex items-center justify-between pt-2 border-t border-slate-700/60">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono">
          {agent.inputSchema.length} input{agent.inputSchema.length === 1 ? '' : 's'}
        </span>
        <span
          className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium
                     bg-purple-500/15 border border-purple-500/40 text-purple-200
                     group-hover:bg-purple-500/30 group-hover:text-purple-100
                     transition-colors"
        >
          <Send size={11} />
          Send
        </span>
      </div>
    </button>
  )
}
