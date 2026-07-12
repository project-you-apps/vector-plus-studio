import {
  FileText, Calendar, TrendingUp, Scale, Tag, DollarSign,
  GitCompareArrows, ShieldAlert, Sparkles, Play,
} from 'lucide-react'
import type { ReportDefinition } from '../reports/report-definitions'

// Central icon map. New reports need one entry here; unknown names fall back
// to FileText so the grid keeps rendering rather than crashing on a typo in
// report-definitions.ts.
const ICON_MAP: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  FileText,
  Calendar,
  TrendingUp,
  Scale,
  Tag,
  DollarSign,
  GitCompareArrows,
  ShieldAlert,
  Sparkles,
}

// One tile in the Reports grid. Visual style matches OverviewScreen's
// StatCard + CRUDScreen's OpPanel — slate-800/40 fill, slate-700 border,
// emerald hover accent. LLM-dependent reports get a subtle sparkles badge
// (only Executive TL;DR in this pass).
export default function ReportCard({
  report,
  onRun,
}: {
  report: ReportDefinition
  onRun: () => void
}) {
  const Icon = ICON_MAP[report.icon] ?? FileText
  return (
    <button
      onClick={onRun}
      className="group text-left rounded-lg border border-slate-700 bg-slate-800/40 p-4
                 transition-all hover:border-emerald-500/60 hover:bg-slate-800/60
                 hover:shadow-[0_0_0_1px_rgba(16,185,129,0.25)] cursor-pointer
                 flex flex-col gap-3 min-h-[168px]"
      title={`Run ${report.displayName}`}
    >
      <div className="flex items-start gap-3">
        <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-2 shrink-0
                        group-hover:bg-emerald-500/20 transition-colors">
          <Icon size={20} className="text-emerald-300" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-slate-100 truncate">
              {report.displayName}
            </h3>
            {report.llmDependency && (
              <span
                className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded
                           bg-purple-500/15 border border-purple-500/40 text-purple-200
                           font-mono flex items-center gap-1 shrink-0"
                title="This report requires an LLM (Claude) to synthesize its output."
              >
                <Sparkles size={9} />
                LLM
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">
            {report.description}
          </p>
        </div>
      </div>

      <div className="mt-auto flex items-center justify-between pt-2 border-t border-slate-700/60">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono">
          {report.inputSchema.length} input{report.inputSchema.length === 1 ? '' : 's'}
        </span>
        <span
          className="flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium
                     bg-emerald-500/15 border border-emerald-500/40 text-emerald-200
                     group-hover:bg-emerald-500/30 group-hover:text-emerald-100
                     transition-colors"
        >
          <Play size={11} />
          Run
        </span>
      </div>
    </button>
  )
}
