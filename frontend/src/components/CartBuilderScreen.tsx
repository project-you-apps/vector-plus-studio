import { useState } from 'react'
import { Hammer, Upload, ExternalLink, FileText, Settings as SettingsIcon, Info } from 'lucide-react'

// Cart Builder — currently the layout skeleton + a fallback that launches
// the existing standalone Flask Cart Builder at http://localhost:5000.
// Full React port is phased; see docs/CARTBUILDER-PORT-PLAN.md.

const STANDALONE_URL = 'http://localhost:5000'

export default function CartBuilderScreen() {
  const [dragOver, setDragOver] = useState(false)

  return (
    <main className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold gradient-text mb-1 flex items-center gap-2">
              <Hammer size={28} className="text-purple-300" />
              Cart Builder
            </h1>
            <p className="text-sm text-slate-500">
              Drag-and-drop documents to build a Membot brain cartridge.
            </p>
          </div>

          {/* Interim: launch the existing standalone Flask app */}
          <a
            href={STANDALONE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 transition-colors"
            title="Open the standalone Flask Cart Builder in a new tab (interim — full React port in progress)"
          >
            <ExternalLink size={12} />
            Open standalone Cart Builder
          </a>
        </div>

        {/* Status banner — interim notice */}
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 flex items-start gap-3">
          <Info size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <div className="text-amber-200 font-medium mb-0.5">Cart Builder port in progress</div>
            <div className="text-xs text-slate-400 leading-relaxed">
              The full Cart Builder is a working Flask app at{' '}
              <code className="text-slate-300">localhost:5000</code> (run{' '}
              <code className="text-slate-300">python app.py</code> from{' '}
              <code className="text-slate-300">cart-builder/cart-builder/</code>).
              The React port lives here; phasing is in{' '}
              <code className="text-slate-300">docs/CARTBUILDER-PORT-PLAN.md</code>.
              Skeleton below is non-functional pending the backend route port.
            </div>
          </div>
        </div>

        {/* Drop zone (skeleton) */}
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            // TODO: wire to /api/cartbuilder/upload once backend route lands
            // For now: hint to use the standalone app
            alert('Drag-and-drop will land in Phase 2 of the port. Use the standalone Cart Builder for now (Open standalone Cart Builder button in the header).')
          }}
          className={`rounded-xl border-2 border-dashed p-12 text-center transition-colors ${
            dragOver
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-slate-700 bg-slate-800/20 hover:border-slate-600'
          }`}
        >
          <Upload size={36} className={`mx-auto mb-3 ${dragOver ? 'text-purple-400' : 'text-slate-600'}`} />
          <div className="text-sm font-medium text-slate-300 mb-1">
            Drop documents here
          </div>
          <div className="text-xs text-slate-500">
            PDF, DOCX, XLSX, TXT, MD, RTF — or click to browse
          </div>
          <div className="mt-4 text-[10px] text-slate-600 italic">
            (Skeleton — wires up in Phase 2)
          </div>
        </div>

        {/* Future-feature placeholders, laid out as the real screen will look */}
        <div className="grid grid-cols-3 gap-3">
          <PlaceholderCard
            icon={FileText}
            title="File Cards"
            body="Per-file preview with metadata editor (owner / description / tags). Replace, soft-remove + undo, hard delete."
          />
          <PlaceholderCard
            icon={SettingsIcon}
            title="Pattern 0 Preview"
            body="Manifest TOC: cart name, creator, file list with chunk counts, embedding model, tags."
          />
          <PlaceholderCard
            icon={Hammer}
            title="Build / Update"
            body="Cart name, build button, live progress bar, model-load message, interrupt detection."
          />
        </div>

        {/* Footnote */}
        <p className="text-center text-xs text-slate-600 italic pt-2">
          Source: <code className="font-mono">cart-builder/cart-builder/</code> (commit{' '}
          <code className="font-mono">c2fb03c</code>, v1.1, 2026-04-03). Port plan:{' '}
          <code className="font-mono">docs/CARTBUILDER-PORT-PLAN.md</code>.
        </p>
      </div>
    </main>
  )
}

function PlaceholderCard({ icon: Icon, title, body }: {
  icon: React.ComponentType<{ size?: number; className?: string }>
  title: string
  body: string
}) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/20 p-4">
      <div className="flex items-center gap-2 mb-2">
        <Icon size={14} className="text-slate-500" />
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
      </div>
      <p className="text-xs text-slate-500 leading-relaxed">{body}</p>
      <div className="mt-2 text-[10px] text-slate-600 italic">Coming soon</div>
    </div>
  )
}
