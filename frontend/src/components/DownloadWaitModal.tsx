import { useEffect } from 'react'
import { Download, X, FileText, FolderOpen, Package, Terminal, Play, RefreshCw } from 'lucide-react'

// Companion modal to the "Download builders" button on the Cart Builder
// screen. The vps-suite.zip is ~1.5 GB — 5-20 min on most home connections —
// so users are prone to wander off or forget where they were. Popping this
// modal on click gives them a numbered pre-flight checklist to run against
// while the browser downloads the zip in the background. The download itself
// is untouched: the anchor's default `download` behavior fires normally; we
// just piggyback the modal on top of the click.
//
// Structure mirrors the ImageBuilderMissingDialog / AboutModal shell (same
// backdrop-blur, same slate-900 body, same X-in-top-right + Escape/backdrop
// close) so it reads as part of the existing modal family. Purple accent
// aligns with the pill's purple "Download builders" button.

interface Props {
  open: boolean
  onClose: () => void
}

// Numbered checklist step. Each step gets a lucide icon + a numbered circle
// so the modal reads as an ordered pre-flight list rather than a wall of
// text. Splitting Step out as a small helper keeps the render loop below
// clean without pulling a full sub-component into its own file.
interface Step {
  n: number
  icon: React.ReactNode
  title: string
  body: React.ReactNode
}

export default function DownloadWaitModal({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  const steps: Step[] = [
    {
      n: 1,
      icon: <FileText size={14} className="text-purple-300" />,
      title: 'Verify Python 3.11+ is installed',
      body: (
        <>
          Open a terminal and run <Code>python --version</Code>. If Python
          isn&rsquo;t installed (or shows a version older than 3.11), grab it
          from{' '}
          <a
            href="https://www.python.org/downloads/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-purple-300 hover:text-purple-200 underline underline-offset-2"
          >
            python.org
          </a>
          .
        </>
      ),
    },
    {
      n: 2,
      icon: <FolderOpen size={14} className="text-purple-300" />,
      title: 'Pick an extraction location',
      body: (
        <>
          Choose a fresh directory <em>outside</em> cloud-sync folders
          (Dropbox, OneDrive, iCloud, Google Drive) — sync tools will fight
          the extraction and lock files mid-write. Something like{' '}
          <Code>C:\vps-suite</Code> or <Code>~/vps-suite</Code> works well.
        </>
      ),
    },
    {
      n: 3,
      icon: <Package size={14} className="text-purple-300" />,
      title: 'When the download finishes: extract the zip',
      body: (
        <>
          Right-click the downloaded <Code>vps-suite.zip</Code> and choose{' '}
          <em>Extract All</em> (Windows built-in works fine). The suite is
          about 5 GB extracted, so give it a minute.
        </>
      ),
    },
    {
      n: 4,
      icon: <Terminal size={14} className="text-purple-300" />,
      title: 'Install Python dependencies',
      body: (
        <>
          From a terminal (admin or regular is fine):
          <div className="mt-2 rounded-md border border-slate-800 bg-slate-950/60 px-3 py-2 font-mono text-[11px] text-slate-300 space-y-1">
            <div>cd vps-suite\desktop-builder</div>
            <div>pip install -r requirements.txt</div>
            <div>cd ..\image-builder</div>
            <div>pip install -r requirements.txt</div>
          </div>
          <div className="mt-1.5 text-[10px] text-slate-500">
            A single shared venv is fine; separate venvs also work.
          </div>
        </>
      ),
    },
    {
      n: 5,
      icon: <Play size={14} className="text-purple-300" />,
      title: 'Launch the builders',
      body: (
        <>
          Double-click <Code>start-cart-builder.bat</Code> and{' '}
          <Code>start-image-builder.bat</Code>. Each opens a terminal window
          that must stay open while you use the app.
        </>
      ),
    },
    {
      n: 6,
      icon: <RefreshCw size={14} className="text-purple-300" />,
      title: 'Return to this browser tab',
      body: (
        <>
          Click the <Code>Recheck</Code> button on both status pills above.
          They should flip from slate-gray to purple once each builder
          answers on its loopback port.
        </>
      ),
    },
  ]

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Preparing your machine while the download runs"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="relative w-full max-w-2xl max-h-[90vh] flex flex-col rounded-2xl border border-purple-500/40 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-start justify-between gap-3 shrink-0">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-md flex items-center justify-center bg-purple-500/20 border border-purple-500/40 shrink-0 mt-0.5">
              <Download size={14} className="text-purple-300" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-100">
                Preparing your machine while the download runs
              </h2>
              <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">
                The ~1.5 GB zip takes 5&ndash;20 minutes on most home
                connections. Here&rsquo;s what to line up in the meantime.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300 shrink-0"
            aria-label="Close"
            title="Close (Esc)"
          >
            <X size={14} />
          </button>
        </div>

        {/* Scrollable body — the checklist is longer than the modal on
            shorter viewports; letting the header + footer stay pinned keeps
            the Close controls always in reach. */}
        <div className="px-5 py-4 overflow-y-auto flex-1 space-y-3">
          {steps.map((step) => (
            <div
              key={step.n}
              className="rounded-lg border border-slate-800 bg-slate-800/30 p-3.5 flex gap-3"
            >
              <div className="w-7 h-7 rounded-full flex items-center justify-center bg-purple-500/15 border border-purple-500/40 text-purple-200 text-xs font-semibold shrink-0">
                {step.n}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {step.icon}
                  <h3 className="text-sm font-semibold text-slate-100">
                    {step.title}
                  </h3>
                </div>
                <div className="text-xs text-slate-400 leading-relaxed">
                  {step.body}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="px-5 py-3 border-t border-slate-700/40 flex items-center justify-between gap-3 shrink-0">
          <p className="text-[11px] text-slate-500 leading-snug">
            Full instructions ship inside the zip as{' '}
            <Code>README.md</Code> in the top-level directory.
          </p>
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-md bg-purple-500/20 border border-purple-500/40 text-purple-100 hover:bg-purple-500/30 text-xs font-medium transition-colors shrink-0"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  )
}

// Inline code chip. Kept local — the modal is the only surface using this
// exact treatment right now, and inlining avoids adding another shared
// primitive for a five-line component.
function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/60 font-mono text-[11px] text-slate-200">
      {children}
    </code>
  )
}
