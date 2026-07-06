import { useEffect } from 'react'
import { AlertCircle, ExternalLink, X } from 'lucide-react'

// Fallback dialog shown at Build-click time when the queue contains files
// that need Image Builder (any image, or a PDF classified as scanned) AND
// Image Builder is not detected on 127.0.0.1:7879. Follows the three-choice
// pattern from the Day 2 spec Q5:
//   [Skip these files and build anyway]       → strip amber files, proceed
//   [Cancel — I'll start Image Builder]       → close, queue preserved
//   [How do I start Image Builder?]           → opens Image Builder repo
//
// Style-parity with DesktopHelperPairModal — same modal shell + Escape +
// backdrop-click close. Amber accent (vs purple pair modal) so the two
// dialogs read as related but distinct (pairing vs missing exe).

const IMAGE_BUILDER_HELP_URL =
  'https://github.com/project-you-apps/vector-plus-image-builder'

interface Props {
  open: boolean
  affectedFiles: string[]           // filenames that would go to Image Builder
  onSkip: () => void                // strip affected files, proceed with build
  onCancel: () => void              // close, don't build; queue preserved
}

export default function ImageBuilderMissingDialog({
  open,
  affectedFiles,
  onSkip,
  onCancel,
}: Props) {
  // Close on Escape — matches DesktopHelperPairModal / BriefingModal keyboard
  // parity. Backdrop click routes through onCancel too.
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onCancel])

  if (!open) return null

  const openHelp = () => {
    // Open README in a new tab. Not the whole dialog dismissal path — user
    // may want to return here after reading, so we leave it open.
    window.open(IMAGE_BUILDER_HELP_URL, '_blank', 'noopener,noreferrer')
  }

  const affectedCount = affectedFiles.length

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Image Builder not running"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div
        className="relative w-full max-w-md rounded-2xl border border-amber-500/40 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3 border-b border-slate-700/40 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md flex items-center justify-center bg-amber-500/20 border border-amber-500/40">
              <AlertCircle size={14} className="text-amber-300" />
            </div>
            <h2 className="text-sm font-medium text-slate-200">
              {affectedCount === 1 ? '1 file needs' : `${affectedCount} files need`} Image Builder
            </h2>
          </div>
          <button
            onClick={onCancel}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-500 hover:text-slate-300"
            aria-label="Close"
            title="Close (Esc)"
          >
            <X size={14} />
          </button>
        </div>

        <div className="px-5 py-5 space-y-4">
          <div className="text-xs text-slate-400 leading-relaxed space-y-2">
            <p>
              These files would be sent to <span className="text-slate-200 font-medium">Vector+ Image Builder</span> for
              on-device OCR (images and scanned PDFs). Image Builder isn&rsquo;t
              running on <span className="font-mono text-slate-300">127.0.0.1:7879</span> right now.
            </p>
          </div>

          {/* Affected files list — capped at 5 with a "+N more" tail so a big
              drop doesn't push the buttons off-screen. Same truncation shape
              as BrowserCartBuilder's binary-file skip notice. */}
          <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2 text-xs text-slate-300 font-mono space-y-0.5">
            {affectedFiles.slice(0, 5).map((name) => (
              <div key={name} className="truncate" title={name}>• {name}</div>
            ))}
            {affectedFiles.length > 5 && (
              <div className="text-slate-500 italic">
                + {affectedFiles.length - 5} more
              </div>
            )}
          </div>

          <div className="space-y-2">
            <button
              onClick={onSkip}
              className="w-full px-3 py-2 rounded-lg bg-slate-800/70 border border-slate-700/60 text-slate-200 hover:bg-slate-700/60 text-xs font-medium transition-colors text-left"
            >
              Skip these files and build anyway
              <div className="text-[10px] text-slate-500 font-normal mt-0.5 normal-case">
                Build the rest of the queue; skipped files aren&rsquo;t in the cart.
              </div>
            </button>
            <button
              onClick={onCancel}
              className="w-full px-3 py-2 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-200 hover:bg-amber-500/30 text-xs font-medium transition-colors text-left"
            >
              Cancel — I&rsquo;ll start Image Builder
              <div className="text-[10px] text-amber-300/70 font-normal mt-0.5 normal-case">
                Queue preserved; re-click Build after the exe reports running.
              </div>
            </button>
            <button
              onClick={openHelp}
              className="w-full px-3 py-2 rounded-lg bg-slate-900/60 border border-slate-700/60 text-slate-300 hover:bg-slate-800/60 text-xs font-medium transition-colors text-left flex items-center gap-2"
            >
              <ExternalLink size={12} className="text-slate-500 shrink-0" />
              <span className="flex-1">
                How do I start Image Builder?
                <span className="block text-[10px] text-slate-500 font-normal mt-0.5 normal-case">
                  Opens the README in a new tab.
                </span>
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
