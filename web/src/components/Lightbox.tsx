import { useEffect } from 'react'
import { X, Download, ExternalLink } from 'lucide-react'

export interface LightboxItem {
  url: string
  filename: string
  type: 'image' | 'video' | 'gif'
}

export function Lightbox({ item, onClose }: { item: LightboxItem | null; onClose: () => void }) {
  useEffect(() => {
    if (!item) return
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [item, onClose])

  if (!item) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      className="fixed inset-0 z-40 flex flex-col items-center justify-center bg-pre/85 backdrop-blur-sm animate-fade-in"
    >
      <div
        className="relative max-w-[92vw] max-h-[88vh] flex flex-col items-center"
        onClick={(e) => e.stopPropagation()}
      >
        {item.type === 'video' ? (
          <video src={item.url} controls autoPlay loop className="max-h-[78vh] max-w-[92vw] rounded-lg shadow-2xl" />
        ) : (
          <img src={item.url} alt={item.filename} className="max-h-[78vh] max-w-[92vw] rounded-lg shadow-2xl object-contain" />
        )}
        <div className="mt-3 flex items-center gap-2">
          <span className="font-mono text-xs text-bg/80">{item.filename}</span>
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center h-7 w-7 rounded text-bg/80 hover:text-bg hover:bg-white/10"
            title="Open in new tab"
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
          <a
            href={item.url}
            download={item.filename}
            className="inline-flex items-center justify-center h-7 w-7 rounded text-bg/80 hover:text-bg hover:bg-white/10"
            title="Download"
          >
            <Download className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
      <button
        onClick={onClose}
        aria-label="Close"
        className="absolute top-5 right-5 inline-flex items-center justify-center h-9 w-9 rounded-full bg-white/10 hover:bg-white/20 text-bg"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
