import { useEffect, useState } from 'react'
import { RefreshCw, Image as ImageIcon, Film, Loader2 } from 'lucide-react'
import { Button, Card } from './ui'
import { Lightbox, type LightboxItem } from './Lightbox'
import { api, type HistoryItem } from '../lib/api'

export function Gallery() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [active, setActive] = useState<LightboxItem | null>(null)

  const load = async () => {
    setRefreshing(true)
    try {
      const data = await api.history()
      setItems(data)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <div className="mb-5 flex items-center justify-between">
        <p className="text-sm text-muted">Recent files in <code className="font-mono text-xs px-1.5 py-0.5 rounded bg-sub border border-line text-ink">outputs/</code>, newest first.</p>
        <Button variant="secondary" onClick={load} disabled={refreshing}>
          <RefreshCw className={refreshing ? 'h-4 w-4 spin' : 'h-4 w-4'} />
          Refresh
        </Button>
      </div>

      {items === null ? (
        <div className="grid place-items-center py-24 text-muted">
          <Loader2 className="h-6 w-6 spin" />
        </div>
      ) : items.length === 0 ? (
        <Card className="p-12">
          <div className="grid place-items-center text-soft">
            <ImageIcon className="h-10 w-10 mb-3 opacity-40" />
            <p className="text-sm">Nothing in <code className="font-mono">outputs/</code> yet — generate something first.</p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3">
          {items.map((it) => (
            <button
              key={it.url}
              onClick={() => setActive({ url: it.url, filename: it.filename, type: it.type })}
              className="group relative aspect-square rounded-lg border border-line bg-sub overflow-hidden hover:border-terra/50 hover:shadow-soft transition-all focus-visible:outline-none focus-visible:ring focus-visible:ring-terra/30"
            >
              {it.type === 'video' ? (
                <video src={it.url} className="h-full w-full object-cover" muted />
              ) : (
                <img src={it.url} alt={it.filename} loading="lazy" className="h-full w-full object-cover" />
              )}
              <div className="absolute inset-x-0 bottom-0 px-2.5 py-1.5 bg-gradient-to-t from-pre/80 to-transparent text-bg text-2xs font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="truncate flex items-center gap-1">
                  {it.type === 'video' && <Film className="h-3 w-3" />}
                  {it.filename}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      <Lightbox item={active} onClose={() => setActive(null)} />
    </div>
  )
}
