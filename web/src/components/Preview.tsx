import { useMemo } from 'react'
import { Image as ImageIcon, Download } from 'lucide-react'
import { Card, Pill, Disclosure } from './ui'
import type { GenerateResponse } from '../lib/api'

export function Preview({
  result,
  loading,
  elapsed,
  initialImage,
}: {
  result: GenerateResponse | null
  loading: boolean
  elapsed: number
  initialImage: string | null
}) {
  const imgSrc = result?.image_url ?? (loading ? null : initialImage)
  const videoSrc = result?.video_url ?? null
  const isVideo = !!videoSrc

  return (
    <div className="space-y-4">
      <Card className="p-4">
        <div className="relative rounded-lg overflow-hidden bg-sub border border-line">
          <div className="aspect-square sm:aspect-auto sm:min-h-[480px] lg:min-h-[560px] grid place-items-center">
            {loading ? (
              <LoadingState elapsed={elapsed} />
            ) : isVideo ? (
              <video
                key={videoSrc}
                src={videoSrc}
                controls
                autoPlay
                loop
                className="max-h-[640px] max-w-full"
              />
            ) : imgSrc ? (
              <img
                key={imgSrc}
                src={imgSrc}
                alt="generated"
                className="max-h-[640px] max-w-full object-contain"
              />
            ) : (
              <EmptyState />
            )}
          </div>
          {imgSrc && !loading && (
            <a
              href={imgSrc}
              download
              className="absolute top-3 right-3 inline-flex items-center justify-center h-8 w-8 rounded-md bg-card/90 border border-line text-muted hover:text-ink backdrop-blur"
              title="Download"
            >
              <Download className="h-4 w-4" />
            </a>
          )}
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-center gap-1.5">
          {loading ? (
            <Pill tone="terra">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-white pulse-dot" />
              Generating · {elapsed.toFixed(1)}s
            </Pill>
          ) : result ? (
            <>
              <Pill tone="terra">{result.elapsed_sec.toFixed(1)}s</Pill>
              <Pill>{result.summary.pn} · {result.summary.steps} steps</Pill>
              <Pill>seed {result.summary.seed}</Pill>
              <Pill>{result.summary.task}</Pill>
            </>
          ) : initialImage ? (
            <>
              <Pill>last result</Pill>
              <Pill>hit Generate for a new one</Pill>
            </>
          ) : (
            <Pill>Ready</Pill>
          )}
        </div>
      </Card>

      {result && (
        <Disclosure title="Run details">
          <RunDetails result={result} />
        </Disclosure>
      )}
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center text-soft p-12">
      <ImageIcon className="h-10 w-10 mb-3 opacity-40" />
      <p className="text-sm">Your generation will appear here</p>
    </div>
  )
}

function LoadingState({ elapsed }: { elapsed: number }) {
  return (
    <div className="flex flex-col items-center text-muted p-12 gap-3">
      <div className="spin h-8 w-8 rounded-full border-[3px] border-line border-t-terra" />
      <p className="text-sm">Generating · <span className="font-mono">{elapsed.toFixed(1)}s</span></p>
    </div>
  )
}

function RunDetails({ result }: { result: GenerateResponse }) {
  const json = useMemo(() => JSON.stringify(result.summary, null, 2), [result.summary])
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-2xs font-semibold uppercase tracking-cap text-muted mb-2">Timings</h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(result.summary.timings).map(([k, v]) => (
            <div key={k} className="px-3 py-2 rounded-md border border-line bg-sub">
              <div className="text-2xs text-muted">{k.replace(/_sec$/, '').replace(/_/g, ' ')}</div>
              <div className="text-sm font-mono text-ink">{v.toFixed(2)}s</div>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h4 className="text-2xs font-semibold uppercase tracking-cap text-muted mb-2">Summary</h4>
        <pre className="rounded-md bg-pre text-bg p-4 text-2xs font-mono overflow-x-auto">{json}</pre>
      </div>
    </div>
  )
}
