import { useEffect, useMemo, useRef, useState } from 'react'
import { Sparkles } from 'lucide-react'
import { Header, type ViewKey } from './components/Header'
import { PromptPanel } from './components/PromptPanel'
import { Controls, type ControlsState } from './components/Controls'
import { Advanced, defaultAdvanced, type AdvancedState } from './components/Advanced'
import { Preview } from './components/Preview'
import { Gallery } from './components/Gallery'
import { About } from './components/About'
import { Button, Toast } from './components/ui'
import { api, type GenerateResponse, type PresetMeta } from './lib/api'

export default function App() {
  const [view, setView] = useState<ViewKey>('create')
  const [presets, setPresets] = useState<PresetMeta | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [prompt, setPrompt] = useState('')
  const [controls, setControls] = useState<ControlsState>({
    task: 'T2I',
    quality: 'Balanced',
    aspect: '1:1 Square',
    seed: 42,
    duration: 0.5,
  })
  const [adv, setAdv] = useState<AdvancedState | null>(null)

  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const tickRef = useRef<number | null>(null)

  // Initial load: presets + last image
  useEffect(() => {
    api.presets()
      .then((p) => {
        setPresets(p)
        setPrompt(p.examples[0] ?? '')
        setAdv(defaultAdvanced(p))
      })
      .catch((e) => setError(`Backend not reachable: ${e.message}`))
  }, [])

  // Live elapsed timer while generating
  useEffect(() => {
    if (loading) {
      const start = performance.now()
      tickRef.current = window.setInterval(() => {
        setElapsed((performance.now() - start) / 1000)
      }, 100)
      return () => { if (tickRef.current) window.clearInterval(tickRef.current) }
    } else {
      setElapsed(0)
    }
  }, [loading])

  const initialImage = useMemo(() => api.fileUrl('outputs/latest_t2i.png') + `?_=${Date.now()}`, [])

  const generate = async () => {
    if (!presets || !adv) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.generate({
        task: controls.task,
        prompt,
        quality: controls.quality,
        aspect: controls.aspect,
        seed: controls.seed,
        duration: controls.duration,
        ...adv,
      })
      setResult(res)
    } catch (e: any) {
      setError(`Generation failed: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const updateControls = (next: Partial<ControlsState>) => setControls((s) => ({ ...s, ...next }))
  const updateAdv = (next: Partial<AdvancedState>) => setAdv((s) => (s ? { ...s, ...next } : s))

  if (!presets || !adv) {
    return (
      <div className="min-h-screen grid place-items-center text-muted">
        <div className="flex items-center gap-3">
          <span className="spin h-5 w-5 rounded-full border-2 border-line border-t-terra" />
          <span className="text-sm">Loading…</span>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-page px-5 sm:px-8 lg:px-10 py-8 lg:py-10 pb-20">
      <Header view={view} onView={setView} />

      {view === 'create' && (
        <main
          className="
            grid gap-6
            grid-cols-1
            lg:grid-cols-[minmax(380px,440px)_minmax(0,1fr)]
            xl:gap-8
            items-start
          "
        >
          <section className="space-y-4">
            <PromptPanel
              value={prompt}
              onChange={setPrompt}
              examples={presets.examples}
              onPick={setPrompt}
            />
            <Controls state={controls} setState={updateControls} presets={presets} />
            <Button
              variant="primary"
              onClick={generate}
              loading={loading}
              className="w-full h-12 text-[0.95rem]"
            >
              <Sparkles className="h-4 w-4" />
              {loading ? 'Generating…' : 'Generate'}
            </Button>
            <Advanced state={adv} setState={updateAdv} />
          </section>
          <section className="lg:sticky lg:top-6">
            <Preview
              result={result}
              loading={loading}
              elapsed={elapsed}
              initialImage={result ? null : initialImage}
            />
          </section>
        </main>
      )}

      {view === 'gallery' && <Gallery />}
      {view === 'about' && <About />}

      {error && <Toast message={error} onClose={() => setError(null)} />}
    </div>
  )
}
